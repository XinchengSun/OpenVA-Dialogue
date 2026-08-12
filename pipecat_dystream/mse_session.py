"""Pipecat native S2S provider for the stable DyStream MSE runtime.

Pipecat owns realtime frame and interruption orchestration while one native
speech-to-speech service owns turn detection, language reasoning, and speech
generation.  The mature runtime keeps ownership of speaker buffering,
generation resets, dual-GPU inference, A/V alignment, ffmpeg/MSE publishing,
and browser live-tail recovery.
"""

from __future__ import annotations

import asyncio
import audioop
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from .qwen_audio_s2s import create_qwen_audio_s2s_from_env


TARGET_SAMPLE_RATE = 16_000


class _StatefulPCM16ToFloat16k:
    """Convert consecutive PCM chunks without resetting at chunk boundaries."""

    def __init__(self):
        self._rate_state = None

    def reset(self):
        self._rate_state = None

    def decode(
        self,
        pcm16: bytes,
        *,
        sample_rate: int,
        num_channels: int,
    ) -> np.ndarray:
        if not pcm16:
            return np.zeros(0, dtype=np.float32)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if len(pcm16) % (2 * num_channels):
            raise ValueError("PCM byte length is not aligned to int16 channels")

        mono = pcm16
        if num_channels == 2:
            mono = audioop.tomono(pcm16, 2, 0.5, 0.5)
        elif num_channels > 2:
            samples = np.frombuffer(pcm16, dtype="<i2").reshape(-1, num_channels)
            mono = np.rint(samples.astype(np.float32).mean(axis=1)).astype("<i2").tobytes()

        if sample_rate != TARGET_SAMPLE_RATE:
            mono, self._rate_state = audioop.ratecv(
                mono,
                2,
                1,
                sample_rate,
                TARGET_SAMPLE_RATE,
                self._rate_state,
            )

        if not mono:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(mono, dtype="<i2").astype(np.float32) / 32768.0


class DyStreamEngineAudioAdapter:
    """Map one Pipecat TTS turn onto the stable engine callback contract."""

    def __init__(self, engine: Any, log: Callable[[str], None]):
        self._engine = engine
        self._log = log
        self._converter = _StatefulPCM16ToFloat16k()
        self._active = False
        self._turn_started_at = 0.0
        self._first_audio_seen = False
        self._samples = 0
        self._context_id: str | None = None
        self._engine_started = False
        self._engine_turn_id: int | None = None
        self._rejected_contexts: set[str] = set()
        self._awaiting_pipeline_interrupt = False
        self._pipeline_interrupt_seen = False

    @property
    def active(self) -> bool:
        return self._active

    def _reject_context(self, context_id: str | None):
        if context_id:
            self._rejected_contexts.add(context_id)
            if len(self._rejected_contexts) > 64:
                self._rejected_contexts.pop()

    def start_tts(self, context_id: str | None) -> bool:
        if self._active:
            return context_id == self._context_id
        if context_id is None:
            self._log("[PIPECAT PROVIDER] drop TTS start without context id")
            return False
        if context_id and context_id in self._rejected_contexts:
            self._log(f"[PIPECAT PROVIDER] drop cancelled TTS start context={context_id}")
            return False
        if self._awaiting_pipeline_interrupt and not self._pipeline_interrupt_seen:
            self._reject_context(context_id)
            self._log("[PIPECAT PROVIDER] drop TTS start queued before interruption barrier")
            return False
        self._converter.reset()
        self._active = True
        self._engine_started = False
        self._engine_turn_id = None
        self._context_id = context_id
        self._awaiting_pipeline_interrupt = False
        self._pipeline_interrupt_seen = False
        self._turn_started_at = time.monotonic()
        self._first_audio_seen = False
        self._samples = 0
        self._log(f"[PIPECAT PROVIDER] TTS turn started context={context_id}")
        return True

    def push_pcm(
        self,
        pcm16: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        context_id: str | None,
    ):
        if (
            not self._active
            or context_id != self._context_id
            or (context_id and context_id in self._rejected_contexts)
        ):
            self._log(f"[PIPECAT PROVIDER] drop audio outside active TTS context={context_id}")
            return
        audio = self._converter.decode(
            pcm16,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        if len(audio) == 0:
            return
        if not self._engine_started:
            self._engine_turn_id = self._engine.begin_assistant_turn()
            self._engine_started = True
        self._engine.enqueue_speaker_audio(audio)
        self._samples += len(audio)
        if not self._first_audio_seen:
            self._first_audio_seen = True
            first_audio_ms = (time.monotonic() - self._turn_started_at) * 1000.0
            rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            self._log(
                "[PIPECAT PROVIDER] first audio "
                f"context={context_id} engine_turn={self._engine_turn_id} "
                f"latency_ms={first_audio_ms:.1f} sample_rate={sample_rate} "
                f"samples_16k={len(audio)} rms={rms:.6f}"
            )

    def stop_tts(self, context_id: str | None):
        if not self._active or context_id != self._context_id:
            return
        engine_started = self._engine_started
        self._active = False
        self._engine_started = False
        self._engine_turn_id = None
        self._context_id = None
        if engine_started:
            self._engine.end_assistant_turn()
        self._log(
            f"[PIPECAT PROVIDER] TTS turn ended samples_16k={self._samples} "
            f"seconds={self._samples / TARGET_SAMPLE_RATE:.3f}"
        )

    def mark_external_interrupt(self):
        """Forget late TTS frames after the host already reset the engine."""
        self._reject_context(self._context_id)
        self._active = False
        self._engine_started = False
        self._engine_turn_id = None
        self._context_id = None
        self._samples = 0
        self._converter.reset()
        self._awaiting_pipeline_interrupt = True
        self._pipeline_interrupt_seen = False

    def mark_host_interrupt_applied(self):
        """Fence the matching Pipecat frame after the host reset synchronously."""
        self._pipeline_interrupt_seen = True

    def _engine_has_pending_output(self) -> bool:
        pending = getattr(self._engine, "assistant_output_pending", None)
        if pending is None:
            return False
        try:
            return bool(pending())
        except Exception as exc:
            self._log(
                "[PIPECAT PROVIDER WARN] failed to inspect pending assistant "
                f"output: {exc!r}"
            )
            return False

    def handle_pipeline_interrupt(self):
        """Reset the host when Pipecat VAD, rather than browser VAD, interrupts."""
        was_active = self._active
        engine_started = self._engine_started
        reset_already_seen = self._pipeline_interrupt_seen
        if was_active:
            self._reject_context(self._context_id)
            self._active = False
            self._engine_started = False
            self._engine_turn_id = None
            self._context_id = None
            self._samples = 0
            self._converter.reset()
        if (
            not reset_already_seen
            and (engine_started or self._engine_has_pending_output())
        ):
            self._engine.interrupt_assistant()
            self._log("[PIPECAT PROVIDER] pipeline interruption reset assistant")
        self._awaiting_pipeline_interrupt = True
        self._pipeline_interrupt_seen = True


class DyStreamMSEAudioSink(FrameProcessor):
    """Send TTS audio to the old engine; never publish media itself."""

    def __init__(self, adapter: DyStreamEngineAudioAdapter, **kwargs):
        super().__init__(**kwargs)
        self._adapter = adapter

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSStartedFrame):
                self._adapter.start_tts(frame.context_id)
            elif isinstance(frame, TTSAudioRawFrame):
                self._adapter.push_pcm(
                    frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    context_id=frame.context_id,
                )
            elif isinstance(frame, TTSStoppedFrame):
                self._adapter.stop_tts(frame.context_id)
            elif isinstance(frame, InterruptionFrame):
                self._adapter.handle_pipeline_interrupt()
            elif isinstance(frame, (CancelFrame, EndFrame)):
                self._adapter.handle_pipeline_interrupt()

        await self.push_frame(frame, direction)


class PipecatMSESession:
    """One process-lifetime Pipecat dialog session feeding the stable MSE host."""

    def __init__(self, engine: Any, log: Callable[[str], None]):
        self._engine = engine
        self._log = log
        self.closed = asyncio.Event()
        self.connected = asyncio.Event()
        self.task: asyncio.Task | None = None

        self._mode = os.getenv("PIPECAT_MSE_DIALOG_MODE", "native_s2s").strip().lower()
        if self._mode not in {"native_s2s", "custom_cascade"}:
            raise RuntimeError(
                "PIPECAT_MSE_DIALOG_MODE must be native_s2s or custom_cascade"
            )

        self._adapter = DyStreamEngineAudioAdapter(engine, log)
        self._s2s = None
        self._custom = None
        if self._mode == "native_s2s":
            self._s2s = create_qwen_audio_s2s_from_env(log)
            processors = [
                self._s2s,
                DyStreamMSEAudioSink(self._adapter),
            ]
            worker_name = "pipecat-mse-native-s2s-provider"
        else:
            # Lazy import is the rollback boundary: native_s2s can still start
            # when custom packages, credentials or the local bridge are absent.
            from .custom_cascade import create_custom_cascade_components

            self._custom = create_custom_cascade_components(log)
            processors = [
                self._custom.stt,
                self._custom.user_aggregator,
                self._custom.llm,
                self._custom.tts,
                DyStreamMSEAudioSink(self._adapter),
                self._custom.assistant_aggregator,
            ]
            worker_name = "pipecat-mse-custom-cascade"

        pipeline = Pipeline(processors)
        self.worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=TARGET_SAMPLE_RATE,
                audio_out_sample_rate=TARGET_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            enable_rtvi=False,
            idle_timeout_secs=None,
            name=worker_name,
        )
        self.runner = WorkerRunner(handle_sigint=False)

        @self.worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(_worker, _frame):
            self.connected.set()
            self._log(f"[PIPECAT MSE] pipeline started mode={self._mode}")

        @self.worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(_worker, _frame):
            self.closed.set()
            self._log("[PIPECAT PROVIDER] pipeline finished")

        @self.worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(_worker, frame):
            self._log(f"[PIPECAT PROVIDER ERROR] {frame}")

    def start(self):
        if self.task is None or self.task.done():
            self.closed.clear()
            self.connected.clear()
            self.task = asyncio.create_task(self._run(), name="pipecat-mse-session")

    async def _run(self):
        try:
            await self.runner.add_workers(self.worker)
            await self.runner.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log(f"[PIPECAT PROVIDER ERROR] session failed: {exc!r}")
        finally:
            self.closed.set()

    async def wait_ready(self, timeout: float | None = None):
        ready_timeout = timeout or float(os.getenv("PIPECAT_STARTUP_TIMEOUT_SEC", "45"))
        started_at = time.monotonic()
        await asyncio.wait_for(self.connected.wait(), timeout=ready_timeout)
        remaining = max(0.001, ready_timeout - (time.monotonic() - started_at))
        if self._mode == "native_s2s":
            await self._s2s.wait_ready(timeout=remaining)
        else:
            await self._custom.wait_ready(timeout=remaining)
        self._log(f"[PIPECAT MSE] pipeline and dialog services ready mode={self._mode}")

    async def send_audio(self, pcm16: bytes) -> bool:
        if not pcm16:
            return True
        if self.closed.is_set() or self.task is None or self.task.done():
            self._log("[PIPECAT PROVIDER WARN] audio received while pipeline is closed")
            return False
        await self.worker.queue_frame(
            InputAudioRawFrame(
                audio=pcm16,
                sample_rate=TARGET_SAMPLE_RATE,
                num_channels=1,
            )
        )
        return True

    async def interrupt(self):
        self._adapter.mark_external_interrupt()
        dropped = self._engine.interrupt_assistant()
        self._adapter.mark_host_interrupt_applied()
        if not self.closed.is_set():
            try:
                await self.worker.queue_frame(InterruptionFrame())
            except Exception as exc:
                self.connected.clear()
                self.closed.set()
                self._log(
                    "[PIPECAT PROVIDER ERROR] interruption enqueue failed; "
                    f"session closed: {exc!r}"
                )
                try:
                    await self.worker.cancel(reason="interruption enqueue failed")
                except Exception as cancel_exc:
                    self._log(
                        "[PIPECAT PROVIDER ERROR] worker cancel after interruption "
                        f"failure also failed: {cancel_exc!r}"
                    )
                raise
        return dropped

    async def close(self):
        if not self.closed.is_set():
            self._adapter.mark_external_interrupt()
            await self.worker.cancel(reason="MSE host shutdown")
        if self.task is not None:
            task_wait = asyncio.gather(self.task, return_exceptions=True)
            try:
                await asyncio.wait_for(asyncio.shield(task_wait), timeout=5.0)
            except asyncio.TimeoutError:
                self.task.cancel()
                await task_wait
        self.connected.clear()
        self.closed.set()

    def health_snapshot(self) -> dict[str, Any]:
        mode = getattr(self, "_mode", "native_s2s")
        if mode == "native_s2s":
            s2s = getattr(self, "_s2s", None)
            provider_ready = bool(getattr(s2s, "ready", False))
            provider_health = getattr(s2s, "health_snapshot", None)
            provider = provider_health() if provider_health is not None else {}
        else:
            custom = getattr(self, "_custom", None)
            provider_ready = bool(getattr(custom, "ready", False))
            provider_health = getattr(custom, "health_snapshot", None)
            provider = provider_health() if provider_health is not None else {}
        return {
            "ready": (
                self.connected.is_set()
                and not self.closed.is_set()
                and provider_ready
            ),
            "closed": self.closed.is_set(),
            "task_done": self.task.done() if self.task is not None else True,
            "tts_active": self._adapter.active,
            "mode": mode,
            "s2s_ready": provider_ready if mode == "native_s2s" else False,
            "s2s": provider if mode == "native_s2s" else {},
            "custom_cascade_ready": (
                provider_ready if mode == "custom_cascade" else False
            ),
            "custom_cascade": provider if mode == "custom_cascade" else {},
        }
