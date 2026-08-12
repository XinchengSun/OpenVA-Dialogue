"""Frame processors that connect Pipecat to the existing DyStream engine."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    LLMFullResponseEndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .audio import TARGET_SAMPLE_RATE, float32_to_pcm16le, pcm16le_to_float32_mono


EmitFrame = Callable[[Any], Awaitable[None]]


@dataclass(frozen=True)
class _QueuedSegment:
    epoch: int
    context_id: str
    frames: np.ndarray
    audio: np.ndarray


class DyStreamOutputClient:
    """Thread-safe bridge from DyStream's worker thread to Pipecat's loop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        emit_frame: EmitFrame,
        *,
        client_id: str = "pipecat-webrtc",
        max_segments: int = 8,
    ):
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self.loop = loop
        self.client_id = client_id
        self._emit_frame = emit_frame
        self._max_segments = max_segments
        self._segments: deque[_QueuedSegment] = deque()
        self._lock = threading.Lock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._running = False
        self._epoch = 0
        self._context_id: str | None = None
        self._received_segments = 0
        self._expected_segments: int | None = None
        self._stop_frame: TTSStoppedFrame | None = None
        self._finish_handle: asyncio.TimerHandle | None = None
        self._turn_started_at = 0.0
        self.dropped_segments = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = self.loop.create_task(
            self._publish_loop(), name="dystream-pipecat-publisher"
        )

    def stop(self):
        self._running = False
        self.interrupt()

        def cancel_task():
            self._wake.set()
            if self._task and not self._task.done():
                self._task.cancel()

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.loop:
            cancel_task()
        elif not self.loop.is_closed():
            self.loop.call_soon_threadsafe(cancel_task)

    async def wait_stopped(self):
        task = self._task
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            pass

    def begin_turn(self, context_id: str):
        self._cancel_finish_timeout()
        with self._lock:
            self._epoch += 1
            self._segments.clear()
            self._context_id = context_id
            self._received_segments = 0
            self._expected_segments = None
            self._stop_frame = None
            self._turn_started_at = time.monotonic()
        self._notify_loop()

    def finish_turn(
        self,
        stop_frame: TTSStoppedFrame,
        *,
        expected_segments: int,
        timeout_seconds: float = 10.0,
    ):
        """Emit TTSStopped only after the final generated media segment."""
        with self._lock:
            if self._context_id is None:
                return
            self._stop_frame = stop_frame
            self._expected_segments = max(0, expected_segments)
        self._cancel_finish_timeout()
        self._finish_handle = self.loop.call_later(timeout_seconds, self._force_finish)
        self._notify_loop()

    def interrupt(self) -> int:
        self._cancel_finish_timeout()
        with self._lock:
            dropped = len(self._segments)
            self._epoch += 1
            self._segments.clear()
            self._context_id = None
            self._received_segments = 0
            self._expected_segments = None
            self._stop_frame = None
        self._notify_loop()
        return dropped

    def clear_backlog(self) -> int:
        """Invalidate pending and in-flight segments without ending the turn."""
        with self._lock:
            dropped = len(self._segments)
            self._epoch += 1
            self._segments.clear()
        self._notify_loop()
        return dropped

    def request_stream_reset(self) -> int:
        return self.clear_backlog()

    def put_text(self, _text: str):
        # Engine log fan-out is optional for this transport.
        return None

    def push_segment(self, frames: np.ndarray, audio: np.ndarray) -> bool:
        """Queue one aligned DyStream segment from its feed thread."""
        frames_array = np.asarray(frames, dtype=np.uint8)
        audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if frames_array.ndim != 4 or frames_array.shape[-1] != 3:
            logger.error(f"Invalid DyStream frame segment shape: {frames_array.shape}")
            return False

        expected_samples = frames_array.shape[0] * 640
        if audio_array.size != expected_samples:
            logger.warning(
                f"DyStream segment audio mismatch: samples={audio_array.size} "
                f"expected={expected_samples}; padding/trimming"
            )
            if audio_array.size < expected_samples:
                audio_array = np.pad(audio_array, (0, expected_samples - audio_array.size))
            else:
                audio_array = audio_array[:expected_samples]

        with self._lock:
            context_id = self._context_id
            if not self._running or context_id is None:
                return False
            if len(self._segments) >= self._max_segments:
                self._segments.popleft()
                self.dropped_segments += 1
            first_segment = self._received_segments == 0
            self._segments.append(
                _QueuedSegment(
                    epoch=self._epoch,
                    context_id=context_id,
                    frames=frames_array.copy(),
                    audio=audio_array.copy(),
                )
            )
            self._received_segments += 1
            first_segment_ms = (time.monotonic() - self._turn_started_at) * 1000.0
        if first_segment:
            logger.info(
                f"[PIPELINE LATENCY] context={context_id} "
                f"tts_start_to_dystream_first_segment_ms={first_segment_ms:.1f}"
            )
        self._notify_loop()
        return True

    def _cancel_finish_timeout(self):
        handle = self._finish_handle
        self._finish_handle = None
        if handle is not None:
            handle.cancel()

    def _force_finish(self):
        with self._lock:
            if self._stop_frame is not None:
                self._expected_segments = self._received_segments
        self._notify_loop()

    def _take_ready_stop_frame(self) -> TTSStoppedFrame | None:
        with self._lock:
            ready = (
                self._stop_frame is not None
                and self._expected_segments is not None
                and self._received_segments >= self._expected_segments
                and not self._segments
            )
            if not ready:
                return None
            frame = self._stop_frame
            self._stop_frame = None
            self._expected_segments = None
            self._context_id = None
        self._cancel_finish_timeout()
        return frame

    def _notify_loop(self):
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._wake.set)

    def _pop_segment(self) -> _QueuedSegment | None:
        with self._lock:
            return self._segments.popleft() if self._segments else None

    def _is_current(self, segment: _QueuedSegment) -> bool:
        with self._lock:
            return (
                self._running
                and segment.epoch == self._epoch
                and segment.context_id == self._context_id
            )

    def _has_segments(self) -> bool:
        with self._lock:
            return bool(self._segments)

    async def _publish_loop(self):
        while self._running:
            await self._wake.wait()
            while self._running:
                segment = self._pop_segment()
                if segment is None:
                    stop_frame = self._take_ready_stop_frame()
                    if stop_frame is not None:
                        await self._emit_frame(stop_frame)
                        continue
                    self._wake.clear()
                    if self._has_segments():
                        self._wake.set()
                        continue
                    break
                await self._emit_segment(segment)

    async def _emit_segment(self, segment: _QueuedSegment):
        for index, frame in enumerate(segment.frames):
            if not self._is_current(segment):
                return
            height, width, _ = frame.shape
            image = OutputImageRawFrame(
                image=np.ascontiguousarray(frame).tobytes(),
                size=(width, height),
                format="RGB",
            )
            # The image goes through Pipecat's audio queue immediately before
            # its matching 40 ms audio slice.
            image.sync_with_audio = True
            await self._emit_frame(image)

            start = index * 640
            audio_slice = segment.audio[start : start + 640]
            audio_frame = TTSAudioRawFrame(
                audio=float32_to_pcm16le(audio_slice),
                sample_rate=TARGET_SAMPLE_RATE,
                num_channels=1,
                context_id=segment.context_id,
            )
            await self._emit_frame(audio_frame)


class DyStreamUserAudioTap(FrameProcessor):
    """Copy browser microphone PCM into DyStream's official audio_other path."""

    def __init__(self, engine: Any, **kwargs):
        super().__init__(**kwargs)
        self._engine = engine

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            audio = pcm16le_to_float32_mono(
                frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            self._engine.enqueue_user_audio(float32_to_pcm16le(audio))
        await self.push_frame(frame, direction)


class DyStreamAvatarProcessor(FrameProcessor):
    """Consume TTS audio and publish DyStream-aligned audio/video frames."""

    def __init__(self, engine: Any, *, max_segments: int = 8, **kwargs):
        super().__init__(**kwargs)
        self._engine = engine
        self._max_segments = max_segments
        self._client: DyStreamOutputClient | None = None
        self._engine_turn_open = False
        self._turn_sequence = 0
        self._tts_samples = 0
        self._tts_started_at = 0.0
        self._first_tts_audio_seen = False

    @property
    def output_client(self) -> DyStreamOutputClient | None:
        return self._client

    def _register_client(self):
        if self._client is not None:
            return
        self._client = DyStreamOutputClient(
            self.get_event_loop(),
            self.push_frame,
            max_segments=self._max_segments,
        )
        self._engine.register_output_client(self._client)

    async def _unregister_client(self):
        client = self._client
        if client is None:
            return
        self._client = None
        self._engine.unregister_output_client(client)
        await client.wait_stopped()

    def _begin_turn(self, context_id: str | None):
        if self._client is None:
            raise RuntimeError("DyStream output client has not been started")
        if self._engine_turn_open:
            self._engine.interrupt_assistant()
        self._turn_sequence += 1
        resolved_context = context_id or f"dystream-turn-{self._turn_sequence}"
        self._client.begin_turn(resolved_context)
        self._engine.begin_assistant_turn()
        self._engine_turn_open = True
        self._tts_samples = 0
        self._tts_started_at = time.monotonic()
        self._first_tts_audio_seen = False

    def _interrupt_turn(self):
        if not self._engine_turn_open:
            self._engine.cancel_prepared_assistant()
            return
        if self._client is not None:
            self._client.interrupt()
        self._engine.interrupt_assistant()
        self._engine_turn_open = False
        self._tts_samples = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._register_client()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if not self._engine_turn_open:
                self._begin_turn(frame.context_id)
            audio = pcm16le_to_float32_mono(
                frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            self._engine.enqueue_speaker_audio(audio)
            self._tts_samples += len(audio)
            if not self._first_tts_audio_seen:
                self._first_tts_audio_seen = True
                logger.info(
                    "[PIPELINE LATENCY] "
                    f"tts_started_to_first_audio_ms="
                    f"{(time.monotonic() - self._tts_started_at) * 1000.0:.1f}"
                )
            # Do not pass the original TTS audio to WebRTC. The output client
            # emits the same waveform paired with the generated face frames.
            return

        if isinstance(frame, TTSStartedFrame) and direction == FrameDirection.DOWNSTREAM:
            self._begin_turn(frame.context_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSStoppedFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._engine_turn_open and self._client is not None and self._tts_samples > 0:
                self._engine.end_assistant_turn()
                hop_samples = max(1, int(16_000 * self._engine.args.hop_ms / 1000))
                speaker_hops = (self._tts_samples + hop_samples - 1) // hop_samples
                tail_samples = int(float(os.getenv("ENGINE_TTS_TAIL_SEC", "0.4")) * 16_000)
                tail_hops = (tail_samples + hop_samples - 1) // hop_samples
                segment_frames, _ = self._engine._segment_frames_aligned_to_pipe_stride(
                    self._engine.args.segment_frames
                )
                expected_samples = (speaker_hops + tail_hops) * hop_samples
                expected_segments = expected_samples // (segment_frames * 640)
                self._client.finish_turn(
                    frame,
                    expected_segments=expected_segments,
                    timeout_seconds=float(os.getenv("PIPECAT_TURN_FINISH_TIMEOUT_SEC", "10")),
                )
            else:
                self._interrupt_turn()
                await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            # Overlap DyStream's generation reset with ASR/LLM instead of
            # paying that cost after the first TTS audio arrives.
            self._engine.prepare_assistant_turn()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame) and not self._engine_turn_open:
            # A provider error or an empty response produces no TTS frames.
            # Do not leave the avatar stuck in the preparation state.
            self._engine.cancel_prepared_assistant()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            self._interrupt_turn()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (CancelFrame, EndFrame)):
            self._interrupt_turn()
            await self._unregister_client()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def cleanup(self):
        self._interrupt_turn()
        await self._unregister_client()
        await super().cleanup()
