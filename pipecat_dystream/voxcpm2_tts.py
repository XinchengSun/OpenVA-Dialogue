"""Pipecat TTS adapter for the local PCM speech bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from typing import Any

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.transcriptions.language import Language


Connector = Callable[[str], Any]
logger = logging.getLogger(__name__)


def _default_connector(uri: str) -> Any:
    from websockets.asyncio.client import connect

    return connect(uri, max_size=None, ping_interval=20, ping_timeout=20)


class LocalPCMTTSService(TTSService):
    """Stream true-rate PCM16 from a localhost speech bridge."""

    def __init__(
        self,
        *,
        uri: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        connector: Connector | None = None,
        **kwargs: Any,
    ) -> None:
        generic_uri = os.getenv("PIPECAT_TTS_BRIDGE_URI", "").strip()
        legacy_vox_uri = os.getenv("VOXCPM2_BRIDGE_URI", "").strip()
        default_model = "VoxCPM2" if legacy_vox_uri and not generic_uri else "fishaudio/s2-pro"
        default_voice = "cloned" if default_model == "VoxCPM2" else "zero-shot-cloned"
        self._model = model or os.getenv("PIPECAT_TTS_MODEL", "").strip() or default_model
        self._voice = voice or os.getenv("PIPECAT_TTS_VOICE", "").strip() or default_voice
        super().__init__(
            # RecoveringOpenAILLMService already emits one complete sentence
            # per TextFrame. Re-aggregating here makes Pipecat 1.6 wait for the
            # first character of the next sentence before it confirms the
            # current boundary, adding up to hundreds of milliseconds.
            text_aggregation_mode=TextAggregationMode.TOKEN,
            push_text_frames=False,
            # Pipecat owns one audio context for the entire LLM turn. Sentence
            # requests yield only PCM, avoiding a listener transition between
            # two sentences from the same reply.
            push_start_frame=True,
            push_stop_frames=True,
            reuse_context_id_within_turn=True,
            stop_frame_timeout_s=float(
                os.getenv("PIPECAT_TTS_STOP_FRAME_TIMEOUT_SEC", "10.0")
            ),
            settings=TTSSettings(
                model=self._model,
                voice=self._voice,
                language=Language.ZH,
            ),
            **kwargs,
        )
        self._uri = (
            uri
            or generic_uri
            or legacy_vox_uri
            or "ws://127.0.0.1:8771"
        )
        self._connector = connector or _default_connector
        self._release_timeout = float(
            os.getenv(
                "PIPECAT_TTS_RELEASE_TIMEOUT_SEC",
                os.getenv("VOXCPM2_RELEASE_TIMEOUT_SEC", "0.25"),
            )
        )
        self._cancel_timeout = float(
            os.getenv(
                "PIPECAT_TTS_CANCEL_SEND_TIMEOUT_SEC",
                os.getenv("VOXCPM2_CANCEL_SEND_TIMEOUT_SEC", "0.10"),
            )
        )
        if self._release_timeout <= 0:
            raise ValueError("TTS release timeout must be positive")
        if self._cancel_timeout <= 0:
            raise ValueError("TTS cancel timeout must be positive")
        self._active: dict[str, dict[str, Any]] = defaultdict(dict)
        self._cancelled_requests: set[str] = set()
        self._release_tasks: set[asyncio.Task[None]] = set()
        self._ready = False
        self._warmup_result: tuple[int, int] | None = None
        self._warmup_elapsed_ms: float | None = None

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def ready(self) -> bool:
        return self._ready

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "model": self._model,
            "voice": self._voice,
            "bridge_uri": self._uri,
            "sample_rate": self.sample_rate,
            "active_requests": sum(len(items) for items in self._active.values()),
            "warmup_elapsed_ms": self._warmup_elapsed_ms,
        }

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Fail startup before accepting microphone traffic if the isolated
        # model process is missing or still loading.
        await self.wait_ready()

    async def wait_ready(self, timeout: float | None = None) -> int:
        timeout_s = timeout or float(
            os.getenv(
                "PIPECAT_TTS_CONNECT_TIMEOUT_SEC",
                os.getenv("VOXCPM2_CONNECT_TIMEOUT_SEC", "30.0"),
            )
        )
        self._ready = False

        async def health_check() -> int:
            async with self._connector(self._uri) as websocket:
                await websocket.send(json.dumps({"type": "health"}))
                event = json.loads(await websocket.recv())
                sample_rate = int(event.get("sample_rate", 0))
                if event.get("type") != "health" or event.get("status") != "ok":
                    raise ConnectionError(f"TTS bridge is not ready: {event}")
                if sample_rate <= 0:
                    raise ConnectionError(
                        f"TTS bridge returned invalid sample rate: {event}"
                    )
                self._sample_rate = sample_rate
                self._ready = True
                return sample_rate

        return await asyncio.wait_for(health_check(), timeout=timeout_s)

    async def warmup(self, text: str, timeout: float) -> tuple[int, int]:
        """Exercise the real synthesis path without publishing any audio frames."""

        warmup_text = text.strip()
        if not warmup_text:
            return (0, 0)
        if self._warmup_result is not None:
            return self._warmup_result
        if timeout <= 0:
            raise ValueError("TTS warmup timeout must be positive")

        context_id = f"__warmup__-{uuid.uuid4().hex}"
        started_at = asyncio.get_running_loop().time()

        async def consume() -> tuple[int, int]:
            chunks = 0
            pcm_bytes = 0
            async for frame in self.run_tts(warmup_text, context_id):
                if isinstance(frame, ErrorFrame):
                    raise RuntimeError(frame.error)
                if isinstance(frame, TTSAudioRawFrame):
                    chunks += 1
                    pcm_bytes += len(frame.audio)
            if pcm_bytes == 0:
                raise RuntimeError("TTS warmup returned no PCM audio")
            return chunks, pcm_bytes

        try:
            result = await asyncio.wait_for(consume(), timeout=timeout)
        finally:
            await self._release_context(context_id, "warmup")
        self._warmup_result = result
        self._warmup_elapsed_ms = (
            asyncio.get_running_loop().time() - started_at
        ) * 1000.0
        logger.info(
            "TTS warmup complete: elapsed_ms=%.1f chunks=%d pcm_bytes=%d",
            self._warmup_elapsed_ms,
            result[0],
            result[1],
        )
        return result

    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None, None]:
        request_id = uuid.uuid4().hex
        metadata_received = False
        audio_received = False
        try:
            async with self._connector(self._uri) as websocket:
                self._active[context_id][request_id] = websocket
                await websocket.send(
                    json.dumps(
                        {
                            "type": "synthesize",
                            "request_id": request_id,
                            "context_id": context_id,
                            "text": text,
                        }
                    )
                )
                async for message in websocket:
                    # After cancellation, late PCM must be dropped, but control
                    # frames still need to be consumed so ``done(cancelled)``
                    # can terminate this generator and release the connection.
                    if request_id in self._cancelled_requests and isinstance(message, bytes):
                        continue
                    if isinstance(message, bytes):
                        if not metadata_received:
                            raise RuntimeError("TTS audio arrived before start metadata")
                        if not message or len(message) % 2:
                            raise RuntimeError("TTS bridge returned invalid PCM16 audio")
                        audio_received = True
                        yield TTSAudioRawFrame(
                            audio=message,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                        continue

                    event = json.loads(message)
                    if str(event.get("request_id", "")) not in {"", request_id}:
                        raise RuntimeError(f"foreign TTS request event: {event}")
                    if str(event.get("context_id", "")) not in {"", context_id}:
                        raise RuntimeError(f"foreign TTS context event: {event}")
                    event_type = event.get("type")
                    if event_type == "start":
                        if metadata_received:
                            raise RuntimeError(f"duplicate TTS start metadata: {event}")
                        sample_rate = int(event["sample_rate"])
                        if (
                            sample_rate <= 0
                            or int(event.get("channels", 0)) != 1
                            or int(event.get("sample_width", 0)) != 2
                        ):
                            raise RuntimeError(f"invalid TTS stream metadata: {event}")
                        self._sample_rate = sample_rate
                        metadata_received = True
                        # A previous request may have failed transiently while
                        # the bridge stayed healthy. Valid wire metadata is a
                        # positive liveness signal, so do not permanently lock
                        # this adapter in an unready state.
                        self._ready = True
                    elif event_type == "done":
                        if not metadata_received:
                            raise RuntimeError(
                                "TTS done arrived before start metadata"
                            )
                        status = str(event.get("status", ""))
                        if status not in {"completed", "cancelled"}:
                            raise RuntimeError(
                                f"invalid TTS completion status: {event}"
                            )
                        if (
                            status == "cancelled"
                            and request_id not in self._cancelled_requests
                        ):
                            raise RuntimeError(
                                f"unexpected TTS cancellation: {event}"
                            )
                        if status == "completed" and not audio_received:
                            raise RuntimeError("TTS completed without PCM audio")
                        self._ready = True
                        return
                    elif event_type == "error":
                        raise RuntimeError(str(event.get("error", "unknown bridge error")))
                    else:
                        raise RuntimeError(f"unknown TTS bridge event: {event}")
                raise ConnectionError("TTS bridge closed before done")
        except Exception as exc:
            if request_id not in self._cancelled_requests:
                self._ready = False
                yield ErrorFrame(error=f"TTS synthesis failed: {exc}")
        finally:
            requests = self._active.get(context_id)
            if requests is not None:
                requests.pop(request_id, None)
                if not requests:
                    self._active.pop(context_id, None)
            self._cancelled_requests.discard(request_id)

    async def on_audio_context_interrupted(self, context_id: str):
        requests = list(self._active.get(context_id, {}).items())
        for request_id, websocket in requests:
            self._cancelled_requests.add(request_id)
            self._schedule_control_task(
                self._send_cancel(websocket, request_id, context_id)
            )
        # Pipecat has already stopped the old audio-context task, but it waits
        # for this hook before it creates the next one.  Context disposal is a
        # control-plane operation and must therefore not sit on the barge-in
        # critical path.
        self._schedule_context_release(context_id, "interrupted")
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        # The serialization queue also waits for this hook before it can move
        # to a new context. Release out of band so a local WebSocket timeout
        # can never become a sentence/turn gap.
        self._schedule_context_release(context_id, "completed")
        await super().on_audio_context_completed(context_id)

    def _schedule_context_release(self, context_id: str, reason: str) -> None:
        if not context_id:
            return
        self._schedule_control_task(self._release_context(context_id, reason))

    def _schedule_control_task(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._release_tasks.add(task)
        task.add_done_callback(self._release_tasks.discard)

    async def _send_cancel(
        self,
        websocket: Any,
        request_id: str,
        context_id: str,
    ) -> None:
        try:
            async with asyncio.timeout(self._cancel_timeout):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "cancel",
                            "request_id": request_id,
                            "context_id": context_id,
                            "discard_context": True,
                        }
                    )
                )
        except Exception as exc:
            logger.warning(
                "failed to cancel TTS request %s in context %s: %s",
                request_id,
                context_id,
                exc,
            )

    async def _release_context(self, context_id: str, reason: str) -> None:
        if not context_id:
            return
        last_error: Exception | None = None
        for _ in range(2):
            try:
                async with asyncio.timeout(self._release_timeout):
                    async with self._connector(self._uri) as websocket:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "release_context",
                                    "context_id": context_id,
                                    "reason": reason,
                                }
                            )
                        )
                        event = json.loads(await websocket.recv())
                        if (
                            event.get("type") != "released"
                            or str(event.get("context_id", "")) != context_id
                        ):
                            raise RuntimeError(
                                f"invalid context release response: {event}"
                            )
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0)
        logger.warning(
            "failed to release TTS context %s (%s) after retry: %s",
            context_id,
            reason,
            last_error,
        )


# Backward-compatible import for deployments and tests that still use the old
# provider-specific class name. The wire adapter itself is provider-neutral.
VoxCPM2LocalTTSService = LocalPCMTTSService
