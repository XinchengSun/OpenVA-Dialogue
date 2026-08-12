"""Pipecat processor for Qwen-Audio 3.0 native realtime speech-to-speech."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


ConnectFactory = Callable[[str, dict[str, str]], Awaitable[Any]]
LogCallback = Callable[[str], None]

DEFAULT_MODEL = "qwen-audio-3.0-realtime-flash"
DEFAULT_VOICE = "longanqian"
COMPATIBILITY_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_INSTRUCTIONS = (
    "你是一个友好、自然的中文语音助手。始终使用中文完整回答用户的问题。"
    "简单问题可以回答一到两句，复杂问题按内容需要自然展开；不要为了追求"
    "首声速度截断答案。除非用户明确要求，不要使用 Markdown、列表或序号。"
)
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000


class QwenAudioRealtimeS2SProcessor(FrameProcessor):
    """Keep one Qwen-Audio realtime session behind a Pipecat frame boundary."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        turn_detection: str = "smart_turn",
        silence_duration_ms: int = 800,
        threshold: float = 0.5,
        instructions: str = DEFAULT_INSTRUCTIONS,
        max_history_turns: int = 10,
        microphone_queue_max_chunks: int = 3,
        microphone_send_timeout_sec: float = 1.0,
        log: LogCallback | None = None,
        connect_factory: ConnectFactory | None = None,
        connect_timeout_sec: float = 10.0,
        shutdown_timeout_sec: float = 1.0,
        reconnect_initial_sec: float = 0.25,
        reconnect_max_sec: float = 5.0,
        ping_interval_sec: float = 20.0,
        ping_timeout_sec: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not api_key:
            raise ValueError("Qwen-Audio realtime API key is required")
        if not base_url.startswith(("wss://", "ws://")):
            raise ValueError("Qwen-Audio realtime base_url must be a WebSocket URL")
        if not 1 <= max_history_turns <= 50:
            raise ValueError("max_history_turns must be between 1 and 50")
        if not 200 <= silence_duration_ms <= 6000:
            raise ValueError("silence_duration_ms must be between 200 and 6000")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1.0 and 1.0")
        if not 1 <= microphone_queue_max_chunks <= 512:
            raise ValueError("microphone_queue_max_chunks must be between 1 and 512")
        if microphone_send_timeout_sec <= 0:
            raise ValueError("microphone_send_timeout_sec must be positive")
        if shutdown_timeout_sec <= 0:
            raise ValueError("shutdown_timeout_sec must be positive")
        normalized_turn_detection = turn_detection.strip().lower()
        if normalized_turn_detection not in {"smart_turn", "server_vad"}:
            raise ValueError(
                "turn_detection must be smart_turn or server_vad; "
                "manual mode is not implemented"
            )
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._base_url = base_url
        self._turn_detection = normalized_turn_detection
        self._silence_duration_ms = silence_duration_ms
        self._threshold = threshold
        self._instructions = instructions
        self._max_history_turns = max_history_turns
        self._microphone_queue_max_chunks = microphone_queue_max_chunks
        self._microphone_send_timeout_sec = microphone_send_timeout_sec
        self._log = log or (lambda _message: None)
        self._connect_factory = connect_factory or self._default_connect
        self._connect_timeout_sec = connect_timeout_sec
        self._shutdown_timeout_sec = shutdown_timeout_sec
        self._reconnect_initial_sec = reconnect_initial_sec
        self._reconnect_max_sec = reconnect_max_sec
        self._ping_interval_sec = ping_interval_sec
        self._ping_timeout_sec = ping_timeout_sec

        self._ready = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._microphone_queue: asyncio.Queue[tuple[int, Any, bytes]] = (
            asyncio.Queue(maxsize=microphone_queue_max_chunks)
        )
        self._connection_task: asyncio.Task | None = None
        self._microphone_sender_task: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task | None = None
        self._auxiliary_tasks: set[asyncio.Task] = set()
        self._websocket: Any | None = None
        self._shutting_down = False
        self._connection_epoch = 0
        self._response_cancel_generation = 0
        self._response_epoch = 0
        self._response_id: str | None = None
        self._response_context_id: str | None = None
        self._response_started = False
        self._suppress_until_created = False
        self._new_turn_marker = False
        self._rejected_responses: set[tuple[int, str]] = set()
        self._terminal_response_order: deque[tuple[int, str]] = deque()
        self._interruption_emitted = False
        self._synthetic_response_counter = 0
        self._event_counter = 0
        self._speech_stopped_at = 0.0
        self._response_started_at = 0.0
        self._audio_done_at = 0.0
        self._first_audio_seen = False
        self._last_error: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self._ready.is_set()
            and self._websocket is not None
            and not self._shutting_down
        )

    async def wait_ready(self, timeout: float | None = None):
        timeout_sec = timeout
        if timeout_sec is None:
            timeout_sec = float(os.getenv("PIPECAT_S2S_CONNECT_TIMEOUT_SEC", "10"))
        await asyncio.wait_for(self._ready.wait(), timeout=timeout_sec)

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model": self._model,
            "voice": self._voice,
            "turn_detection": self._turn_detection,
            "max_history_turns": self._max_history_turns,
            "connection_epoch": self._connection_epoch,
            "response_active": self._response_started,
            "microphone_queue_depth": self._microphone_queue.qsize(),
            "microphone_queue_max_chunks": self._microphone_queue_max_chunks,
            "microphone_send_timeout_sec": self._microphone_send_timeout_sec,
            "last_error": self._last_error,
        }

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._start_background_tasks()
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_input_audio(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_external_interruption()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (CancelFrame, EndFrame)):
            await self._shutdown()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        await self._shutdown()
        await super().cleanup()

    def _start_background_tasks(self):
        if self._shutdown_task is not None:
            if not self._shutdown_task.done():
                return
            self._shutdown_task = None
        if self._connection_task is None or self._connection_task.done():
            self._shutting_down = False
            self._connection_task = self.create_task(
                self._connection_loop(),
                "qwen-audio-s2s-connection",
            )
        if (
            self._microphone_sender_task is None
            or self._microphone_sender_task.done()
        ):
            self._microphone_sender_task = self.create_task(
                self._microphone_sender_loop(),
                "qwen-audio-s2s-microphone-sender",
            )

    async def _connection_loop(self):
        delay = self._reconnect_initial_sec
        while not self._shutting_down:
            self._connection_epoch += 1
            epoch = self._connection_epoch
            self._interruption_emitted = False
            self._new_turn_marker = False
            attempt_connected = False
            attempt_became_ready = False
            self._drain_microphone_queue()
            self._ready.clear()
            self._websocket = None
            try:
                websocket = await asyncio.wait_for(
                    self._connect_factory(
                        self._build_url(),
                        {
                            "Authorization": f"Bearer {self._api_key}",
                        },
                    ),
                    timeout=self._connect_timeout_sec,
                )
                if self._shutting_down or epoch != self._connection_epoch:
                    await self._safe_close(websocket)
                    return
                self._websocket = websocket
                attempt_connected = True
                await self._send_event(
                    self._session_update_event(),
                    websocket=websocket,
                    epoch=epoch,
                )
                self._log(
                    f"[PIPECAT S2S] websocket connected epoch={epoch}; "
                    "waiting for session.updated"
                )
                await self._receive_events(websocket, epoch)
                if not self._shutting_down:
                    raise ConnectionError("Qwen-Audio websocket closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt_became_ready = self._ready.is_set()
                if not self._shutting_down:
                    self._log(
                        f"[PIPECAT S2S WARN] connection epoch={epoch} lost: {exc!r}"
                    )
                    if attempt_connected:
                        await self._handle_connection_loss(epoch)
            finally:
                websocket = self._websocket
                if epoch == self._connection_epoch:
                    self._ready.clear()
                    self._websocket = None
                await self._safe_close(websocket)

            if not self._shutting_down:
                if attempt_became_ready:
                    delay = self._reconnect_initial_sec
                await asyncio.sleep(delay)
                delay = min(self._reconnect_max_sec, max(delay * 2.0, 0.01))

    async def _receive_events(self, websocket: Any, epoch: int):
        async for message in websocket:
            if (
                epoch != self._connection_epoch
                or websocket is not self._websocket
                or self._shutting_down
            ):
                return
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            event = json.loads(message)
            await self._handle_provider_event(event, epoch)

    async def _handle_provider_event(self, event: dict[str, Any], epoch: int):
        if (
            epoch != self._connection_epoch
            or self._websocket is None
            or self._shutting_down
        ):
            return
        event_type = event.get("type")

        if event_type == "session.created":
            self._log(f"[PIPECAT S2S] session created epoch={epoch}")
        elif event_type == "session.updated":
            self._last_error = None
            self._ready.set()
            self._log(f"[PIPECAT S2S] session ready epoch={epoch}")
        elif event_type == "response.created":
            await self._start_response(event, epoch)
        elif event_type == "response.audio.delta":
            await self._push_audio_delta(event, epoch)
        elif event_type == "response.done":
            await self._finish_response(event, epoch)
        elif event_type == "input_audio_buffer.speech_started":
            await self._handle_provider_speech_started(epoch)
        elif event_type == "input_audio_buffer.speech_stopped":
            self._invalidate_pending_cancel()
            self._speech_stopped_at = time.monotonic()
            reason = event.get("reason", "valid")
            self._log(f"[PIPECAT S2S] speech stopped reason={reason}")
        elif event_type == "response.audio.done":
            event_response_id = self._event_response_id(event)
            if (
                self._response_started
                and self._response_epoch == epoch
                and (
                    not event_response_id
                    or not self._response_id
                    or event_response_id == self._response_id
                )
            ):
                self._audio_done_at = time.monotonic()
                self._log("[PIPECAT S2S] response audio done")
        elif event_type == "error":
            error = event.get("error", {})
            message = error.get("message", "unknown provider error")
            error_type = error.get("type")
            error_code = error.get("code")
            error_param = error.get("param")
            if error_code == "response_idle_timeout":
                self._last_error = None
                self._ready.clear()
                self._log(
                    "[PIPECAT S2S WARN] provider idle limit reached; "
                    "rolling realtime session immediately"
                )
                await self._safe_close(self._websocket)
                return
            self._last_error = (
                f"{error_type or 'unknown'}:{error_code or 'unknown'}:{message}"
            )
            if (
                error_type == "invalid_request_error"
                and self._is_benign_cancel_error(error)
            ):
                self._last_error = None
                self._log(f"[PIPECAT S2S WARN] provider rejected event: {message}")
                return
            self._log(
                "[PIPECAT S2S ERROR] "
                f"type={error_type} code={error_code} param={error_param}: {message}"
            )
            if self._response_started:
                await self._invalidate_response(send_cancel=False)
                await self._emit_interruption("provider error")
            if error_type == "server_error":
                websocket = self._websocket
                await self._safe_close(websocket)
        # response.audio.done is only logged. response.done remains the
        # provider's authoritative whole-response boundary.

    async def _start_response(
        self,
        event: dict[str, Any],
        epoch: int,
    ) -> bool:
        response_id = self._event_response_id(event)
        if response_id and (epoch, response_id) in self._rejected_responses:
            return False
        if self._suppress_until_created and not self._new_turn_marker:
            self._remember_terminal_response(epoch, response_id)
            self._log(
                "[PIPECAT S2S] response.created dropped behind interruption "
                f"barrier response_id={response_id or 'unknown'}"
            )
            return False
        if self._response_started:
            if response_id and response_id == self._response_id:
                return True
            await self._invalidate_response(send_cancel=False)
            await self._emit_interruption("provider replaced active response")
            if not self._new_turn_marker:
                self._remember_terminal_response(epoch, response_id)
                return False

        self._response_epoch = epoch
        self._invalidate_pending_cancel()
        self._response_id = response_id
        self._response_context_id = self._new_context_id(response_id, epoch)
        self._response_started = True
        self._suppress_until_created = False
        self._new_turn_marker = False
        self._interruption_emitted = False
        self._last_error = None
        self._response_started_at = time.monotonic()
        self._audio_done_at = 0.0
        self._first_audio_seen = False
        since_speech_ms = (
            (self._response_started_at - self._speech_stopped_at) * 1000.0
            if self._speech_stopped_at > 0.0
            else -1.0
        )
        self._log(
            "[PIPECAT S2S] response created "
            f"context={self._response_context_id} "
            f"after_speech_ms={since_speech_ms:.1f}"
        )
        await self.push_frame(
            TTSStartedFrame(context_id=self._response_context_id),
            FrameDirection.DOWNSTREAM,
        )
        return (
            self._response_started
            and self._response_epoch == epoch
            and self._response_id == response_id
        )

    async def _push_audio_delta(self, event: dict[str, Any], epoch: int):
        event_response_id = self._event_response_id(event)
        if (
            event_response_id
            and (epoch, event_response_id) in self._rejected_responses
        ):
            return
        if self._suppress_until_created and not self._response_started:
            return

        if not self._response_started:
            started = await self._start_response(event, epoch)
            if not started:
                return
        elif event_response_id and self._response_id is None:
            self._response_id = event_response_id
        if (
            not self._response_started
            or self._response_epoch != epoch
            or (
                event_response_id
                and self._response_id
                and event_response_id != self._response_id
            )
        ):
            return

        encoded_audio = event.get("delta")
        if not isinstance(encoded_audio, str) or not encoded_audio:
            return
        try:
            audio = base64.b64decode(encoded_audio, validate=True)
        except Exception as exc:
            self._log(f"[PIPECAT S2S WARN] invalid audio delta dropped: {exc!r}")
            return
        if not audio:
            return

        if not self._first_audio_seen:
            self._first_audio_seen = True
            first_audio_ms = (
                (time.monotonic() - self._speech_stopped_at) * 1000.0
                if self._speech_stopped_at > 0.0
                else -1.0
            )
            model_audio_ms = (
                (time.monotonic() - self._response_started_at) * 1000.0
                if self._response_started_at > 0.0
                else -1.0
            )
            self._log(
                "[PIPECAT S2S] first audio "
                f"context={self._response_context_id} epoch={epoch} "
                f"after_speech_ms={first_audio_ms:.1f} "
                f"after_response_ms={model_audio_ms:.1f}"
            )
        await self.push_frame(
            TTSAudioRawFrame(
                audio=audio,
                sample_rate=OUTPUT_SAMPLE_RATE,
                num_channels=1,
                context_id=self._response_context_id,
            ),
            FrameDirection.DOWNSTREAM,
        )

    async def _finish_response(self, event: dict[str, Any], epoch: int):
        event_response_id = self._event_response_id(event)
        response = event.get("response")
        status = response.get("status") if isinstance(response, dict) else None
        if (
            not self._response_started
            or self._response_epoch != epoch
            or (
                event_response_id
                and self._response_id
                and event_response_id != self._response_id
            )
        ):
            if status in {"cancelled", "failed", "incomplete"}:
                self._remember_terminal_response(epoch, event_response_id)
            return
        if status and status != "completed":
            await self._invalidate_response(send_cancel=False)
            await self._emit_interruption(f"response {status}")
            return
        context_id = self._response_context_id
        response_ms = (
            (time.monotonic() - self._response_started_at) * 1000.0
            if self._response_started_at > 0.0
            else -1.0
        )
        audio_done_gap_ms = (
            (time.monotonic() - self._audio_done_at) * 1000.0
            if self._audio_done_at > 0.0
            else -1.0
        )
        self._remember_terminal_response(
            self._response_epoch,
            self._response_id,
        )
        self._clear_response()
        await self.push_frame(
            TTSStoppedFrame(context_id=context_id),
            FrameDirection.DOWNSTREAM,
        )
        self._log(
            f"[PIPECAT S2S] response completed context={context_id} "
            f"duration_ms={response_ms:.1f} "
            f"after_audio_done_ms={audio_done_gap_ms:.1f}"
        )
        self._interruption_emitted = False

    async def _handle_external_interruption(self):
        # The caller forwards the original InterruptionFrame. Mark this before
        # awaiting response.cancel so a concurrent provider speech_started
        # cannot emit a duplicate frame.
        self._interruption_emitted = True
        await self._invalidate_response(
            send_cancel=True,
            cancel_if_inactive=True,
        )

    async def _handle_provider_speech_started(self, epoch: int):
        if epoch != self._connection_epoch:
            return
        self._log("[PIPECAT S2S] speech started")
        self._new_turn_marker = True
        await self._invalidate_response(send_cancel=True)
        await self._emit_interruption("provider speech_started")

    async def _handle_connection_loss(self, epoch: int):
        if epoch != self._connection_epoch:
            return
        self._new_turn_marker = False
        self._drain_microphone_queue()
        await self._invalidate_response(send_cancel=False)
        await self._emit_interruption("provider disconnected")

    async def _invalidate_response(
        self,
        *,
        send_cancel: bool,
        cancel_if_inactive: bool = False,
    ):
        response_was_active = self._response_started
        websocket = self._websocket
        epoch = self._connection_epoch
        cancel_generation = self._invalidate_pending_cancel()
        self._remember_terminal_response(
            self._response_epoch,
            self._response_id,
        )
        self._clear_response()
        self._suppress_until_created = True

        if (
            send_cancel
            and (response_was_active or cancel_if_inactive)
            and self.ready
            and websocket is not None
        ):
            self._create_auxiliary_task(
                self._send_cancel_best_effort(
                    websocket,
                    epoch,
                    cancel_generation,
                ),
                f"qwen-audio-s2s-response-cancel-{cancel_generation}",
            )

    async def _send_cancel_best_effort(
        self,
        websocket: Any,
        epoch: int,
        cancel_generation: int,
    ):
        try:
            await self._send_event(
                {"type": "response.cancel"},
                websocket=websocket,
                epoch=epoch,
                send_guard=lambda: (
                    cancel_generation == self._response_cancel_generation
                    and not self._response_started
                ),
            )
        except Exception as exc:
            self._log(f"[PIPECAT S2S WARN] response.cancel failed: {exc!r}")
            await self._handle_send_failure(websocket, epoch)

    def _invalidate_pending_cancel(self) -> int:
        self._response_cancel_generation += 1
        return self._response_cancel_generation

    def _create_auxiliary_task(
        self,
        coroutine: Awaitable[Any],
        name: str,
    ) -> asyncio.Task:
        task = self.create_task(coroutine, name)
        self._auxiliary_tasks.add(task)
        task.add_done_callback(self._auxiliary_tasks.discard)
        return task

    def _remember_terminal_response(
        self,
        epoch: int,
        response_id: str | None,
    ):
        if epoch <= 0 or not response_id:
            return
        key = (epoch, response_id)
        if key in self._rejected_responses:
            return
        self._rejected_responses.add(key)
        self._terminal_response_order.append(key)
        while len(self._terminal_response_order) > 64:
            expired = self._terminal_response_order.popleft()
            self._rejected_responses.discard(expired)

    def _clear_response(self):
        self._response_epoch = 0
        self._response_id = None
        self._response_context_id = None
        self._response_started = False
        self._response_started_at = 0.0
        self._audio_done_at = 0.0
        self._first_audio_seen = False

    async def _emit_interruption(self, reason: str):
        if self._interruption_emitted:
            return
        self._interruption_emitted = True
        self._log(f"[PIPECAT S2S] interruption: {reason}")
        await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    async def _send_input_audio(self, frame: InputAudioRawFrame):
        if (
            frame.sample_rate != INPUT_SAMPLE_RATE
            or frame.num_channels != 1
        ):
            self._log(
                "[PIPECAT S2S WARN] input audio dropped; expected "
                f"16k mono, got {frame.sample_rate}Hz/{frame.num_channels}ch"
            )
            return
        if not frame.audio or not self.ready:
            return
        websocket = self._websocket
        epoch = self._connection_epoch
        if websocket is None:
            return
        queued_audio = bytes(frame.audio)
        try:
            self._microphone_queue.put_nowait(
                (epoch, websocket, queued_audio)
            )
        except asyncio.QueueFull:
            _, _, dropped_audio = self._microphone_queue.get_nowait()
            self._microphone_queue.task_done()
            self._microphone_queue.put_nowait(
                (epoch, websocket, queued_audio)
            )
            dropped_ms = len(dropped_audio) / (INPUT_SAMPLE_RATE * 2) * 1000.0
            self._log(
                "[PIPECAT S2S WARN] microphone sender backlog reached "
                f"{self._microphone_queue_max_chunks} chunks; dropped oldest "
                f"{dropped_ms:.1f}ms and kept websocket alive"
            )

    async def _microphone_sender_loop(self):
        while not self._shutting_down:
            try:
                epoch, websocket, audio = await self._microphone_queue.get()
            except asyncio.CancelledError:
                raise
            try:
                if (
                    epoch != self._connection_epoch
                    or websocket is not self._websocket
                    or not self.ready
                ):
                    continue
                await asyncio.wait_for(
                    self._send_event(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio).decode("ascii"),
                        },
                        websocket=websocket,
                        epoch=epoch,
                    ),
                    timeout=self._microphone_send_timeout_sec,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(f"[PIPECAT S2S WARN] microphone send failed: {exc!r}")
                await self._handle_send_failure(websocket, epoch)
            finally:
                self._microphone_queue.task_done()

    async def _handle_send_failure(self, websocket: Any, epoch: int):
        if (
            epoch == self._connection_epoch
            and websocket is self._websocket
        ):
            self._ready.clear()
            self._drain_microphone_queue()
        await self._safe_close(websocket)

    def _drain_microphone_queue(self):
        while True:
            try:
                self._microphone_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._microphone_queue.task_done()

    async def _send_event(
        self,
        event: dict[str, Any],
        *,
        websocket: Any | None = None,
        epoch: int | None = None,
        send_guard: Callable[[], bool] | None = None,
    ) -> bool:
        target = websocket or self._websocket
        expected_epoch = epoch if epoch is not None else self._connection_epoch
        if send_guard is not None and not send_guard():
            return False
        if (
            target is None
            or expected_epoch != self._connection_epoch
            or self._shutting_down
        ):
            raise ConnectionError("Qwen-Audio websocket is not available")
        payload = dict(event)
        payload.setdefault("event_id", self._next_event_id())
        async with self._send_lock:
            if (
                expected_epoch != self._connection_epoch
                or target is not self._websocket
                or self._shutting_down
            ):
                raise ConnectionError("Qwen-Audio websocket changed before send")
            if send_guard is not None and not send_guard():
                return False
            await target.send(json.dumps(payload, ensure_ascii=False))
        return True

    def _session_update_event(self) -> dict[str, Any]:
        turn_detection: dict[str, Any] = {"type": self._turn_detection}
        if self._turn_detection == "server_vad":
            turn_detection.update(
                {
                    "threshold": self._threshold,
                    "silence_duration_ms": self._silence_duration_ms,
                }
            )
        session: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "voice": self._voice,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "max_history_turns": self._max_history_turns,
            "turn_detection": turn_detection,
        }
        if self._instructions:
            session["instructions"] = self._instructions
        return {"type": "session.update", "session": session}

    def _build_url(self) -> str:
        if "model=" in self._base_url:
            return self._base_url
        separator = "&" if "?" in self._base_url else "?"
        return f"{self._base_url}{separator}model={quote(self._model, safe='')}"

    def _new_context_id(self, response_id: str | None, epoch: int) -> str:
        if response_id:
            return f"qwen-audio-{epoch}-{response_id}"
        self._synthetic_response_counter += 1
        return f"qwen-audio-{epoch}-synthetic-{self._synthetic_response_counter}"

    @staticmethod
    def _event_response_id(event: dict[str, Any]) -> str | None:
        response_id = event.get("response_id")
        if response_id:
            return str(response_id)
        response = event.get("response")
        if isinstance(response, dict) and response.get("id"):
            return str(response["id"])
        return None

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event_{int(time.time() * 1000)}_{self._event_counter}"

    @staticmethod
    def _is_benign_cancel_error(error: dict[str, Any]) -> bool:
        param = str(error.get("param") or "").lower()
        if param == "response.cancel":
            return True
        message = str(error.get("message") or "").lower()
        return (
            "cancel" in message
            and any(
                marker in message
                for marker in (
                    "no active response",
                    "no inference",
                    "not in progress",
                    "already cancelled",
                    "already canceled",
                )
            )
        )

    async def _shutdown(self):
        task = self._shutdown_task
        if task is None:
            task = self.create_task(
                self._shutdown_impl(),
                "qwen-audio-s2s-shutdown",
            )
            self._shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_impl(self):
        self._shutting_down = True
        self._ready.clear()
        self._connection_epoch += 1
        self._invalidate_pending_cancel()
        self._drain_microphone_queue()
        self._clear_response()
        self._new_turn_marker = False
        websocket = self._websocket
        self._websocket = None
        tasks = [
            task
            for task in (
                self._connection_task,
                self._microphone_sender_task,
                *tuple(self._auxiliary_tasks),
            )
            if task is not None and task is not asyncio.current_task()
        ]
        self._connection_task = None
        self._microphone_sender_task = None
        try:
            for task in tasks:
                task.cancel()
            if tasks:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=self._shutdown_timeout_sec,
                )
                for task in pending:
                    task.cancel()
                if pending:
                    retried_done, pending = await asyncio.wait(
                        pending,
                        timeout=self._shutdown_timeout_sec,
                    )
                    done.update(retried_done)
                if done:
                    await asyncio.gather(*done, return_exceptions=True)
                if pending:
                    self._log(
                        "[PIPECAT S2S WARN] shutdown left "
                        f"{len(pending)} unresponsive task(s)"
                    )
            await self._safe_close(websocket)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            self._auxiliary_tasks.difference_update(tasks)
            self._drain_microphone_queue()
            self._clear_response()
            self._new_turn_marker = False

    async def _safe_close(self, websocket: Any | None):
        if websocket is None:
            return
        try:
            await asyncio.wait_for(
                websocket.close(),
                timeout=self._shutdown_timeout_sec,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self._log("[PIPECAT S2S WARN] websocket close timed out")
        except Exception:
            return

    async def _default_connect(
        self,
        url: str,
        headers: dict[str, str],
    ) -> Any:
        import websockets

        return await websockets.connect(
            url,
            additional_headers=headers,
            open_timeout=self._connect_timeout_sec,
            ping_interval=self._ping_interval_sec,
            ping_timeout=self._ping_timeout_sec,
        )


def create_qwen_audio_s2s_from_env(
    log: LogCallback | None = None,
) -> QwenAudioRealtimeS2SProcessor:
    """Build the realtime processor from PIPECAT_S2S_* environment variables."""

    api_key = next(
        (
            value.strip()
            for name in (
                "PIPECAT_S2S_API_KEY",
                "DASHSCOPE_API_KEY",
                "PIPECAT_LLM_API_KEY",
                "OPENAI_API_KEY",
            )
            if (value := os.getenv(name))
            and value.strip()
        ),
        "",
    )
    if not api_key:
        raise RuntimeError(
            "Set PIPECAT_S2S_API_KEY (or DASHSCOPE_API_KEY) for Qwen-Audio realtime"
        )

    base_url = os.getenv("PIPECAT_S2S_BASE_URL", "").strip()
    if not base_url:
        workspace_id = os.getenv("PIPECAT_S2S_WORKSPACE_ID", "").strip()
        if workspace_id:
            base_url = (
                f"wss://{workspace_id}.cn-beijing.maas.aliyuncs.com"
                "/api-ws/v1/realtime"
            )
        else:
            base_url = COMPATIBILITY_BASE_URL
            if log is not None:
                log(
                    "[PIPECAT S2S WARN] PIPECAT_S2S_WORKSPACE_ID/base URL "
                    "not set; using the verified DashScope compatibility "
                    "endpoint. Prefer the workspace MAAS endpoint for "
                    "production."
                )

    turn_detection = os.getenv(
        "PIPECAT_S2S_TURN_DETECTION",
        "smart_turn",
    ).strip().lower()
    if turn_detection not in {"smart_turn", "server_vad"}:
        raise RuntimeError(
            "PIPECAT_S2S_TURN_DETECTION must be smart_turn or server_vad; "
            "manual mode is not implemented"
        )
    return QwenAudioRealtimeS2SProcessor(
        api_key=api_key,
        model=os.getenv("PIPECAT_S2S_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        voice=os.getenv("PIPECAT_S2S_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE,
        base_url=base_url,
        turn_detection=turn_detection,
        silence_duration_ms=int(
            os.getenv("PIPECAT_S2S_VAD_SILENCE_MS", "800")
        ),
        threshold=float(os.getenv("PIPECAT_S2S_VAD_THRESHOLD", "0.5")),
        instructions=(
            os.getenv("PIPECAT_S2S_INSTRUCTIONS", "").strip()
            or os.getenv("PIPECAT_SYSTEM_INSTRUCTION", "").strip()
            or DEFAULT_INSTRUCTIONS
        ),
        max_history_turns=int(
            os.getenv("PIPECAT_S2S_MAX_HISTORY_TURNS", "10")
        ),
        microphone_queue_max_chunks=int(
            os.getenv("PIPECAT_S2S_MIC_QUEUE_MAX_CHUNKS", "3")
        ),
        microphone_send_timeout_sec=float(
            os.getenv("PIPECAT_S2S_MIC_SEND_TIMEOUT_SEC", "1.0")
        ),
        log=log,
        connect_timeout_sec=float(
            os.getenv("PIPECAT_S2S_CONNECT_TIMEOUT_SEC", "10")
        ),
        reconnect_initial_sec=float(
            os.getenv("PIPECAT_S2S_RECONNECT_INITIAL_SEC", "0.25")
        ),
        reconnect_max_sec=float(
            os.getenv("PIPECAT_S2S_RECONNECT_MAX_SEC", "5")
        ),
        ping_interval_sec=float(
            os.getenv("PIPECAT_S2S_PING_INTERVAL_SEC", "20")
        ),
        ping_timeout_sec=float(
            os.getenv("PIPECAT_S2S_PING_TIMEOUT_SEC", "20")
        ),
    )
