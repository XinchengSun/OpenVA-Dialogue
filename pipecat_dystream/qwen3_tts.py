"""Low-latency DashScope Qwen3 realtime TTS for Pipecat.

The provider SDK receives WebSocket callbacks on its own thread.  This adapter
routes every response back into Pipecat's audio-context queue on the asyncio
event loop.  A connection generation prevents audio from an interrupted turn
from leaking into the next turn.
"""

from __future__ import annotations

import asyncio
import base64
import os
from collections import Counter, deque
from collections.abc import AsyncGenerator, Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.transcriptions.language import Language


ProviderFactory = Callable[[Any], Any]


class _QwenCallback:
    """Forward synchronous SDK callbacks to one service connection generation."""

    def __init__(self, service: "Qwen3RealtimeTTSService", generation: int):
        self._service = service
        self._generation = generation

    def on_open(self) -> None:
        return None

    def on_close(self, close_status_code, close_msg) -> None:
        self._service._submit_provider_close(
            self._generation,
            close_status_code,
            close_msg,
        )

    def on_event(self, message: dict[str, Any]) -> None:
        self._service._submit_provider_event(self._generation, message)


class Qwen3RealtimeTTSService(TTSService):
    """Keep one Qwen3 realtime WebSocket open and stream raw PCM into Pipecat."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-tts-flash-realtime",
        voice: str = "Cherry",
        sample_rate: int = 16_000,
        language_type: str = "Chinese",
        provider_factory: ProviderFactory | None = None,
        **kwargs,
    ):
        super().__init__(
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            push_text_frames=False,
            # A provider reconnect can exceed Pipecat's 3s context timeout.
            # Open the context in run_tts only after the WebSocket is ready.
            push_start_frame=False,
            # If the provider silently omits response.done, Pipecat's context
            # timeout still emits a matching stop frame to the DyStream sink.
            push_stop_frames=True,
            sample_rate=sample_rate,
            reuse_context_id_within_turn=True,
            settings=TTSSettings(
                model=model,
                voice=voice,
                language=Language.ZH,
            ),
            **kwargs,
        )
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._language_type = language_type
        self._provider_factory = provider_factory or self._default_provider_factory

        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._provider: Any | None = None
        # Epoch controls reconnect ownership. Callback token is incremented for
        # every individual SDK connection attempt, including failed attempts.
        self._generation = 0
        self._callback_token = 0
        self._ready = asyncio.Event()
        self._connect_lock = asyncio.Lock()
        self._provider_events: asyncio.Queue | None = None
        self._event_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._shutting_down = False
        self._shutdown_complete = False
        self._interrupt_reconnect_pending = False

        # Commits are ordered on one provider connection. response.created gives
        # each commit a response_id, which is then present on every audio delta.
        self._next_commit_token = 0
        self._pending_commits: deque[tuple[int, str]] = deque()
        self._response_contexts: dict[str, tuple[int, str]] = {}
        self._context_pending: Counter[str] = Counter()
        self._flush_requested: set[str] = set()
        self._closed_contexts: set[str] = set()

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def ready(self) -> bool:
        return self._provider is not None and self._ready.is_set()

    def _default_provider_factory(self, callback: Any) -> Any:
        # QwenTtsRealtime reads the key from dashscope.api_key at construction.
        import dashscope
        from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime

        dashscope.api_key = self._api_key
        return QwenTtsRealtime(
            model=self._model,
            callback=callback,
        )

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._event_loop = asyncio.get_running_loop()
        self._shutting_down = False
        self._shutdown_complete = False
        if self._event_task is None:
            self._provider_events = asyncio.Queue()
            self._event_task = asyncio.create_task(
                self._provider_event_loop(),
                name="qwen3-tts-provider-events",
            )
        try:
            await self._connect_with_retries(self._generation)
        except Exception as exc:
            logger.warning(
                f"[QWEN3 TTS] initial connection deferred to background: {exc!r}"
            )
            self._start_background_task(
                self._retry_until_connected(self._generation)
            )

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._shutting_down = True
        await self._shutdown_provider()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._shutting_down = True
        await self._shutdown_provider()

    async def cleanup(self):
        await super().cleanup()
        self._shutting_down = True
        await self._shutdown_provider()

    async def wait_ready(self, timeout: float | None = None):
        timeout_s = timeout or float(os.getenv("PIPECAT_TTS_CONNECT_TIMEOUT_SEC", "6.0"))
        await asyncio.wait_for(self._ready.wait(), timeout=timeout_s)

    def _submit_provider_event(self, generation: int, message: dict[str, Any]):
        loop = self._event_loop
        queue = self._provider_events
        if loop is None or loop.is_closed() or queue is None:
            return
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("event", generation, message),
            )
        except RuntimeError:
            return

    def _submit_provider_close(
        self,
        generation: int,
        close_status_code: Any,
        close_msg: Any,
    ):
        loop = self._event_loop
        queue = self._provider_events
        if loop is None or loop.is_closed() or queue is None:
            return
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                (
                    "close",
                    generation,
                    (close_status_code, close_msg),
                ),
            )
        except RuntimeError:
            return

    async def _provider_event_loop(self):
        queue = self._provider_events
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            kind, generation, payload = item
            try:
                if kind == "event":
                    await self._handle_provider_event(generation, payload)
                else:
                    close_status_code, close_msg = payload
                    await self._handle_provider_close(
                        generation,
                        close_status_code,
                        close_msg,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[QWEN3 TTS] provider event failed: {exc!r}")
            finally:
                queue.task_done()

    def _start_background_task(self, coroutine):
        if self._shutting_down:
            coroutine.close()
            return
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_task_done)

    def _background_task_done(self, task: asyncio.Task):
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(f"[QWEN3 TTS] background task failed: {error!r}")

    async def _connect_provider(self, expected_generation: int):
        async with self._connect_lock:
            if self._shutting_down or expected_generation != self._generation:
                return
            if self.ready:
                return

            self._ready.clear()
            self._callback_token += 1
            callback_token = self._callback_token
            callback = _QwenCallback(self, callback_token)
            try:
                # Create the handle before entering to_thread so cancellation
                # can always close a still-running SDK connect attempt.
                provider = self._provider_factory(callback)
            except Exception:
                if callback_token == self._callback_token:
                    self._callback_token += 1
                raise

            def connect():
                provider.connect()
                provider.update_session(
                    voice=self._voice,
                    mode="commit",
                    sample_rate=self.sample_rate,
                    audio_format="pcm",
                    language_type=self._language_type,
                )
                return provider

            started_at = asyncio.get_running_loop().time()
            try:
                await asyncio.to_thread(connect)
            except asyncio.CancelledError:
                if callback_token == self._callback_token:
                    self._callback_token += 1
                    self._ready.clear()
                await self._close_provider(provider, cancel_response=False)
                raise
            except Exception:
                if callback_token == self._callback_token:
                    # Make late session.updated/on_close callbacks from this
                    # half-open SDK thread stale before attempting close.
                    self._callback_token += 1
                    self._ready.clear()
                await self._close_provider(provider, cancel_response=False)
                raise

            if (
                self._shutting_down
                or expected_generation != self._generation
                or callback_token != self._callback_token
            ):
                await self._close_provider(provider, cancel_response=False)
                return

            self._provider = provider
            try:
                await self.wait_ready()
            except asyncio.CancelledError:
                self._provider = None
                if callback_token == self._callback_token:
                    self._callback_token += 1
                await self._close_provider(provider, cancel_response=False)
                raise
            except Exception:
                self._provider = None
                if callback_token == self._callback_token:
                    self._callback_token += 1
                await self._close_provider(provider, cancel_response=False)
                raise

            logger.info(
                "[QWEN3 TTS] websocket ready "
                f"model={self._model} voice={self._voice} "
                f"connect_ms={(asyncio.get_running_loop().time() - started_at) * 1000.0:.1f}"
            )
            self._interrupt_reconnect_pending = False
            await self._call_event_handler("on_connected")

    async def _connect_with_retries(self, expected_generation: int):
        attempts = max(1, int(os.getenv("PIPECAT_TTS_CONNECT_ATTEMPTS", "2")))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                await self._connect_provider(expected_generation)
                if self.ready:
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[QWEN3 TTS] connect attempt={attempt}/{attempts} "
                    f"failed: {exc!r}"
                )
            if (
                attempt < attempts
                and not self._shutting_down
                and expected_generation == self._generation
            ):
                await asyncio.sleep(0.25)
        if last_error is not None:
            raise last_error
        raise ConnectionError("Qwen3 TTS websocket is not ready")

    async def _retry_until_connected(self, expected_generation: int):
        retry_delay = 0.5
        while (
            not self._shutting_down
            and expected_generation == self._generation
            and not self.ready
        ):
            try:
                await self._connect_with_retries(expected_generation)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[QWEN3 TTS] background reconnect failed: {exc!r}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2.0, 5.0)

    async def _ensure_connected(self):
        if self.ready:
            return
        await self._connect_with_retries(self._generation)
        if not self.ready:
            raise ConnectionError("Qwen3 TTS websocket is not ready")

    async def _close_provider(self, provider: Any | None, *, cancel_response: bool):
        if provider is None:
            return

        def close():
            if cancel_response:
                try:
                    provider.cancel_response()
                except Exception:
                    pass
            try:
                provider.close()
            except Exception:
                pass

        try:
            await asyncio.wait_for(asyncio.to_thread(close), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("[QWEN3 TTS] provider close timed out")

    def _clear_request_state(self, *, clear_closed_contexts: bool):
        self._pending_commits.clear()
        self._response_contexts.clear()
        self._context_pending.clear()
        self._flush_requested.clear()
        if clear_closed_contexts:
            self._closed_contexts.clear()

    async def _shutdown_provider(self):
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        provider = self._provider
        self._provider = None
        self._generation += 1
        self._callback_token += 1
        self._ready.clear()
        self._clear_request_state(clear_closed_contexts=True)
        await self._close_provider(provider, cancel_response=True)
        tasks = [task for task in self._background_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        if self._event_task is not None:
            if not self._event_task.done() and self._provider_events is not None:
                await self._provider_events.put(None)
            await asyncio.gather(self._event_task, return_exceptions=True)
            self._event_task = None
        self._provider_events = None
        await self._call_event_handler("on_disconnected")

    async def _invalidate_and_reconnect(
        self,
        *,
        cancel_response: bool,
        clear_closed_contexts: bool,
    ):
        provider = self._provider
        self._provider = None
        self._generation += 1
        self._callback_token += 1
        reconnect_generation = self._generation
        self._ready.clear()
        self._clear_request_state(
            clear_closed_contexts=clear_closed_contexts,
        )

        async def reconnect():
            await self._close_provider(provider, cancel_response=cancel_response)
            if not self._shutting_down and reconnect_generation == self._generation:
                await self._retry_until_connected(reconnect_generation)

        self._start_background_task(reconnect())

    async def on_audio_context_interrupted(self, context_id: str):
        # Invalidate first. The server may keep sending for roughly a second
        # after response.cancel, but its callback generation is now stale.
        await self.stop_all_metrics()
        if not self._interrupt_reconnect_pending:
            self._interrupt_reconnect_pending = True
            await self._invalidate_and_reconnect(
                cancel_response=True,
                clear_closed_contexts=True,
            )
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        self._closed_contexts.discard(context_id)
        await super().on_audio_context_completed(context_id)

    async def flush_audio(self, context_id: str | None = None):
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id:
            return
        self._flush_requested.add(flush_id)
        if self._context_pending.get(flush_id, 0) == 0:
            await self._finish_context(flush_id)

    async def _finish_context(self, context_id: str, error: ErrorFrame | None = None):
        if context_id in self._closed_contexts:
            return
        self._closed_contexts.add(context_id)
        if self.audio_context_available(context_id):
            if error is not None:
                await self.append_to_audio_context(context_id, error)
            await self.append_to_audio_context(
                context_id,
                TTSStoppedFrame(context_id=context_id),
            )
            await self.remove_audio_context(context_id)
        self._context_pending.pop(context_id, None)
        self._flush_requested.discard(context_id)

    def _contexts_for_current_generation(self) -> set[str]:
        contexts = set(self._context_pending)
        contexts.update(context_id for _, context_id in self._pending_commits)
        contexts.update(context_id for _, context_id in self._response_contexts.values())
        return contexts

    async def _fail_current_generation(
        self,
        error_message: str,
        *,
        extra_contexts: set[str] | None = None,
    ):
        contexts = self._contexts_for_current_generation()
        if extra_contexts:
            contexts.update(extra_contexts)
        for context_id in contexts:
            await self._finish_context(
                context_id,
                ErrorFrame(error=error_message),
            )
        await self._invalidate_and_reconnect(
            cancel_response=False,
            clear_closed_contexts=False,
        )

    async def _handle_provider_event(
        self,
        callback_token: int,
        message: dict[str, Any],
    ):
        if callback_token != self._callback_token or self._shutting_down:
            return
        event_type = str(message.get("type", ""))

        if event_type == "session.updated":
            self._ready.set()
            return

        if event_type == "response.created":
            response_id = str(message.get("response", {}).get("id", ""))
            if not response_id or not self._pending_commits:
                logger.warning("[QWEN3 TTS] response.created without a pending commit")
                return
            self._response_contexts[response_id] = self._pending_commits.popleft()
            return

        if event_type == "response.audio.delta":
            response_id = str(message.get("response_id", ""))
            commit_record = self._response_contexts.get(response_id)
            if not commit_record:
                return
            _, context_id = commit_record
            if not self.audio_context_available(context_id):
                return
            try:
                audio = base64.b64decode(message.get("delta", ""), validate=True)
            except Exception as exc:
                await self.push_error(
                    error_msg=f"Invalid Qwen3 TTS audio chunk: {exc}",
                    exception=exc,
                )
                return
            if not audio:
                return
            if len(audio) % 2:
                await self.push_error(error_msg="Qwen3 TTS returned unaligned PCM16 audio")
                return
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            return

        if event_type == "response.done":
            response = message.get("response", {})
            response_id = str(response.get("id", ""))
            commit_record = self._response_contexts.pop(response_id, None)
            if not commit_record:
                return
            _, context_id = commit_record
            if response.get("status") != "completed":
                await self._fail_current_generation(
                    f"Qwen3 TTS response failed: {response}",
                    extra_contexts={context_id},
                )
                return
            pending = self._context_pending.get(context_id, 0)
            if pending > 1:
                self._context_pending[context_id] = pending - 1
            else:
                self._context_pending.pop(context_id, None)
            if (
                context_id in self._flush_requested
                and self._context_pending.get(context_id, 0) == 0
            ):
                await self._finish_context(context_id)
            return

        if event_type == "error":
            await self._fail_current_generation(
                f"Qwen3 TTS provider error: {message}"
            )

    async def _handle_provider_close(
        self,
        callback_token: int,
        close_status_code: Any,
        close_msg: Any,
    ):
        if callback_token != self._callback_token or self._shutting_down:
            return
        await self._fail_current_generation(
            "Qwen3 TTS websocket closed unexpectedly: "
            f"code={close_status_code!r} message={close_msg!r}"
        )

    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None, None]:
        send_generation: int | None = None
        try:
            if context_id in self._closed_contexts:
                # A provider failure may close a context while a later sentence
                # from the same LLM turn is already queued. Drop it without
                # disturbing the newly reconnected provider.
                logger.warning(
                    f"[QWEN3 TTS] drop text for closing context={context_id}"
                )
                yield None
                return
            await self._ensure_connected()
            if not self.audio_context_available(context_id):
                await self.create_audio_context(context_id)
                await self.start_ttfb_metrics()
                await self.append_to_audio_context(
                    context_id,
                    TTSStartedFrame(context_id=context_id),
                )

            provider = self._provider
            if provider is None:
                raise ConnectionError("Qwen3 TTS websocket disappeared before synthesis")

            send_generation = self._generation
            send_callback_token = self._callback_token
            self._next_commit_token += 1
            commit_record = (self._next_commit_token, context_id)
            self._pending_commits.append(commit_record)
            self._context_pending[context_id] += 1

            def commit_text():
                provider.append_text(text)
                provider.commit()

            try:
                await asyncio.to_thread(commit_text)
            except Exception:
                # Remove only this exact submission. Multiple sentences in one
                # turn share the same context_id.
                removed = False
                try:
                    self._pending_commits.remove(commit_record)
                    removed = True
                except ValueError:
                    pass
                if removed:
                    pending = self._context_pending.get(context_id, 0)
                    if pending > 1:
                        self._context_pending[context_id] = pending - 1
                    else:
                        self._context_pending.pop(context_id, None)
                raise

            if (
                provider is not self._provider
                or send_generation != self._generation
                or send_callback_token != self._callback_token
            ):
                # Interruption/close already owns cleanup and reconnection.
                yield None
                return

            try:
                await self.start_tts_usage_metrics(text)
            except Exception as exc:
                logger.warning(f"[QWEN3 TTS] usage metrics failed: {exc!r}")
            yield None
        except Exception as exc:
            error = ErrorFrame(error=f"Qwen3 TTS synthesis failed: {exc}")
            if self.audio_context_available(context_id):
                await self._finish_context(context_id, error)
                yield None
            else:
                yield error
            if send_generation is None or send_generation == self._generation:
                await self._invalidate_and_reconnect(
                    cancel_response=False,
                    clear_closed_contexts=False,
                )
