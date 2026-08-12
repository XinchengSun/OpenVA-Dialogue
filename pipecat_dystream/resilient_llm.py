"""Small recovery layer for transient OpenAI-compatible LLM timeouts."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Any

import httpx
from loguru import logger
from openai import APITimeoutError
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import assert_given

from .realtime_search import (
    RealtimeSearchPolicy,
    add_dashscope_search_params,
    last_user_text,
    normalize_realtime_search_strategy,
)


TIMEOUT_EXCEPTIONS = (APITimeoutError, httpx.TimeoutException, TimeoutError)

_SENTENCE_TERMINATORS = frozenset("。！？!?")
_SENTENCE_CLOSERS = frozenset("\"”’'」』】》〉）)]}")


class _ReplyLimitReached(Exception):
    """Internal normal-completion signal after the last allowed sentence."""


@dataclass
class _ReplyStreamState:
    buffer: str = ""
    sentences_emitted: int = 0
    finish_reason: str | None = None
    visible_text_emitted: bool = False
    limit_reached: bool = False


def _terminator_width(text: str, index: int) -> int:
    """Return the width of a sentence terminator, avoiding decimal points."""

    char = text[index]
    if char in _SENTENCE_TERMINATORS:
        return 1
    if char == "…" and index + 1 < len(text) and text[index + 1] == "…":
        return 2
    if text.startswith("...", index):
        return 3
    if char != ".":
        return 0

    previous = text[index - 1] if index > 0 else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return 0
    return 1


def _split_complete_sentences(text: str, *, final: bool) -> tuple[list[str], str]:
    """Split complete sentences while retaining a possibly incomplete suffix.

    A terminator at the current chunk boundary is held until the next chunk so
    that a closing quote or bracket arriving separately remains attached to the
    sentence that owns it. At natural EOF, ``final=True`` releases that sentence.
    """

    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        width = _terminator_width(text, index)
        if not width:
            index += 1
            continue

        end = index + width
        while end < len(text):
            trailing_width = _terminator_width(text, end)
            if trailing_width:
                end += trailing_width
            elif text[end] in _SENTENCE_CLOSERS or text[end].isspace():
                end += 1
            else:
                break

        if end == len(text) and not final:
            break
        sentences.append(text[start:end])
        start = end
        index = end

    return sentences, text[start:]


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def trim_context_history(context, max_messages: int) -> int:
    """Keep the newest conversational messages and return the number removed."""
    messages = list(context.get_messages())
    if max_messages <= 0 or len(messages) <= max_messages:
        return 0
    removed = len(messages) - max_messages
    context.set_messages(messages[-max_messages:])
    return removed


def _chunk_has_answer_content(chunk) -> bool:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return False
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return False
    if getattr(delta, "content", None) or getattr(delta, "tool_calls", None):
        return True
    audio = getattr(delta, "audio", None)
    return bool(audio and getattr(audio, "transcript", None))


async def _close_stream(stream) -> None:
    if stream is None:
        return
    close = getattr(stream, "close", None)
    if close is None:
        close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


class RecoveringOpenAILLMService(OpenAILLMService):
    """Fail fast on provider timeouts without poisoning later conversation turns."""

    def __init__(
        self,
        *args,
        timeout_fallback_text: str,
        history_max_messages: int = 12,
        reply_max_sentences: int = 3,
        hedge_model: str = "",
        hedge_delay_sec: float = 0.75,
        hedge_deadline_sec: float = 3.0,
        realtime_search_mode: str = "off",
        realtime_search_strategy: str = "turbo",
        realtime_search_model: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._timeout_fallback_text = timeout_fallback_text
        self._history_max_messages = max(0, int(history_max_messages))
        self._reply_max_sentences = int(reply_max_sentences)
        if self._reply_max_sentences <= 0:
            raise ValueError("reply_max_sentences must be positive")
        self._hedge_model = hedge_model.strip()
        self._hedge_delay_sec = max(0.0, float(hedge_delay_sec))
        self._hedge_deadline_sec = max(
            self._hedge_delay_sec,
            float(hedge_deadline_sec),
        )
        self._realtime_search_policy = RealtimeSearchPolicy(realtime_search_mode)
        self._realtime_search_strategy = normalize_realtime_search_strategy(
            realtime_search_strategy
        )
        self._realtime_search_model = realtime_search_model.strip()
        self._realtime_search_decisions = 0
        self._realtime_search_requests = 0
        self._realtime_search_bypasses = 0
        self._realtime_search_failures = 0
        self._realtime_search_last_reason: str | None = None
        self._realtime_search_last_route_ms: float | None = None
        self._realtime_search_last_first_content_ms: float | None = None
        self._reply_state: ContextVar[_ReplyStreamState | None] = ContextVar(
            f"reply-stream-state-{id(self)}",
            default=None,
        )

    @property
    def reply_max_sentences(self) -> int:
        return self._reply_max_sentences

    @property
    def realtime_search_mode(self) -> str:
        return self._realtime_search_policy.mode

    def realtime_search_snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.realtime_search_mode,
            "strategy": self._realtime_search_strategy,
            "model": self._realtime_search_model or self._settings.model,
            "decisions": self._realtime_search_decisions,
            "search_requests": self._realtime_search_requests,
            "bypass_requests": self._realtime_search_bypasses,
            "failures": self._realtime_search_failures,
            "last_reason": self._realtime_search_last_reason,
            "last_route_ms": self._realtime_search_last_route_ms,
            "last_first_content_ms": self._realtime_search_last_first_content_ms,
        }

    async def _emit_visible_text(self, state: _ReplyStreamState, text: str) -> None:
        if not text:
            return
        await super()._push_llm_text(text)
        state.visible_text_emitted = True

    async def _emit_complete_sentences(
        self,
        state: _ReplyStreamState,
        *,
        final: bool,
    ) -> bool:
        sentences, state.buffer = _split_complete_sentences(
            state.buffer,
            final=final,
        )
        for sentence in sentences:
            if state.sentences_emitted >= self._reply_max_sentences:
                state.buffer = ""
                state.limit_reached = True
                return True
            await self._emit_visible_text(state, sentence)
            state.sentences_emitted += 1
            if state.sentences_emitted >= self._reply_max_sentences:
                state.buffer = ""
                state.limit_reached = True
                return True
        return False

    async def _push_llm_text(self, text: str):
        state = self._reply_state.get()
        if state is None:
            await super()._push_llm_text(text)
            return
        if not text or state.limit_reached:
            return

        state.buffer += text
        if await self._emit_complete_sentences(state, final=False):
            raise _ReplyLimitReached

    async def _finalize_reply(self, state: _ReplyStreamState) -> None:
        reached_limit = await self._emit_complete_sentences(state, final=True)
        if reached_limit:
            return

        tail = state.buffer
        state.buffer = ""
        if not tail.strip():
            return
        if state.finish_reason == "length":
            logger.warning(
                "[LLM REPLY LIMIT] discarded incomplete tail after provider "
                f"finish_reason=length chars={len(tail)}"
            )
            return

        # A natural provider EOF can omit punctuation. Preserve that complete
        # thought as one final sentence rather than dropping a valid reply.
        await self._emit_visible_text(state, tail)
        state.sentences_emitted += 1

    async def _track_completion_stream(
        self,
        stream,
        *,
        realtime_search_started_at: float | None = None,
    ):
        """Track finish_reason and close the provider stream on an early limit."""

        search_first_content_seen = False
        try:
            async for chunk in stream:
                if (
                    realtime_search_started_at is not None
                    and not search_first_content_seen
                    and _chunk_has_answer_content(chunk)
                ):
                    search_first_content_seen = True
                    elapsed_ms = (
                        time.monotonic() - realtime_search_started_at
                    ) * 1000.0
                    self._realtime_search_last_first_content_ms = elapsed_ms
                    logger.info(
                        "[LLM REALTIME SEARCH] first_content "
                        f"elapsed_ms={elapsed_ms:.1f}"
                    )
                state = self._reply_state.get()
                if state is not None:
                    for choice in getattr(chunk, "choices", None) or ():
                        finish_reason = getattr(choice, "finish_reason", None)
                        if finish_reason is not None:
                            state.finish_reason = str(finish_reason)
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception:
            if realtime_search_started_at is not None:
                self._realtime_search_failures += 1
            raise
        finally:
            with suppress(Exception):
                await _close_stream(stream)

    async def get_chat_completions(self, context):
        route_started_at = time.perf_counter()
        query = last_user_text(context)
        decision = self._realtime_search_policy.decide(query)
        route_ms = (time.perf_counter() - route_started_at) * 1000.0
        self._realtime_search_decisions += 1
        self._realtime_search_last_reason = decision.reason
        self._realtime_search_last_route_ms = route_ms

        if decision.enabled:
            self._realtime_search_requests += 1
            self._realtime_search_last_first_content_ms = None
            adapter = self.get_llm_adapter()
            invocation_params = adapter.get_llm_invocation_params(
                context,
                system_instruction=assert_given(self._settings.system_instruction),
                convert_developer_to_user=not self.supports_developer_role,
            )
            params = add_dashscope_search_params(
                self.build_chat_completion_params(invocation_params),
                decision,
                strategy=self._realtime_search_strategy,
            )
            if self._realtime_search_model:
                params["model"] = self._realtime_search_model
            logger.info(
                "[LLM REALTIME SEARCH] route "
                f"forced={int(decision.forced)} reason={decision.reason} "
                f"model={params.get('model', self._settings.model)} "
                f"route_ms={route_ms:.3f} query_chars={len(query)}"
            )
            provider_started_at = time.monotonic()
            try:
                stream = await self._client.chat.completions.create(**params)
            except Exception:
                self._realtime_search_failures += 1
                raise
            return self._track_completion_stream(
                stream,
                realtime_search_started_at=provider_started_at,
            )

        self._realtime_search_bypasses += 1
        if (
            not self._hedge_model
            or self._hedge_model == self._settings.model
        ):
            stream = await super().get_chat_completions(context)
            return self._track_completion_stream(stream)

        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating hedged chat from context "
            f"{adapter.get_messages_for_logging(context)}"
        )
        invocation_params = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        params = self.build_chat_completion_params(invocation_params)
        return self._track_completion_stream(
            self._stream_first_answer_chunk(params)
        )

    async def _stream_first_answer_chunk(self, params):
        """Yield the model whose stream produces meaningful answer text first."""
        queue: asyncio.Queue = asyncio.Queue()
        streams = {
            "primary": str(params["model"]),
            "hedge": self._hedge_model,
        }
        tasks = {}
        buffers = {"primary": [], "hedge": []}
        started_at = time.monotonic()

        async def produce(label: str, delay_sec: float):
            stream = None
            try:
                if delay_sec:
                    await asyncio.sleep(delay_sec)
                request_params = dict(params)
                request_params["model"] = streams[label]
                stream = await self._client.chat.completions.create(
                    **request_params
                )
                async for chunk in stream:
                    await queue.put((label, "chunk", chunk))
                await queue.put((label, "done", None))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await queue.put((label, "error", exc))
            finally:
                with suppress(Exception):
                    await _close_stream(stream)

        tasks["primary"] = asyncio.create_task(produce("primary", 0.0))
        tasks["hedge"] = asyncio.create_task(
            produce("hedge", self._hedge_delay_sec)
        )
        active = set(tasks)
        errors = []
        winner = None

        try:
            while winner is None and active:
                remaining = self._hedge_deadline_sec - (
                    time.monotonic() - started_at
                )
                if remaining <= 0:
                    raise TimeoutError("hedged LLM first-token deadline exceeded")
                try:
                    label, event, value = await asyncio.wait_for(
                        queue.get(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        "hedged LLM first-token deadline exceeded"
                    ) from exc

                if event == "chunk":
                    buffers[label].append(value)
                    if _chunk_has_answer_content(value):
                        winner = label
                        break
                elif event == "error":
                    errors.append(value)
                    active.discard(label)
                elif event == "done":
                    active.discard(label)

            if winner is None:
                if errors:
                    raise errors[-1]
                raise RuntimeError("hedged LLM streams ended without answer text")

            loser = "hedge" if winner == "primary" else "primary"
            tasks[loser].cancel()
            logger.info(
                f"[LLM HEDGE] winner={winner} model={streams[winner]} "
                f"first_content_ms={(time.monotonic() - started_at) * 1000.0:.1f}"
            )

            for chunk in buffers[winner]:
                yield chunk
            buffers[winner].clear()

            while True:
                label, event, value = await queue.get()
                if label != winner:
                    continue
                if event == "chunk":
                    yield value
                elif event == "error":
                    raise value
                else:
                    break
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def _process_context(self, context):
        removed = trim_context_history(context, self._history_max_messages)
        if removed:
            logger.info(
                f"[LLM CONTEXT] trimmed={removed} "
                f"remaining={len(context.get_messages())}"
            )

        state = _ReplyStreamState()
        state_token = self._reply_state.set(state)

        try:
            try:
                await super()._process_context(context)
            except _ReplyLimitReached:
                logger.info(
                    "[LLM REPLY LIMIT] completed at sentence boundary "
                    f"sentences={state.sentences_emitted}"
                )
                return
            except asyncio.CancelledError:
                raise
            except TIMEOUT_EXCEPTIONS:
                # Provider fragments still waiting in ``buffer`` have not reached
                # TTS. Discard them so neither timeout handling nor the next turn
                # can speak a half sentence.
                state.buffer = ""
                if state.visible_text_emitted:
                    raise

                # Pipecat's generic timeout path does not stop this timer. Without
                # doing it here, the next turn reports a stale multi-second TTFB.
                with suppress(Exception):
                    await self.stop_ttfb_metrics()

                # The fallback is already a complete sentence. Bypass the stream
                # buffer so it is visible before the timeout is re-raised through
                # Pipecat's existing error path.
                await self._emit_visible_text(state, self._timeout_fallback_text)
                state.sentences_emitted += 1
                logger.warning(
                    "[LLM RECOVERY] timeout fallback pushed paired_with_user=1"
                )
                raise

            await self._finalize_reply(state)
        finally:
            state.buffer = ""
            self._reply_state.reset(state_token)
