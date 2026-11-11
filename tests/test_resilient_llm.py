import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_dystream.resilient_llm import (
    RecoveringOpenAILLMService,
    _split_complete_sentences,
    _first_clause_end,
)


def completion_chunk(text=None, *, finish_reason=None):
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=text,
                    tool_calls=None,
                    audio=None,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


class FakeStream:
    def __init__(self, chunks, *, chunk_delay=0.0):
        self._chunks = iter(chunks)
        self._chunk_delay = chunk_delay
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunk_delay:
            await asyncio.sleep(self._chunk_delay)
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self):
        self.close_count += 1


class FailingStream(FakeStream):
    async def __anext__(self):
        raise httpx.ReadError("provider stream failed")


class FakeCompletions:
    def __init__(self, streams):
        self._streams = list(streams)
        self.calls = []

    async def create(self, **params):
        self.calls.append(params)
        return self._streams.pop(0)


def make_service(
    *,
    streams=(),
    max_sentences=3,
    first_clause_min_chars=0,
    realtime_search_mode="off",
    realtime_search_model="",
    hedge_model="",
    settings_extra=None,
):
    service = RecoveringOpenAILLMService(
        api_key="test-key",
        settings=RecoveringOpenAILLMService.Settings(
            model="test-model",
            system_instruction="用中文简短回答。",
            max_tokens=256,
            extra=settings_extra or {},
        ),
        timeout_fallback_text="网络有点慢，请重试。",
        history_max_messages=12,
        reply_max_sentences=max_sentences,
        first_clause_min_chars=first_clause_min_chars,
        realtime_search_mode=realtime_search_mode,
        realtime_search_model=realtime_search_model,
        hedge_model=hedge_model,
    )
    completions = FakeCompletions(streams)
    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    service.push_frame = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    return service, completions


def emitted_texts(service):
    return [
        call.args[0].text
        for call in service.push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    ]


class SentenceParserTests(unittest.TestCase):
    def test_decimal_point_is_not_a_sentence_boundary(self):
        sentences, tail = _split_complete_sentences(
            "数值是3.14。下一句。",
            final=True,
        )
        self.assertEqual(sentences, ["数值是3.14。", "下一句。"])
        self.assertEqual(tail, "")

    def test_closing_quote_stays_with_its_sentence_across_chunks(self):
        sentences, tail = _split_complete_sentences("他说：“你好。", final=False)
        self.assertEqual(sentences, [])
        self.assertEqual(tail, "他说：“你好。")


class ReplySentenceLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_clause_preserves_text_and_sentence_budget(self):
        first = "每天傍晚我都会独自沿着河堤散步，"
        tail = "听桥下的流水慢慢讲完一天的故事。"
        stream = FakeStream([completion_chunk(first), completion_chunk(tail[:3]),
                             completion_chunk(tail[3:]), completion_chunk("第二句。"),
                             completion_chunk(finish_reason="stop")])
        service, _ = make_service(streams=[stream], max_sentences=1, first_clause_min_chars=12)
        await service._process_context(LLMContext())
        self.assertEqual(emitted_texts(service), [first, tail])

    def test_first_clause_keeps_an_already_complete_sentence_together(self):
        self.assertEqual(_first_clause_end("每天傍晚我都会独自沿着河堤散步，听流水讲故事。", 12), 0)

    async def test_comma_and_semicolon_in_successive_streamed_sentences(self):
        first = "每天傍晚我都会独自沿着河堤散步"
        second = "山上最后一层薄雾已经慢慢散去"
        for comma, semicolon, space in (("，", "；", ""), (",", ";", " ")):
            pieces = [first, comma + space, "听流水", "讲故事。", second, semicolon + space,
                      "我们就", "会看到小屋"]
            stream = FakeStream([*(completion_chunk(piece) for piece in pieces),
                                 completion_chunk(finish_reason="stop")])
            service, _ = make_service(streams=[stream], max_sentences=2, first_clause_min_chars=12)
            await service._process_context(LLMContext())
            self.assertEqual(emitted_texts(service), [
                first + comma, space + "听流水讲故事。",
                second + semicolon, space + "我们就会看到小屋",
            ])

    def test_first_clause_ignores_short_phrases_numbers_and_non_chinese(self):
        for text in ("好呀，我来讲个故事。", "今天天气晴朗适合带着孩子出去玩，好。",
                     "今年报名的总人数已经超过12,345人",
                     "“等到山上最后一层薄雾慢慢散去；我们就能看见",
                     "a long sentence in English, with another clause"):
            self.assertEqual(_first_clause_end(text, 12), 0)

    async def test_streams_only_three_complete_sentences_and_closes_early(self):
        stream = FakeStream(
            [
                completion_chunk("第一"),
                completion_chunk("句。第二句"),
                completion_chunk("！第三句。"),
                completion_chunk("第四句。"),
                completion_chunk(finish_reason="stop"),
            ]
        )
        service, completions = make_service(streams=[stream])
        context = LLMContext([{"role": "user", "content": "介绍一下"}])

        await service._process_context(context)

        self.assertEqual(
            emitted_texts(service),
            ["第一句。", "第二句！", "第三句。"],
        )
        self.assertEqual(stream.close_count, 1)
        self.assertEqual(completions.calls[0]["max_tokens"], 256)

    async def test_natural_eof_flushes_unpunctuated_tail(self):
        stream = FakeStream(
            [
                completion_chunk("这是自然结束的完整回答"),
                completion_chunk(finish_reason="stop"),
            ]
        )
        service, _ = make_service(streams=[stream])

        await service._process_context(LLMContext())

        self.assertEqual(emitted_texts(service), ["这是自然结束的完整回答"])
        self.assertEqual(stream.close_count, 1)

    async def test_length_finish_discards_only_incomplete_tail(self):
        stream = FakeStream(
            [
                completion_chunk("第一句。第二句说到一半"),
                completion_chunk(finish_reason="length"),
            ]
        )
        service, _ = make_service(streams=[stream])

        await service._process_context(LLMContext())

        self.assertEqual(emitted_texts(service), ["第一句。"])
        self.assertEqual(stream.close_count, 1)

    async def test_timeout_before_complete_sentence_uses_fallback(self):
        service, _ = make_service()

        async def partial_then_timeout(bound_service, _context):
            await bound_service._push_llm_text("只说到一半")
            raise httpx.ReadTimeout("provider stalled")

        with patch.object(
            OpenAILLMService,
            "_process_context",
            new=partial_then_timeout,
        ):
            with self.assertRaises(httpx.ReadTimeout):
                await service._process_context(LLMContext())

        self.assertEqual(emitted_texts(service), ["网络有点慢，请重试。"])

    async def test_timeout_after_complete_sentence_adds_no_fallback_or_fragment(self):
        service, _ = make_service()

        async def sentence_then_timeout(bound_service, _context):
            await bound_service._push_llm_text("第一句。第二句说到一半")
            raise httpx.ReadTimeout("provider stalled")

        with patch.object(
            OpenAILLMService,
            "_process_context",
            new=sentence_then_timeout,
        ):
            with self.assertRaises(httpx.ReadTimeout):
                await service._process_context(LLMContext())

        self.assertEqual(emitted_texts(service), ["第一句。"])

    async def test_concurrent_requests_do_not_share_sentence_buffers(self):
        streams = [
            FakeStream(
                [completion_chunk("甲句。"), completion_chunk(finish_reason="stop")],
                chunk_delay=0.001,
            ),
            FakeStream(
                [completion_chunk("乙句。"), completion_chunk(finish_reason="stop")],
                chunk_delay=0.001,
            ),
        ]
        service, _ = make_service(streams=streams)

        await asyncio.gather(
            service._process_context(LLMContext()),
            service._process_context(LLMContext()),
        )

        self.assertCountEqual(emitted_texts(service), ["甲句。", "乙句。"])
        self.assertEqual([stream.close_count for stream in streams], [1, 1])

    async def test_explicit_default_remains_three_and_negative_limit_is_rejected(self):
        service, _ = make_service()
        self.assertEqual(service.reply_max_sentences, 3)
        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            make_service(max_sentences=-1)

    async def test_unlimited_story_retains_all_sentences_and_ending(self):
        story = ["小镇有一盏旧台灯。", "店主想把它扔掉。", "孩子把断线接好了。",
                 "那晚停电，台灯照亮了楼道。", "邻居们围在灯下讲故事。", "店主从此把它留在窗前。"]
        stream = FakeStream([completion_chunk(s) for s in story] + [completion_chunk(finish_reason="stop")])
        service, _ = make_service(streams=[stream], max_sentences=0)
        await service._process_context(LLMContext([{"role":"user", "content":"你讲一个故事"}]))
        self.assertEqual("".join(emitted_texts(service)), "".join(story))
        self.assertEqual(len(emitted_texts(service)), 6)
        self.assertEqual(stream.close_count, 1)


class RealtimeSearchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_chat_keeps_original_request_params(self):
        stream = FakeStream([completion_chunk("你好。"), completion_chunk(finish_reason="stop")])
        service, completions = make_service(
            streams=[stream],
            realtime_search_mode="smart",
        )

        returned = await service.get_chat_completions(
            LLMContext([{"role": "user", "content": "给我讲个笑话"}])
        )
        async for _ in returned:
            pass

        self.assertEqual(len(completions.calls), 1)
        self.assertNotIn("extra_body", completions.calls[0])
        snapshot = service.realtime_search_snapshot()
        self.assertEqual(snapshot["bypass_requests"], 1)
        self.assertEqual(snapshot["search_requests"], 0)

    async def test_ordinary_chat_calls_the_exact_parent_path(self):
        stream = FakeStream([completion_chunk("普通回答。")])
        service, completions = make_service(realtime_search_mode="smart")
        context = LLMContext([{"role": "user", "content": "你好"}])

        with patch.object(
            OpenAILLMService,
            "get_chat_completions",
            new=AsyncMock(return_value=stream),
        ) as parent_call:
            returned = await service.get_chat_completions(context)
            async for _ in returned:
                pass

        parent_call.assert_awaited_once_with(context)
        self.assertEqual(completions.calls, [])

    async def test_off_mode_bypasses_even_a_live_query(self):
        stream = FakeStream([completion_chunk("父类回答。")])
        service, completions = make_service(realtime_search_mode="off")
        context = LLMContext([{"role": "user", "content": "杭州天气怎么样？"}])

        with patch.object(
            OpenAILLMService,
            "get_chat_completions",
            new=AsyncMock(return_value=stream),
        ) as parent_call:
            returned = await service.get_chat_completions(context)
            async for _ in returned:
                pass

        parent_call.assert_awaited_once_with(context)
        self.assertEqual(completions.calls, [])

    async def test_only_the_latest_user_turn_controls_routing(self):
        stream = FakeStream([completion_chunk("普通回答。")])
        service, completions = make_service(realtime_search_mode="smart")
        context = LLMContext(
            [
                {"role": "user", "content": "杭州天气怎么样？"},
                {"role": "assistant", "content": "旧回答"},
                {"role": "user", "content": "给我讲个笑话"},
            ]
        )

        with patch.object(
            OpenAILLMService,
            "get_chat_completions",
            new=AsyncMock(return_value=stream),
        ) as parent_call:
            returned = await service.get_chat_completions(context)
            async for _ in returned:
                pass

        parent_call.assert_awaited_once_with(context)
        self.assertEqual(completions.calls, [])

    async def test_live_query_uses_one_forced_search_stream(self):
        stream = FakeStream([completion_chunk("杭州今天有雨。"), completion_chunk(finish_reason="stop")])
        service, completions = make_service(
            streams=[stream],
            realtime_search_mode="smart",
            hedge_model="backup-model",
            settings_extra={
                "extra_body": {
                    "enable_thinking": False,
                    "search_options": {"enable_source": False},
                }
            },
        )

        returned = await service.get_chat_completions(
            LLMContext([{"role": "user", "content": "杭州今天天气怎么样？"}])
        )
        async for _ in returned:
            pass

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(completions.calls[0]["model"], "test-model")
        self.assertEqual(
            completions.calls[0]["extra_body"],
            {
                "enable_thinking": False,
                "enable_search": True,
                "search_options": {
                    "enable_source": False,
                    "search_strategy": "turbo",
                    "forced_search": True,
                },
            },
        )
        self.assertEqual(
            service._settings.extra,
            {
                "extra_body": {
                    "enable_thinking": False,
                    "search_options": {"enable_source": False},
                }
            },
        )
        snapshot = service.realtime_search_snapshot()
        self.assertEqual(snapshot["search_requests"], 1)
        self.assertEqual(snapshot["bypass_requests"], 0)
        self.assertEqual(snapshot["last_reason"], "live_subject")
        self.assertIsNotNone(snapshot["last_first_content_ms"])

    async def test_live_query_can_use_a_dedicated_search_model(self):
        stream = FakeStream([
            completion_chunk("杭州今天有雨。"),
            completion_chunk(finish_reason="stop"),
        ])
        service, completions = make_service(
            streams=[stream],
            realtime_search_mode="smart",
            realtime_search_model="qwen-flash",
        )

        returned = await service.get_chat_completions(
            LLMContext([{"role": "user", "content": "杭州今天天气怎么样？"}])
        )
        async for _ in returned:
            pass

        self.assertEqual(completions.calls[0]["model"], "qwen-flash")
        self.assertEqual(
            service.realtime_search_snapshot()["model"],
            "qwen-flash",
        )

    async def test_cancelling_search_consumer_closes_provider_stream(self):
        stream = FakeStream(
            [completion_chunk("迟到的搜索结果。")],
            chunk_delay=1.0,
        )
        service, _ = make_service(
            streams=[stream],
            realtime_search_mode="smart",
        )
        returned = await service.get_chat_completions(
            LLMContext([{"role": "user", "content": "查一下今天的新闻"}])
        )

        async def consume():
            async for _ in returned:
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(stream.close_count, 1)

    async def test_search_stream_failure_is_counted_and_closed(self):
        stream = FailingStream([])
        service, _ = make_service(
            streams=[stream],
            realtime_search_mode="smart",
        )
        returned = await service.get_chat_completions(
            LLMContext([{"role": "user", "content": "今天有什么新闻"}])
        )

        with self.assertRaises(httpx.ReadError):
            async for _ in returned:
                pass

        self.assertEqual(stream.close_count, 1)
        self.assertEqual(service.realtime_search_snapshot()["failures"], 1)


if __name__ == "__main__":
    unittest.main()
