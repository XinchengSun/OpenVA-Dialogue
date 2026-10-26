"""Exercise hedge stream lifecycle without importing Pipecat or provider SDKs."""

from __future__ import annotations

import ast
import asyncio
from contextlib import suppress
from pathlib import Path
import time
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "pipecat_dystream" / "resilient_llm.py"


def load_hedge_stream(source=SOURCE):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "_chunk_has_answer_content", "_close_stream",
        }:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.startswith("_HEDGE_")
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "RecoveringOpenAILLMService":
            selected.extend(
                method for method in node.body
                if isinstance(method, ast.AsyncFunctionDef)
                and method.name == "_stream_first_answer_chunk"
            )
    namespace = {
        "asyncio": asyncio, "suppress": suppress, "time": time,
        "logger": SimpleNamespace(info=lambda *_args: None),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_stream_first_answer_chunk"]


def chunk(text=None, *, reasoning=None, finish_reason=None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=text, reasoning_content=reasoning,
                                  role="assistant", tool_calls=None, audio=None),
            finish_reason=finish_reason,
        )],
        usage=usage,
    )


class FakeStream:
    def __init__(self, chunks=(), *, before_read=None, failure=None, close_gate=None):
        self.chunks = iter(chunks)
        self.before_read = before_read
        self.failure = failure
        self.close_gate = close_gate
        self.read_count = 0
        self.close_count = 0
        self.close_started = asyncio.Event()
        self.close_cancelled = False
        self.closed = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.before_read is not None:
            await self.before_read.wait()
        try:
            value = next(self.chunks)
        except StopIteration:
            if self.failure is not None:
                raise self.failure
            raise StopAsyncIteration
        self.read_count += 1
        return value

    async def close(self):
        self.close_count += 1
        self.close_started.set()
        try:
            if self.close_gate is not None:
                await self.close_gate.wait()
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise
        finally:
            self.closed.set()


def make_service(primary, hedge, *, delay=0.0, deadline=1.0, on_create=None):
    async def create(**params):
        model = params["model"]
        if on_create is not None:
            on_create(model)
        return primary if model == "primary-model" else hedge

    return SimpleNamespace(
        _client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        _hedge_model="hedge-model", _hedge_delay_sec=delay, _hedge_deadline_sec=deadline,
    )


class HedgeBackpressureTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_consumer_bounds_provider_read_ahead(self):
        primary = FakeStream(chunk(str(index)) for index in range(2_000))
        service = make_service(primary, FakeStream(), delay=10.0)
        stream = load_hedge_stream()(service, {"model": "primary-model"})
        try:
            self.assertEqual((await anext(stream)).choices[0].delta.content, "0")
            for _ in range(10):
                await asyncio.sleep(0)
            # A small bounded queue plus the producer's one in-flight chunk.
            self.assertLessEqual(primary.read_count, 32)
        finally:
            await stream.aclose()
        self.assertEqual(primary.close_count, 1)

    async def test_loser_is_cancelled_and_joined_when_consumer_closes(self):
        hedge_opened = asyncio.Event()
        primary = FakeStream([chunk("answer")], before_read=hedge_opened)
        hedge = FakeStream(before_read=asyncio.Event())
        service = make_service(
            primary, hedge,
            on_create=lambda model: hedge_opened.set() if model == "hedge-model" else None,
        )
        stream = load_hedge_stream()(service, {"model": "primary-model"})
        try:
            self.assertEqual((await anext(stream)).choices[0].delta.content, "answer")
        finally:
            await stream.aclose()
        self.assertEqual(primary.close_count, 1)
        self.assertEqual(hedge.close_count, 1)

    async def test_slow_loser_close_does_not_delay_winner_first_token(self):
        hedge_opened = asyncio.Event()
        primary = FakeStream([chunk("answer")], before_read=hedge_opened)
        hedge = FakeStream(before_read=asyncio.Event(), close_gate=asyncio.Event())
        service = make_service(
            primary, hedge, deadline=0.1,
            on_create=lambda model: hedge_opened.set() if model == "hedge-model" else None,
        )
        stream = load_hedge_stream()(service, {"model": "primary-model"})
        try:
            value = await asyncio.wait_for(anext(stream), 0.1)
            self.assertEqual(value.choices[0].delta.content, "answer")
            await asyncio.wait_for(hedge.close_started.wait(), 0.5)
            self.assertFalse(hedge.closed.is_set())
        finally:
            await asyncio.wait_for(stream.aclose(), 0.5)
        self.assertTrue(hedge.close_cancelled)
        self.assertTrue(hedge.closed.is_set())

    async def test_loser_close_times_out_even_while_consumer_is_paused(self):
        hedge_opened = asyncio.Event()
        primary = FakeStream([chunk("answer")], before_read=hedge_opened)
        hedge = FakeStream(before_read=asyncio.Event(), close_gate=asyncio.Event())
        service = make_service(
            primary, hedge,
            on_create=lambda model: hedge_opened.set() if model == "hedge-model" else None,
        )
        method = load_hedge_stream()
        method.__globals__["_HEDGE_STREAM_CLOSE_TIMEOUT_SEC"] = 0.02
        stream = method(service, {"model": "primary-model"})
        try:
            await anext(stream)
            await asyncio.wait_for(hedge.closed.wait(), 0.5)
            self.assertTrue(hedge.close_cancelled)
        finally:
            await stream.aclose()

    async def test_cancelled_consumer_bounds_stalled_close_and_joins_cleanup(self):
        existing_tasks = asyncio.all_tasks()
        both_opened = asyncio.Event()
        primary = FakeStream(before_read=asyncio.Event(), close_gate=asyncio.Event())
        hedge = FakeStream(before_read=asyncio.Event(), close_gate=asyncio.Event())
        service = make_service(primary, hedge, on_create=lambda model: (
            both_opened.set() if model == "hedge-model" else None
        ))
        method = load_hedge_stream()
        method.__globals__["_HEDGE_STREAM_CLOSE_TIMEOUT_SEC"] = 0.02
        stream = method(service, {"model": "primary-model"})
        task = asyncio.create_task(anext(stream))
        await both_opened.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.5)
        self.assertTrue(primary.close_cancelled)
        self.assertTrue(hedge.close_cancelled)
        self.assertTrue(primary.closed.is_set())
        self.assertTrue(hedge.closed.is_set())
        self.assertEqual(asyncio.all_tasks() - existing_tasks, set())

    async def test_reasoning_only_candidate_is_bounded_while_hedge_can_win(self):
        primary = FakeStream(chunk(reasoning="hidden") for _ in range(10_000))
        answer = chunk("hedge answer")
        hedge = FakeStream([answer])
        service = make_service(primary, hedge, deadline=30.0)
        stream = load_hedge_stream()(service, {"model": "primary-model"})
        result = [value async for value in stream]
        self.assertEqual(result, [answer])
        self.assertLessEqual(primary.read_count, 65)
        self.assertEqual([primary.close_count, hedge.close_count], [1, 1])

    async def test_preserves_winner_metadata_answer_finish_and_usage_order(self):
        chunks = [chunk(), chunk(reasoning="brief"), chunk("answer"),
                  chunk(finish_reason="stop"), chunk(usage={"total_tokens": 4})]
        primary = FakeStream(chunks)
        service = make_service(primary, FakeStream(), delay=10.0)
        result = [value async for value in load_hedge_stream()(service, {"model": "primary-model"})]
        self.assertEqual(result, chunks)
        self.assertEqual(primary.close_count, 1)

    async def test_excess_preanswer_metadata_fails_explicitly_without_unbounded_buffer(self):
        primary = FakeStream(chunk(reasoning="hidden") for _ in range(1_000))
        hedge = FakeStream(chunk(reasoning="hidden") for _ in range(1_000))
        service = make_service(primary, hedge, deadline=30.0)
        with self.assertRaisesRegex(RuntimeError, "exceeded 64 pre-answer chunks"):
            await anext(load_hedge_stream()(service, {"model": "primary-model"}))
        self.assertEqual([primary.read_count, hedge.read_count], [65, 65])
        self.assertEqual([primary.close_count, hedge.close_count], [1, 1])

    async def test_error_after_answer_is_forwarded_and_provider_closed(self):
        primary = FakeStream([chunk("answer")], failure=ValueError("provider failed"))
        service = make_service(primary, FakeStream(), delay=10.0)
        stream = load_hedge_stream()(service, {"model": "primary-model"})
        self.assertEqual((await anext(stream)).choices[0].delta.content, "answer")
        with self.assertRaisesRegex(ValueError, "provider failed"):
            await anext(stream)
        self.assertEqual(primary.close_count, 1)

    async def test_empty_candidates_fail_and_close(self):
        primary, hedge = FakeStream(), FakeStream()
        service = make_service(primary, hedge)
        with self.assertRaisesRegex(RuntimeError, "without answer text"):
            await anext(load_hedge_stream()(service, {"model": "primary-model"}))
        self.assertEqual([primary.close_count, hedge.close_count], [1, 1])

    async def test_deadline_closes_both_stalled_providers(self):
        primary = FakeStream(before_read=asyncio.Event())
        hedge = FakeStream(before_read=asyncio.Event())
        service = make_service(primary, hedge, deadline=0.2)
        with self.assertRaisesRegex(TimeoutError, "deadline exceeded"):
            await anext(load_hedge_stream()(service, {"model": "primary-model"}))
        self.assertEqual([primary.close_count, hedge.close_count], [1, 1])

    async def test_cancelled_consumer_closes_both_stalled_providers(self):
        both_opened = asyncio.Event()
        primary = FakeStream(before_read=asyncio.Event())
        hedge = FakeStream(before_read=asyncio.Event())
        service = make_service(primary, hedge, on_create=lambda model: (
            both_opened.set() if model == "hedge-model" else None
        ))
        stream = load_hedge_stream()(service, {"model": "primary-model"})
        task = asyncio.create_task(anext(stream))
        await both_opened.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([primary.close_count, hedge.close_count], [1, 1])


if __name__ == "__main__":
    unittest.main()
