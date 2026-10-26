"""CPU regressions for producer backpressure and deterministic TTS cleanup."""

import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voice_service.official_voxcpm2_backend import OfficialPromptCacheBackend
from voice_service.voxcpm2_server import NanoVoxCPM2Backend, VoxCPM2WebSocketBridge


class _BurstModel:
    sample_rate = 1000

    def __init__(self, count=1000):
        self.tts_model = self
        self.count = count
        self.produced = 0
        self.closed = False
        self.model_closed = False

    def build_prompt_cache(self, **_kwargs):
        return {"revision": 0}

    def generate_with_prompt_cache_streaming(self, **_kwargs):
        try:
            for _ in range(self.count):
                self.produced += 1
                yield np.array([0.25], dtype=np.float32), None, [np.ones((1, 1, 2))]
        finally:
            self.closed = True

    def merge_prompt_cache(self, cache, _text, _feature):
        return {"revision": cache["revision"] + 1}

    def close(self):
        self.model_closed = True


class VoxCPM2InfrastructureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.reference = Path(temporary.name) / "reference.wav"
        self.reference.write_bytes(b"fake")

    def _backend(self, model, *, factory=None):
        with patch.dict(os.environ, {"VOXCPM2_PCM_QUEUE_CHUNKS": "2"}):
            return OfficialPromptCacheBackend(
                model_path="fake", reference_wav=str(self.reference),
                model_factory=factory or (lambda *_args, **_kwargs: model),
            )

    async def test_slow_consumer_is_bounded_and_close_does_not_commit(self):
        model = _BurstModel()
        backend = self._backend(model)
        await backend.start()
        stream = backend.generate_pcm16("long", "turn")
        try:
            await anext(stream)
            await asyncio.sleep(0.1)
            # One consumed chunk, two queued chunks and one producer-held chunk.
            self.assertLessEqual(model.produced, 4)
        finally:
            await asyncio.wait_for(stream.aclose(), 1)
            self.assertEqual(backend._contexts["turn"].cache["revision"], 0)
            await backend.stop()
        self.assertTrue(model.closed)
        self.assertEqual(backend._active_tasks, set())

    async def test_backpressure_preserves_every_pcm_sample_and_cache_commit(self):
        model = _BurstModel(count=37)
        backend = self._backend(model)
        await backend.start()
        try:
            chunks = []
            async for chunk in backend.generate_pcm16("all", "turn"):
                chunks.append(chunk)
                await asyncio.sleep(0)
            self.assertEqual(chunks, [b"\xff\x1f"] * 37)
            self.assertEqual(backend._contexts["turn"].cache["revision"], 1)
        finally:
            await backend.stop()

    async def test_release_unblocks_full_producer_without_stale_pcm(self):
        model = _BurstModel()
        backend = self._backend(model)
        await backend.start()
        stream = backend.generate_pcm16("long", "turn")
        try:
            await anext(stream)
            await asyncio.sleep(0.1)
            await backend.release_context("turn")
            remaining = await asyncio.wait_for(
                self._collect(stream), 1,
            )
            self.assertEqual(remaining, [])
            self.assertTrue(model.closed)
        finally:
            await stream.aclose()
            await backend.stop()

    async def test_stalled_consumer_releases_model_for_another_context(self):
        model = _BurstModel(count=16)
        with patch.dict(os.environ, {"VOXCPM2_PCM_STALL_TIMEOUT_SEC": "0.1"}):
            backend = self._backend(model)
        await backend.start()
        stalled = backend.generate_pcm16("long", "stalled")
        try:
            await anext(stalled)
            # Keep the first stream open and abandoned while another turn runs.
            following = await asyncio.wait_for(
                self._collect(backend.generate_pcm16("next", "following")), 2,
            )
            self.assertEqual(len(following), 16)
            self.assertEqual(backend._contexts["following"].cache["revision"], 1)
            with self.assertRaisesRegex(TimeoutError, "PCM consumer stalled"):
                await asyncio.wait_for(self._collect(stalled), 1)
            self.assertEqual(backend._contexts["stalled"].cache["revision"], 0)
        finally:
            await stalled.aclose()
            await backend.stop()

    @staticmethod
    async def _collect(stream):
        return [chunk async for chunk in stream]

    async def test_cancelled_model_load_closes_model_before_start_returns(self):
        model = _BurstModel()
        entered = threading.Event()
        release = threading.Event()

        def factory(*_args, **_kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test load gate timed out")
            return model

        backend = self._backend(model, factory=factory)
        start = asyncio.create_task(backend.start())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            start.cancel()
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await start
            self.assertTrue(model.model_closed)
            self.assertIsNone(backend._model)
        finally:
            release.set()
            await backend.stop()

    async def test_repeated_cancel_during_load_cleanup_still_closes_model(self):
        model = _BurstModel()
        entered = threading.Event()
        release = threading.Event()

        def factory(*_args, **_kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test load gate timed out")
            return model

        backend = self._backend(model, factory=factory)
        start = asyncio.create_task(backend.start())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            start.cancel()
            await asyncio.sleep(0)
            start.cancel()
            await asyncio.sleep(0)
            self.assertFalse(start.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(start, 1)
            self.assertTrue(model.model_closed)
        finally:
            release.set()
            await backend.stop()

    async def test_terminal_worker_error_fits_behind_full_pcm_queue(self):
        failed = threading.Event()

        class ErrorModel(_BurstModel):
            def generate_with_prompt_cache_streaming(self, **kwargs):
                yield from super().generate_with_prompt_cache_streaming(**kwargs)
                failed.set()
                raise RuntimeError("worker failed at end")

        model = ErrorModel(count=3)
        backend = self._backend(model)
        await backend.start()
        stream = backend.generate_pcm16("fail", "turn")
        try:
            await anext(stream)
            # Consumed one, leaving two queued chunks plus the terminal error.
            self.assertTrue(await asyncio.to_thread(failed.wait, 1))
            with self.assertRaisesRegex(RuntimeError, "worker failed at end"):
                await asyncio.wait_for(self._collect(stream), 1)
            self.assertEqual(backend._contexts["turn"].cache["revision"], 0)
            self.assertEqual(backend._active_tasks, set())
        finally:
            await stream.aclose()
            await backend.stop()

    async def test_send_failure_closes_backend_generator_in_same_task(self):
        closed = []
        owner = asyncio.current_task()

        class Backend:
            async def generate_pcm16(self, *_args):
                try:
                    yield b"\x01\x00"
                    await asyncio.Future()
                finally:
                    closed.append(asyncio.current_task())

        class Socket:
            async def send(self, _message):
                raise ConnectionError("client gone")

        bridge = VoxCPM2WebSocketBridge(Backend())
        with self.assertRaisesRegex(ConnectionError, "client gone"):
            await bridge._stream_audio(Socket(), "request", "turn", "hello")
        self.assertEqual(closed, [owner])

    async def test_nano_consumer_close_closes_pool_generator_immediately(self):
        closed = []

        class Pool:
            async def generate(self, **_kwargs):
                try:
                    yield np.array([0.25], dtype=np.float32)
                    await asyncio.Future()
                finally:
                    closed.append(True)

        backend = NanoVoxCPM2Backend(
            model_path="fake", prompt_wav="fake", prompt_text="ref", devices=[0],
        )
        backend._pool = Pool()
        backend._prompt_id = "prompt"
        backend.sample_rate = 1000
        stream = backend.generate_pcm16("text", "turn")
        await anext(stream)
        await stream.aclose()
        self.assertEqual(closed, [True])

    async def test_send_failure_releases_request_context(self):
        released = []

        class Backend:
            sample_rate = 1000

            async def generate_pcm16(self, *_args):
                yield b"\x01\x00"

            async def release_context(self, context_id):
                released.append(context_id)

        class Socket:
            first = True

            async def recv(self):
                if self.first:
                    self.first = False
                    return json.dumps({
                        "type": "synthesize", "request_id": "request",
                        "context_id": "turn", "text": "hello",
                    })
                await asyncio.Future()

            async def send(self, message):
                if isinstance(message, bytes):
                    raise ConnectionError("client gone")

        await VoxCPM2WebSocketBridge(Backend()).handle_connection(Socket())
        self.assertEqual(released, ["turn"])

    async def test_cancelling_blocked_send_closes_and_releases_before_exit(self):
        sending = asyncio.Event()
        closed = []
        released = []

        class Backend:
            sample_rate = 1000

            async def generate_pcm16(self, *_args):
                try:
                    yield b"\x01\x00"
                    await asyncio.Future()
                finally:
                    closed.append(True)

            async def release_context(self, context_id):
                released.append(context_id)

        class Socket:
            first = True

            async def recv(self):
                if self.first:
                    self.first = False
                    return json.dumps({
                        "type": "synthesize", "request_id": "request",
                        "context_id": "turn", "text": "hello",
                    })
                await asyncio.Future()

            async def send(self, message):
                if isinstance(message, bytes):
                    sending.set()
                    await asyncio.Future()

        task = asyncio.create_task(
            VoxCPM2WebSocketBridge(Backend()).handle_connection(Socket())
        )
        await asyncio.wait_for(sending.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(closed, [True])
        self.assertEqual(released, ["turn"])

    async def test_hung_pcm_and_error_sends_time_out_and_release_context(self):
        closed = []
        released = []

        class Backend:
            sample_rate = 1000

            async def generate_pcm16(self, *_args):
                try:
                    yield b"\x01\x00"
                    await asyncio.Future()
                finally:
                    closed.append(True)

            async def release_context(self, context_id):
                released.append(context_id)

        class Socket:
            first = True

            async def recv(self):
                if self.first:
                    self.first = False
                    return json.dumps({
                        "type": "synthesize", "request_id": "request",
                        "context_id": "turn", "text": "hello",
                    })
                await asyncio.Future()

            async def send(self, message):
                if isinstance(message, str) and json.loads(message)["type"] == "start":
                    return
                await asyncio.Future()

        with patch.dict(os.environ, {"VOXCPM2_SEND_TIMEOUT_SEC": "0.05"}):
            bridge = VoxCPM2WebSocketBridge(Backend())
        await asyncio.wait_for(bridge.handle_connection(Socket()), 1)
        self.assertEqual(closed, [True])
        self.assertEqual(released, ["turn"])

    def test_stall_and_send_timeouts_must_be_positive_and_finite(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"VOXCPM2_PCM_STALL_TIMEOUT_SEC": value}):
                    with self.assertRaises(ValueError):
                        self._backend(_BurstModel())
                with patch.dict(os.environ, {"VOXCPM2_SEND_TIMEOUT_SEC": value}):
                    with self.assertRaises(ValueError):
                        VoxCPM2WebSocketBridge(_BurstModel())


if __name__ == "__main__":
    unittest.main()
