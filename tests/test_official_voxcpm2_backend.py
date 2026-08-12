import asyncio
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from voice_service.official_voxcpm2_backend import OfficialPromptCacheBackend
from voice_service.voxcpm2_server import _backend_from_env


class _FakeTTSModel:
    def __init__(self, waveforms=None, *, sample_rate=1_000):
        self.sample_rate = sample_rate
        self.waveforms = waveforms or {
            "hello": [np.asarray([0.25, -0.25], dtype=np.float32)]
        }
        self.build_calls = []
        self.generate_calls = []
        self.merge_calls = []

    def build_prompt_cache(self, **kwargs):
        self.build_calls.append(kwargs)
        return {"revision": 0, "reference": "base"}

    def generate_with_prompt_cache_streaming(self, **kwargs):
        self.generate_calls.append(
            {
                **kwargs,
                "thread_id": threading.get_ident(),
                "cache_revision": kwargs["prompt_cache"]["revision"],
            }
        )
        for index, waveform in enumerate(self.waveforms[kwargs["target_text"]]):
            feature = np.full((1, 1, 2), index + 1, dtype=np.float32)
            yield waveform, None, [feature]

    def merge_prompt_cache(self, cache, text, feature):
        self.merge_calls.append((cache["revision"], text, np.asarray(feature).copy()))
        return {
            "revision": cache["revision"] + 1,
            "reference": cache["reference"],
            "text": text,
        }


class _FakeModel:
    def __init__(self, tts_model):
        self.tts_model = tts_model
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _Factory:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def __call__(self, model_path, **kwargs):
        self.calls.append((model_path, kwargs, threading.get_ident()))
        return self.model


class _BlockingTTSModel(_FakeTTSModel):
    def __init__(self, *, block_text="first"):
        super().__init__(
            {
                "first": [np.asarray([0.1], dtype=np.float32)],
                "second": [np.asarray([0.2], dtype=np.float32)],
            }
        )
        self.block_text = block_text
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate_with_prompt_cache_streaming(self, **kwargs):
        self.generate_calls.append(
            {
                **kwargs,
                "thread_id": threading.get_ident(),
                "cache_revision": kwargs["prompt_cache"]["revision"],
            }
        )
        if kwargs["target_text"] == self.block_text:
            self.entered.set()
            if not self.release.wait(timeout=2.0):
                raise RuntimeError("test gate timed out")
        feature = np.ones((1, 1, 2), dtype=np.float32)
        yield self.waveforms[kwargs["target_text"]][0], None, [feature]


class _PauseAfterFirstYieldTTSModel(_FakeTTSModel):
    def __init__(self):
        super().__init__(
            {
                "cancelled": [np.asarray([0.1], dtype=np.float32)],
                "retry": [np.asarray([0.2], dtype=np.float32)],
            }
        )
        self.after_first = threading.Event()
        self.release = threading.Event()

    def generate_with_prompt_cache_streaming(self, **kwargs):
        text = kwargs["target_text"]
        self.generate_calls.append(
            {
                **kwargs,
                "thread_id": threading.get_ident(),
                "cache_revision": kwargs["prompt_cache"]["revision"],
            }
        )
        feature = np.ones((1, 1, 2), dtype=np.float32)
        yield self.waveforms[text][0], None, [feature]
        if text == "cancelled":
            self.after_first.set()
            if not self.release.wait(timeout=2.0):
                raise RuntimeError("test gate timed out")
            yield np.asarray([0.3], dtype=np.float32), None, [feature]


class _MergeBlockingTTSModel(_FakeTTSModel):
    def __init__(self):
        super().__init__({"race": [np.asarray([0.1], dtype=np.float32)]})
        self.merge_entered = threading.Event()
        self.merge_release = threading.Event()

    def merge_prompt_cache(self, cache, text, feature):
        self.merge_calls.append((cache["revision"], text, np.asarray(feature).copy()))
        self.merge_entered.set()
        if not self.merge_release.wait(timeout=2.0):
            raise RuntimeError("test gate timed out")
        return {"revision": cache["revision"] + 1, "reference": "base"}


class _ErrorTTSModel(_FakeTTSModel):
    def __init__(self):
        super().__init__({"broken": [np.asarray([0.1], dtype=np.float32)]})

    def generate_with_prompt_cache_streaming(self, **kwargs):
        self.generate_calls.append(
            {
                **kwargs,
                "thread_id": threading.get_ident(),
                "cache_revision": kwargs["prompt_cache"]["revision"],
            }
        )
        yield np.asarray([0.1], dtype=np.float32), None, [
            np.ones((1, 1, 2), dtype=np.float32)
        ]
        raise RuntimeError("generation failed")


class OfficialPromptCacheBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.prompt_wav = Path(self._temporary_directory.name) / "reference.wav"
        self.prompt_wav.write_bytes(b"RIFFfake")

    def _make_backend(self, tts_model, **overrides):
        model = _FakeModel(tts_model)
        factory = _Factory(model)
        backend = OfficialPromptCacheBackend(
            model_path="official-model",
            reference_wav=str(self.prompt_wav),
            model_factory=factory,
            **overrides,
        )
        return backend, model, factory

    @staticmethod
    async def _collect(backend, text, context_id):
        return [
            chunk
            async for chunk in backend.generate_pcm16(text, context_id)
        ]

    @staticmethod
    def _pcm16(samples):
        values = np.asarray(samples, dtype=np.float32)
        return (np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    async def test_start_builds_reference_only_cache_and_context_is_lazy(self):
        tts_model = _FakeTTSModel()
        backend, model, factory = self._make_backend(tts_model)
        main_thread = threading.get_ident()

        await backend.start()

        self.assertEqual(
            tts_model.build_calls,
            [{"reference_wav_path": str(self.prompt_wav)}],
        )
        self.assertEqual(backend.sample_rate, 1_000)
        self.assertEqual(backend._contexts, {})
        self.assertEqual(factory.calls[0][0], "official-model")
        self.assertNotEqual(factory.calls[0][2], main_thread)
        self.assertEqual(factory.calls[0][1]["device"], "cuda:0")

        chunks = await self._collect(backend, "hello", "turn-1")

        self.assertEqual(chunks, [self._pcm16([0.25, -0.25])])
        self.assertEqual(tts_model.generate_calls[0]["cache_revision"], 0)
        self.assertEqual(backend._contexts["turn-1"].cache["revision"], 1)
        await backend.stop()
        self.assertEqual(model.close_calls, 1)

    async def test_source_path_is_inserted_only_during_started_lifecycle(self):
        tts_model = _FakeTTSModel()
        source_path = str(Path(self._temporary_directory.name).resolve())
        backend, _, _ = self._make_backend(tts_model, source_path=source_path)
        self.assertNotIn(source_path, sys.path)

        await backend.start()

        self.assertEqual(sys.path[0], source_path)
        await backend.stop()
        self.assertNotIn(source_path, sys.path)

    async def test_server_environment_factory_matches_official_constructor(self):
        source_path = str(Path(self._temporary_directory.name).resolve())
        with patch.dict(
            os.environ,
            {
                "VOXCPM2_BACKEND": "official_prompt_cache",
                "VOXCPM2_MODEL_PATH": "official-model",
                "VOXCPM2_PROMPT_WAV": str(self.prompt_wav),
                "VOXCPM2_OFFICIAL_SOURCE": source_path,
                "VOXCPM2_OFFICIAL_DEVICE": "cuda:7",
                "VOXCPM2_OFFICIAL_OPTIMIZE": "0",
                "VOXCPM2_INFERENCE_TIMESTEPS": "8",
                "VOXCPM2_CFG_VALUE": "1.5",
                "VOXCPM2_SEED": "7",
            },
            clear=True,
        ):
            backend = _backend_from_env()

        self.assertIsInstance(backend, OfficialPromptCacheBackend)
        self.assertEqual(backend._reference_wav, self.prompt_wav)
        self.assertEqual(backend._source_path, source_path)
        self.assertEqual(backend._device, "cuda:7")
        self.assertEqual(backend._inference_timesteps, 8)
        self.assertEqual(backend._cfg_value, 1.5)
        self.assertEqual(backend._seed, 7)
        self.assertFalse(backend._optimize)

    async def test_sync_generator_runs_off_event_loop(self):
        tts_model = _BlockingTTSModel()
        backend, _, _ = self._make_backend(tts_model)
        await backend.start()
        main_thread = threading.get_ident()

        task = asyncio.create_task(self._collect(backend, "first", "turn-1"))
        entered = await asyncio.wait_for(
            asyncio.to_thread(tts_model.entered.wait, 0.5),
            timeout=1.0,
        )
        self.assertTrue(entered)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        tts_model.release.set()

        await asyncio.wait_for(task, timeout=1.0)
        self.assertNotEqual(tts_model.generate_calls[0]["thread_id"], main_thread)
        await backend.stop()

    async def test_same_context_is_serial_and_second_sentence_uses_merged_cache(self):
        tts_model = _BlockingTTSModel()
        backend, _, _ = self._make_backend(tts_model)
        await backend.start()

        first = asyncio.create_task(self._collect(backend, "first", "turn"))
        self.assertTrue(
            await asyncio.wait_for(
                asyncio.to_thread(tts_model.entered.wait, 0.5),
                timeout=1.0,
            )
        )
        second = asyncio.create_task(self._collect(backend, "second", "turn"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(tts_model.generate_calls), 1)
        tts_model.release.set()

        await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)

        self.assertEqual(
            [call["cache_revision"] for call in tts_model.generate_calls],
            [0, 1],
        )
        self.assertEqual(backend._contexts["turn"].cache["revision"], 2)
        await backend.stop()

    async def test_global_model_lock_serializes_different_contexts(self):
        tts_model = _BlockingTTSModel()
        backend, _, _ = self._make_backend(tts_model)
        await backend.start()

        first = asyncio.create_task(self._collect(backend, "first", "turn-a"))
        self.assertTrue(
            await asyncio.wait_for(
                asyncio.to_thread(tts_model.entered.wait, 0.5),
                timeout=1.0,
            )
        )
        second = asyncio.create_task(self._collect(backend, "second", "turn-b"))
        await asyncio.sleep(0.1)
        self.assertEqual(len(tts_model.generate_calls), 1)
        tts_model.release.set()

        await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)

        self.assertEqual(
            [call["cache_revision"] for call in tts_model.generate_calls],
            [0, 0],
        )
        await backend.stop()

    async def test_consumer_cancel_does_not_commit_partial_features(self):
        tts_model = _PauseAfterFirstYieldTTSModel()
        backend, _, _ = self._make_backend(tts_model)
        await backend.start()
        stream = backend.generate_pcm16("cancelled", "turn")

        first_chunk = await anext(stream)
        self.assertEqual(first_chunk, self._pcm16([0.1]))
        self.assertTrue(
            await asyncio.wait_for(
                asyncio.to_thread(tts_model.after_first.wait, 0.5),
                timeout=1.0,
            )
        )
        close_task = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        tts_model.release.set()
        await asyncio.wait_for(close_task, timeout=1.0)

        self.assertEqual(tts_model.merge_calls, [])
        self.assertEqual(backend._contexts["turn"].cache["revision"], 0)
        await self._collect(backend, "retry", "turn")
        self.assertEqual(tts_model.generate_calls[-1]["cache_revision"], 0)
        await backend.stop()

    async def test_cancel_while_waiting_for_worker_still_cleans_active_task(self):
        tts_model = _PauseAfterFirstYieldTTSModel()
        backend, _, _ = self._make_backend(tts_model)
        await backend.start()
        stream = backend.generate_pcm16("cancelled", "turn")
        await anext(stream)
        self.assertTrue(
            await asyncio.wait_for(
                asyncio.to_thread(tts_model.after_first.wait, 0.5),
                timeout=1.0,
            )
        )

        close_task = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        close_task.cancel()
        await asyncio.sleep(0)
        tts_model.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=1.0)

        self.assertEqual(backend._active_tasks, set())
        self.assertEqual(backend._contexts["turn"].active_stops, set())
        self.assertEqual(tts_model.merge_calls, [])
        await backend.stop()

    async def test_generation_error_does_not_commit(self):
        tts_model = _ErrorTTSModel()
        backend, _, _ = self._make_backend(tts_model)
        await backend.start()

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await self._collect(backend, "broken", "turn")

        self.assertEqual(tts_model.merge_calls, [])
        self.assertEqual(backend._contexts["turn"].cache["revision"], 0)
        await backend.stop()

    async def test_release_during_merge_cannot_commit_and_is_idempotent(self):
        tts_model = _MergeBlockingTTSModel()
        backend, _, _ = self._make_backend(tts_model)
        await backend.start()
        task = asyncio.create_task(self._collect(backend, "race", "turn"))
        self.assertTrue(
            await asyncio.wait_for(
                asyncio.to_thread(tts_model.merge_entered.wait, 0.5),
                timeout=1.0,
            )
        )

        await backend.release_context("turn", interrupted=True)
        await backend.release_context("turn", interrupted=True)
        tts_model.merge_release.set()
        await asyncio.wait_for(task, timeout=1.0)

        self.assertNotIn("turn", backend._contexts)
        calls_before_late_request = len(tts_model.generate_calls)
        self.assertEqual(await self._collect(backend, "race", "turn"), [])
        self.assertEqual(len(tts_model.generate_calls), calls_before_late_request)
        await backend.stop()

    async def test_released_context_tombstones_are_bounded(self):
        tts_model = _FakeTTSModel()
        with patch.dict(
            os.environ,
            {"VOXCPM2_CONTEXT_TOMBSTONE_LIMIT": "3"},
            clear=False,
        ):
            backend, _, _ = self._make_backend(tts_model)
        await backend.start()

        for index in range(5):
            await backend.release_context(f"turn-{index}")

        self.assertEqual(
            list(backend._released_contexts),
            ["turn-2", "turn-3", "turn-4"],
        )
        await backend.stop()

    async def test_leading_trim_matches_nano_cross_chunk_preroll(self):
        waveform = np.concatenate(
            (
                np.zeros(200, dtype=np.float32),
                np.full(60, 0.25, dtype=np.float32),
            )
        )
        chunks = [waveform[:73], waveform[73:151], waveform[151:211], waveform[211:]]
        tts_model = _FakeTTSModel({"trim": chunks})
        with patch.dict(
            os.environ,
            {
                "VOXCPM2_LEADING_TRIM_ENABLED": "1",
                "VOXCPM2_LEADING_TRIM_DBFS": "-60.0",
                "VOXCPM2_LEADING_TRIM_CONFIRM_MS": "20",
                "VOXCPM2_LEADING_TRIM_PREROLL_MS": "40",
                "VOXCPM2_LEADING_TRIM_MAX_SCAN_MS": "320",
            },
            clear=False,
        ):
            backend, _, _ = self._make_backend(tts_model)
        await backend.start()

        rendered = b"".join(await self._collect(backend, "trim", "turn"))

        self.assertEqual(rendered, self._pcm16(waveform[160:]))
        self.assertEqual(rendered[: 40 * 2], self._pcm16(np.zeros(40)))
        await backend.stop()

    async def test_stop_clears_contexts_and_is_idempotent(self):
        tts_model = _FakeTTSModel()
        backend, model, _ = self._make_backend(tts_model)
        await backend.start()
        await self._collect(backend, "hello", "turn")

        await backend.stop()
        await backend.stop()

        self.assertEqual(model.close_calls, 1)
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._base_cache)
        self.assertEqual(backend._contexts, {})
        self.assertEqual(backend.sample_rate, 0)


if __name__ == "__main__":
    unittest.main()
