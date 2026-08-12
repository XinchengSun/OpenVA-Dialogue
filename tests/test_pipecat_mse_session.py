import audioop
import asyncio
import unittest
from types import SimpleNamespace

import numpy as np

from pipecat_dystream.mse_session import (
    DyStreamEngineAudioAdapter,
    PipecatMSESession,
    _StatefulPCM16ToFloat16k,
)


class _FakeEngine:
    def __init__(self):
        self.begin_calls = 0
        self.audio_chunks = []
        self.end_calls = 0
        self.interrupt_calls = 0
        self.pending_output = False

    def begin_assistant_turn(self):
        self.begin_calls += 1
        self.pending_output = True
        return self.begin_calls

    def enqueue_speaker_audio(self, audio):
        self.audio_chunks.append(np.asarray(audio, dtype=np.float32).copy())

    def end_assistant_turn(self):
        self.end_calls += 1
        self.pending_output = True

    def interrupt_assistant(self):
        self.interrupt_calls += 1
        self.pending_output = False
        return self.interrupt_calls

    def assistant_output_pending(self):
        return self.pending_output


class StatefulPCM16ToFloat16kTests(unittest.TestCase):
    def test_chunked_resample_matches_one_continuous_resample(self):
        sample_rate = 24_000
        t = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
        samples = (0.25 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
        pcm = samples.tobytes()

        expected_bytes, _ = audioop.ratecv(pcm, 2, 1, sample_rate, 16_000, None)
        expected = np.frombuffer(expected_bytes, dtype="<i2").astype(np.float32) / 32768.0

        converter = _StatefulPCM16ToFloat16k()
        cuts = (0, 1378, 4096, 10002, len(pcm))
        actual_parts = [
            converter.decode(
                pcm[cuts[i] : cuts[i + 1]],
                sample_rate=sample_rate,
                num_channels=1,
            )
            for i in range(len(cuts) - 1)
        ]
        actual = np.concatenate(actual_parts)

        np.testing.assert_array_equal(actual, expected)


class DyStreamEngineAudioAdapterTests(unittest.TestCase):
    def test_maps_one_tts_turn_to_existing_engine_callbacks(self):
        engine = _FakeEngine()
        logs = []
        adapter = DyStreamEngineAudioAdapter(engine, logs.append)
        pcm = np.array([0, 1000, -1000, 2000], dtype="<i2").tobytes()

        adapter.start_tts("ctx-1")
        self.assertEqual(engine.begin_calls, 0)
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-1",
        )
        adapter.stop_tts("ctx-1")

        self.assertEqual(engine.begin_calls, 1)
        self.assertEqual(engine.end_calls, 1)
        self.assertEqual(engine.interrupt_calls, 0)
        self.assertEqual(len(engine.audio_chunks), 1)
        self.assertTrue(
            any("context=ctx-1 engine_turn=1" in message for message in logs)
        )
        np.testing.assert_allclose(
            engine.audio_chunks[0],
            np.array([0, 1000, -1000, 2000], dtype=np.float32) / 32768.0,
        )

    def test_external_interrupt_prevents_late_stop_from_ending_new_state(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)

        adapter.start_tts("ctx-old")
        adapter.mark_external_interrupt()
        adapter.stop_tts("ctx-old")

        self.assertEqual(engine.begin_calls, 0)
        self.assertEqual(engine.end_calls, 0)
        self.assertFalse(adapter.active)

    def test_empty_tts_context_never_stops_the_continuous_idle_engine(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)

        adapter.start_tts("ctx-empty")
        adapter.stop_tts("ctx-empty")

        self.assertEqual(engine.begin_calls, 0)
        self.assertEqual(engine.end_calls, 0)
        self.assertEqual(engine.interrupt_calls, 0)


class PipecatMSEHealthTests(unittest.TestCase):
    def test_pipeline_is_not_healthy_until_native_s2s_is_ready(self):
        session = object.__new__(PipecatMSESession)
        session.connected = asyncio.Event()
        session.connected.set()
        session.closed = asyncio.Event()
        session.task = None
        session._adapter = SimpleNamespace(active=False)
        session._s2s = SimpleNamespace(
            ready=False,
            health_snapshot=lambda: {"connected": False},
        )

        snapshot = session.health_snapshot()

        self.assertFalse(snapshot["ready"])
        self.assertFalse(snapshot["s2s_ready"])
        self.assertEqual(snapshot["mode"], "native_s2s")

    def test_pipeline_health_includes_ready_native_s2s_provider(self):
        session = object.__new__(PipecatMSESession)
        session.connected = asyncio.Event()
        session.connected.set()
        session.closed = asyncio.Event()
        session.task = None
        session._adapter = SimpleNamespace(active=False)
        session._s2s = SimpleNamespace(
            ready=True,
            health_snapshot=lambda: {
                "connected": True,
                "model": "qwen-audio-3.0-realtime-flash",
            },
        )

        snapshot = session.health_snapshot()

        self.assertTrue(snapshot["ready"])
        self.assertTrue(snapshot["s2s_ready"])
        self.assertTrue(snapshot["s2s"]["connected"])

    def test_custom_cascade_health_is_separate_from_native_s2s(self):
        session = object.__new__(PipecatMSESession)
        session._mode = "custom_cascade"
        session.connected = asyncio.Event()
        session.connected.set()
        session.closed = asyncio.Event()
        session.task = None
        session._adapter = SimpleNamespace(active=False)
        session._custom = SimpleNamespace(
            ready=True,
            health_snapshot=lambda: {
                "ready": True,
                "asr": {"model": "paraformer-zh-streaming"},
                "tts": {"model": "VoxCPM2"},
            },
        )

        snapshot = session.health_snapshot()

        self.assertTrue(snapshot["ready"])
        self.assertEqual(snapshot["mode"], "custom_cascade")
        self.assertTrue(snapshot["custom_cascade_ready"])
        self.assertFalse(snapshot["s2s_ready"])
        self.assertEqual(snapshot["s2s"], {})

    def test_external_interrupt_drops_late_audio_from_cancelled_context(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)
        pcm = np.array([1000, -1000], dtype="<i2").tobytes()

        adapter.start_tts("ctx-old")
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-old",
        )
        adapter.mark_external_interrupt()
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-old",
        )

        self.assertEqual(engine.begin_calls, 1)
        self.assertEqual(len(engine.audio_chunks), 1)
        self.assertFalse(adapter.active)

    def test_interrupt_barrier_rejects_old_start_then_allows_new_context(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)
        pcm = np.array([1000, -1000], dtype="<i2").tobytes()

        adapter.mark_external_interrupt()
        self.assertFalse(adapter.start_tts("ctx-old"))
        adapter.handle_pipeline_interrupt()
        self.assertFalse(adapter.start_tts("ctx-old"))
        self.assertTrue(adapter.start_tts("ctx-new"))
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-new",
        )

        self.assertEqual(engine.begin_calls, 1)
        self.assertEqual(len(engine.audio_chunks), 1)
        self.assertTrue(adapter.active)

    def test_audio_from_non_active_context_is_dropped(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)
        pcm = np.array([1000, -1000], dtype="<i2").tobytes()

        adapter.start_tts("ctx-current")
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-stale",
        )

        self.assertEqual(engine.begin_calls, 0)
        self.assertEqual(engine.audio_chunks, [])

    def test_contextless_tts_cannot_start_a_turn(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)

        self.assertFalse(adapter.start_tts(None))
        self.assertEqual(engine.begin_calls, 0)
        self.assertFalse(adapter.active)

    def test_pipeline_interrupt_resets_engine_once_when_tts_is_active(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)
        pcm = np.array([1000, -1000], dtype="<i2").tobytes()

        adapter.start_tts("ctx-1")
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-1",
        )
        adapter.handle_pipeline_interrupt()
        adapter.handle_pipeline_interrupt()

        self.assertEqual(engine.interrupt_calls, 1)
        self.assertFalse(adapter.active)

    def test_synchronous_host_interrupt_is_not_repeated_by_pipeline_frame(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)
        pcm = np.array([1000, -1000], dtype="<i2").tobytes()
        adapter.start_tts("ctx-1")
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-1",
        )

        adapter.mark_external_interrupt()
        engine.interrupt_assistant()
        adapter.mark_host_interrupt_applied()
        adapter.handle_pipeline_interrupt()

        self.assertEqual(engine.interrupt_calls, 1)
        self.assertFalse(adapter.active)

    def test_pipeline_interrupt_before_first_pcm_does_not_reset_idle_engine(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)

        adapter.start_tts("ctx-1")
        adapter.handle_pipeline_interrupt()

        self.assertEqual(engine.begin_calls, 0)
        self.assertEqual(engine.interrupt_calls, 0)
        self.assertFalse(adapter.active)

    def test_interrupt_after_response_done_clears_engine_tail_once(self):
        engine = _FakeEngine()
        adapter = DyStreamEngineAudioAdapter(engine, lambda _msg: None)
        pcm = np.array([1000, -1000], dtype="<i2").tobytes()

        adapter.start_tts("ctx-1")
        adapter.push_pcm(
            pcm,
            sample_rate=16_000,
            num_channels=1,
            context_id="ctx-1",
        )
        adapter.stop_tts("ctx-1")
        self.assertFalse(adapter.active)
        self.assertTrue(engine.assistant_output_pending())

        adapter.handle_pipeline_interrupt()
        adapter.handle_pipeline_interrupt()

        self.assertEqual(engine.interrupt_calls, 1)
        self.assertFalse(engine.assistant_output_pending())


if __name__ == "__main__":
    unittest.main()
