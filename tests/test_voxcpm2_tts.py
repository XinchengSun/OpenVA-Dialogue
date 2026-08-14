import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from voice_service.voxcpm2_server import (
    NanoVoxCPM2Backend,
    VoxCPM2WebSocketBridge,
    _run_server,
    _warm_backend,
)

try:
    from pipecat.clocks.system_clock import SystemClock
    from pipecat.frames.frames import (
        EndFrame,
        ErrorFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        StartFrame,
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        TextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.utils.asyncio.task_manager import TaskManager
    from pipecat_dystream.voxcpm2_tts import (
        LocalPCMTTSService,
        VoxCPM2LocalTTSService,
    )
except ModuleNotFoundError:
    LocalPCMTTSService = None
    VoxCPM2LocalTTSService = None


class _FakeBackend:
    sample_rate = 24_000

    def __init__(self, *, block_after_first=False, empty=False):
        self.block_after_first = block_after_first
        self.empty = empty
        self.cancelled = False
        self.texts = []
        self.contexts = []
        self.released_contexts = []

    async def generate_pcm16(self, text, context_id=""):
        self.texts.append(text)
        self.contexts.append(context_id)
        if self.empty:
            return
        try:
            yield b"\x01\x00\x02\x00"
            if self.block_after_first:
                await asyncio.Future()
            yield b"\x03\x00"
        finally:
            self.cancelled = self.block_after_first

    async def release_context(self, context_id):
        self.released_contexts.append(context_id)


class _FakeNanoPool:
    def __init__(self, waveforms=()):
        self.add_prompt_calls = []
        self.encode_calls = []
        self.generate_calls = []
        self.stop_calls = 0
        self.waveforms = list(waveforms)

    async def wait_for_ready(self):
        return None

    async def get_model_info(self):
        return {"output_sample_rate": 24_000}

    async def add_prompt(self, wav, wav_format, prompt_text):
        self.add_prompt_calls.append((wav, wav_format, prompt_text))
        return "prompt-1"

    async def encode_latents(self, wav, wav_format):
        self.encode_calls.append((wav, wav_format))
        return b"latents"

    async def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        for waveform in self.waveforms:
            yield waveform

    async def stop(self):
        self.stop_calls += 1


class _FakeLifecycleBackend:
    sample_rate = 24_000

    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self):
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1


class _FakeServeContext:
    def __init__(self):
        self.entered = asyncio.Event()

    async def __aenter__(self):
        self.entered.set()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = asyncio.Queue()
        for message in messages:
            self._messages.put_nowait(message)
        self.sent = []

    async def recv(self):
        return await self._messages.get()

    async def send(self, message):
        self.sent.append(message)


class _FakeClientWebSocket:
    def __init__(self, messages):
        self._messages = asyncio.Queue()
        for message in messages:
            self._messages.put_nowait(message)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._messages.get()
        if message is StopAsyncIteration:
            raise StopAsyncIteration
        return message

    async def recv(self):
        return await self.__anext__()

    async def send(self, message):
        self.sent.append(message)


class _FakeConnection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _ConnectorQueue:
    def __init__(self, *websockets):
        self.websockets = list(websockets)

    def __call__(self, uri):
        return _FakeConnection(self.websockets.pop(0))


class VoxCPM2BridgeTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _pcm16(samples):
        values = np.asarray(samples, dtype=np.float32)
        return (np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    async def _render_waveforms(self, waveforms, *, enabled, **overrides):
        env = {
            "VOXCPM2_LEADING_TRIM_ENABLED": "1" if enabled else "0",
            "VOXCPM2_LEADING_TRIM_DBFS": "-60.0",
            "VOXCPM2_LEADING_TRIM_CONFIRM_MS": "20",
            "VOXCPM2_LEADING_TRIM_PREROLL_MS": "40",
            "VOXCPM2_LEADING_TRIM_MAX_SCAN_MS": "320",
            **{name: str(value) for name, value in overrides.items()},
        }
        pool = _FakeNanoPool(waveforms)
        with patch.dict(os.environ, env, clear=True):
            backend = NanoVoxCPM2Backend(
                model_path="model",
                prompt_wav="ref.wav",
                prompt_text="transcript",
                devices=[0],
            )
        backend._pool = pool
        backend._prompt_id = "prompt-1"
        backend.sample_rate = 1_000
        return [chunk async for chunk in backend.generate_pcm16("test")]

    async def test_leading_trim_is_disabled_by_default_and_bit_exact(self):
        waveforms = [
            np.zeros(80, dtype=np.float32),
            np.full(60, 0.25, dtype=np.float32),
        ]
        pool = _FakeNanoPool(waveforms)
        with patch.dict(os.environ, {}, clear=True):
            backend = NanoVoxCPM2Backend(
                model_path="model",
                prompt_wav="ref.wav",
                prompt_text="transcript",
                devices=[0],
            )
        backend._pool = pool
        backend._prompt_id = "prompt-1"
        backend.sample_rate = 1_000

        chunks = [chunk async for chunk in backend.generate_pcm16("test")]

        self.assertFalse(backend._leading_trim_enabled)
        self.assertEqual(chunks, [self._pcm16(waveform) for waveform in waveforms])

    async def test_leading_trim_keeps_immediate_speech_bit_exact(self):
        waveforms = [
            np.full(10, 0.25, dtype=np.float32),
            np.full(30, 0.20, dtype=np.float32),
        ]

        chunks = await self._render_waveforms(waveforms, enabled=True)

        self.assertEqual(chunks, [self._pcm16(waveform) for waveform in waveforms])

    async def test_leading_trim_no_onset_falls_back_bit_exact(self):
        waveforms = [
            np.full(100, 0.0001, dtype=np.float32),
            np.full(100, -0.0001, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.full(100, 0.0001, dtype=np.float32),
        ]

        chunks = await self._render_waveforms(waveforms, enabled=True)

        self.assertEqual(chunks, [self._pcm16(waveform) for waveform in waveforms])

    async def test_leading_trim_short_no_onset_eos_is_bit_exact(self):
        waveforms = [
            np.zeros(80, dtype=np.float32),
            np.full(80, 0.0001, dtype=np.float32),
        ]

        chunks = await self._render_waveforms(waveforms, enabled=True)

        self.assertEqual(chunks, [self._pcm16(waveform) for waveform in waveforms])

    async def test_leading_trim_scans_across_chunks_and_keeps_40ms_preroll(self):
        waveform = np.concatenate(
            (
                np.zeros(200, dtype=np.float32),
                np.full(60, 0.25, dtype=np.float32),
            )
        )
        waveforms = [waveform[:73], waveform[73:151], waveform[151:211], waveform[211:]]

        chunks = await self._render_waveforms(waveforms, enabled=True)

        rendered = b"".join(chunks)
        self.assertEqual(rendered, self._pcm16(waveform[160:]))
        self.assertEqual(rendered[: 40 * 2], self._pcm16(np.zeros(40)))

    async def test_leading_trim_preroll_preserves_weak_attack(self):
        waveform = np.concatenate(
            (
                np.zeros(200, dtype=np.float32),
                np.full(20, 0.0005, dtype=np.float32),
                np.full(40, 0.25, dtype=np.float32),
            )
        )

        chunks = await self._render_waveforms(
            [waveform[:160], waveform[160:225], waveform[225:]],
            enabled=True,
        )

        rendered = b"".join(chunks)
        self.assertEqual(rendered, self._pcm16(waveform[180:]))
        self.assertEqual(rendered[20 * 2 : 40 * 2], self._pcm16(waveform[200:220]))

    async def test_startup_cancellation_stops_created_pool(self):
        pool = _FakeNanoPool()
        pool.wait_for_ready = AsyncMock(side_effect=asyncio.CancelledError())
        with patch(
            "voice_service.voxcpm2_server.Path.is_file",
            return_value=True,
        ):
            backend = NanoVoxCPM2Backend(
                model_path="model",
                prompt_wav="ref.wav",
                prompt_text="transcript",
                devices=[0],
                pool_factory=lambda **kwargs: pool,
            )
            with self.assertRaises(asyncio.CancelledError):
                await backend.start()

        self.assertEqual(pool.stop_calls, 1)
        self.assertIsNone(backend._pool)

    async def test_shutdown_signal_path_stops_backend_once(self):
        backend = _FakeLifecycleBackend()
        serve_context = _FakeServeContext()
        shutdown_requested = asyncio.Event()
        server_task = asyncio.create_task(
            _run_server(
                "127.0.0.1",
                8770,
                _backend=backend,
                _serve_factory=lambda *args, **kwargs: serve_context,
                _shutdown_requested=shutdown_requested,
            )
        )

        await asyncio.wait_for(serve_context.entered.wait(), timeout=1.0)
        # The installed SIGTERM/SIGINT callback performs exactly this event set.
        shutdown_requested.set()
        await asyncio.wait_for(server_task, timeout=1.0)

        self.assertEqual(backend.start_calls, 1)
        self.assertEqual(backend.stop_calls, 1)

    async def test_startup_failure_always_stops_created_pool(self):
        for failing_method in (
            "wait_for_ready",
            "get_model_info",
            "add_prompt",
        ):
            with self.subTest(failing_method=failing_method):
                pool = _FakeNanoPool()
                setattr(
                    pool,
                    failing_method,
                    AsyncMock(side_effect=RuntimeError("startup failed")),
                )
                with (
                    patch(
                        "voice_service.voxcpm2_server.Path.is_file",
                        return_value=True,
                    ),
                    patch(
                        "voice_service.voxcpm2_server.Path.read_bytes",
                        return_value=b"RIFFfake",
                    ),
                ):
                    backend = NanoVoxCPM2Backend(
                        model_path="model",
                        prompt_wav="ref.wav",
                        prompt_text="transcript",
                        devices=[0],
                        pool_factory=lambda **kwargs: pool,
                    )
                    with self.assertRaisesRegex(RuntimeError, "startup failed"):
                        await backend.start()

                self.assertEqual(pool.stop_calls, 1)
                self.assertIsNone(backend._pool)

    async def test_prompt_is_added_once_and_reused(self):
        pool = _FakeNanoPool()
        with (
            patch("voice_service.voxcpm2_server.Path.is_file", return_value=True),
            patch("voice_service.voxcpm2_server.Path.read_bytes", return_value=b"RIFFfake"),
        ):
            backend = NanoVoxCPM2Backend(
                model_path="model",
                prompt_wav="ref.wav",
                prompt_text="准确逐字稿",
                devices=[0],
                pool_factory=lambda **kwargs: pool,
            )
            await backend.start()
            async for _ in backend.generate_pcm16("第一句"):
                pass
            async for _ in backend.generate_pcm16("第二句"):
                pass
            await backend.stop()

        self.assertEqual(len(pool.add_prompt_calls), 1)
        self.assertEqual(pool.encode_calls, [])
        self.assertEqual(
            pool.generate_calls,
            [
                {"target_text": "第一句", "prompt_id": "prompt-1"},
                {"target_text": "第二句", "prompt_id": "prompt-1"},
            ],
        )
        self.assertEqual(pool.stop_calls, 1)

    async def test_missing_transcript_encodes_reference_once(self):
        pool = _FakeNanoPool()
        with (
            patch("voice_service.voxcpm2_server.Path.is_file", return_value=True),
            patch("voice_service.voxcpm2_server.Path.read_bytes", return_value=b"RIFFfake"),
        ):
            backend = NanoVoxCPM2Backend(
                model_path="model",
                prompt_wav="ref.wav",
                prompt_text="",
                devices=[0],
                pool_factory=lambda **kwargs: pool,
            )
            await backend.start()
            async for _ in backend.generate_pcm16("你好"):
                pass
            await backend.stop()

        self.assertEqual(pool.add_prompt_calls, [])
        self.assertEqual(len(pool.encode_calls), 1)
        self.assertEqual(
            pool.generate_calls,
            [{"target_text": "你好", "ref_audio_latents": b"latents"}],
        )

    async def test_streams_pcm_with_backend_sample_rate(self):
        backend = _FakeBackend()
        websocket = _FakeWebSocket(
            [json.dumps({"type": "synthesize", "request_id": "r1", "text": "你好"})]
        )

        await VoxCPM2WebSocketBridge(backend).handle_connection(websocket)

        start = json.loads(websocket.sent[0])
        done = json.loads(websocket.sent[-1])
        self.assertEqual(start["sample_rate"], 24_000)
        self.assertEqual(start["audio_format"], "pcm_s16le")
        self.assertEqual(websocket.sent[1:3], [b"\x01\x00\x02\x00", b"\x03\x00"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(backend.contexts, ["r1"])
        self.assertEqual(backend.texts, ["你好"])

    async def test_cancel_closes_backend_generator(self):
        backend = _FakeBackend(block_after_first=True)
        websocket = _FakeWebSocket(
            [
                json.dumps({"type": "synthesize", "request_id": "r2", "text": "长回复"}),
                json.dumps({"type": "cancel", "request_id": "r2"}),
            ]
        )

        await VoxCPM2WebSocketBridge(backend).handle_connection(websocket)

        done = json.loads(websocket.sent[-1])
        self.assertEqual(done["status"], "cancelled")
        self.assertTrue(backend.cancelled)
        self.assertEqual(backend.released_contexts, ["r2"])

    async def test_release_context_is_forwarded_to_backend(self):
        backend = _FakeBackend()
        websocket = _FakeWebSocket(
            [json.dumps({"type": "release_context", "context_id": "ctx-release"})]
        )

        await VoxCPM2WebSocketBridge(backend).handle_connection(websocket)

        self.assertEqual(backend.released_contexts, ["ctx-release"])
        self.assertEqual(
            json.loads(websocket.sent[0]),
            {"type": "released", "context_id": "ctx-release"},
        )

    async def test_zero_audio_completion_is_reported_as_error(self):
        backend = _FakeBackend(empty=True)
        websocket = _FakeWebSocket(
            [json.dumps({"type": "synthesize", "request_id": "r-empty", "text": "test"})]
        )

        await VoxCPM2WebSocketBridge(backend).handle_connection(websocket)

        events = [json.loads(item) for item in websocket.sent if isinstance(item, str)]
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("without PCM", events[-1]["error"])

    async def test_health_reports_true_sample_rate(self):
        backend = _FakeBackend()
        websocket = _FakeWebSocket([json.dumps({"type": "health"})])

        await VoxCPM2WebSocketBridge(backend).handle_connection(websocket)

        response = json.loads(websocket.sent[0])
        self.assertEqual(response, {"type": "health", "status": "ok", "sample_rate": 24_000})

    async def test_health_invokes_optional_network_backend_liveness_check(self):
        backend = _FakeBackend()
        backend.health_check = AsyncMock()
        websocket = _FakeWebSocket([json.dumps({"type": "health"})])

        await VoxCPM2WebSocketBridge(backend).handle_connection(websocket)

        backend.health_check.assert_awaited_once_with()
        self.assertEqual(json.loads(websocket.sent[0])["status"], "ok")

    async def test_warmup_consumes_the_complete_pcm_stream(self):
        backend = _FakeBackend()

        chunks, pcm_bytes = await _warm_backend(backend, "warmup")

        self.assertEqual((chunks, pcm_bytes), (2, 6))
        self.assertEqual(backend.texts, ["warmup"])
        self.assertEqual(backend.released_contexts, ["__warmup__"])


@unittest.skipIf(VoxCPM2LocalTTSService is None, "Pipecat is not installed")
class VoxCPM2PipecatAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_stop_frame_timeout_allows_a_queued_sentence_to_start(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_TTS_STOP_FRAME_TIMEOUT_SEC": "10.0"},
            clear=True,
        ):
            service = LocalPCMTTSService(connector=lambda uri: None)

        self.assertEqual(service._stop_frame_timeout_s, 10.0)

    def test_generic_fish_config_takes_precedence_and_reports_provider(self):
        with patch.dict(
            os.environ,
            {
                "PIPECAT_TTS_BRIDGE_URI": "ws://127.0.0.1:8771",
                "PIPECAT_TTS_MODEL": "fishaudio/s2-pro",
                "PIPECAT_TTS_VOICE": "zero-shot-cloned",
                "VOXCPM2_BRIDGE_URI": "ws://127.0.0.1:8770",
            },
            clear=True,
        ):
            service = LocalPCMTTSService(connector=lambda uri: None)

        snapshot = service.health_snapshot()
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["model"], "fishaudio/s2-pro")
        self.assertEqual(snapshot["voice"], "zero-shot-cloned")
        self.assertEqual(snapshot["bridge_uri"], "ws://127.0.0.1:8771")
        self.assertEqual(snapshot["active_requests"], 0)
        self.assertIsNone(snapshot["warmup_elapsed_ms"])

    def test_legacy_voxcpm_config_keeps_fallback_identity(self):
        with patch.dict(
            os.environ,
            {"VOXCPM2_BRIDGE_URI": "ws://127.0.0.1:8770"},
            clear=True,
        ):
            service = LocalPCMTTSService(connector=lambda uri: None)

        snapshot = service.health_snapshot()
        self.assertEqual(snapshot["model"], "VoxCPM2")
        self.assertEqual(snapshot["voice"], "cloned")
        self.assertEqual(snapshot["bridge_uri"], "ws://127.0.0.1:8770")

    async def test_complete_upstream_sentence_starts_tts_without_lookahead(self):
        health = _FakeClientWebSocket(
            [json.dumps({"type": "health", "status": "ok", "sample_rate": 24_000})]
        )
        synthesis = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                ),
                b"\x01\x00",
                json.dumps({"type": "done", "status": "completed"}),
            ]
        )
        service = VoxCPM2LocalTTSService(
            connector=_ConnectorQueue(health, synthesis),
            task_manager=TaskManager(),
        )
        clock = SystemClock()
        clock.start()
        service._clock = clock
        service.push_frame = AsyncMock()
        await service.start(
            StartFrame(audio_in_sample_rate=16_000, audio_out_sample_rate=16_000)
        )

        await service.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        await service.process_frame(TextFrame("第一句。"), FrameDirection.DOWNSTREAM)
        for _ in range(50):
            if synthesis.sent:
                break
            await asyncio.sleep(0)

        self.assertTrue(synthesis.sent, "complete sentence waited for lookahead")
        self.assertEqual(json.loads(synthesis.sent[0])["text"], "第一句。")

        await service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await service.stop(EndFrame())

    async def test_valid_start_and_done_restore_ready_after_transient_failure(self):
        websocket = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                ),
                b"\x01\x00",
                json.dumps({"type": "done", "status": "completed"}),
            ]
        )
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(websocket)
        )
        service._ready = False

        frames = [frame async for frame in service.run_tts("recovery", "ctx-ready")]

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TTSAudioRawFrame)
        self.assertTrue(service.ready)

    async def test_completed_without_audio_is_rejected_and_marks_unready(self):
        websocket = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                ),
                json.dumps({"type": "done", "status": "completed"}),
            ]
        )
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(websocket)
        )
        service._ready = True

        frames = [frame async for frame in service.run_tts("empty", "ctx-empty")]

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertFalse(service.ready)

    async def test_done_without_start_is_rejected_and_marks_unready(self):
        websocket = _FakeClientWebSocket(
            [json.dumps({"type": "done", "status": "completed"})]
        )
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(websocket)
        )
        service._ready = True

        frames = [frame async for frame in service.run_tts("bad", "ctx-bad")]

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertFalse(service.ready)

    async def test_socket_eof_before_done_is_rejected(self):
        websocket = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                ),
                b"\x01\x00",
                StopAsyncIteration,
            ]
        )
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(websocket)
        )
        service._ready = True

        frames = [frame async for frame in service.run_tts("cut", "ctx-cut")]

        self.assertEqual(len(frames), 2)
        self.assertIsInstance(frames[0], TTSAudioRawFrame)
        self.assertIsInstance(frames[1], ErrorFrame)
        self.assertFalse(service.ready)

    async def test_unrequested_cancelled_status_is_rejected(self):
        websocket = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                ),
                json.dumps({"type": "done", "status": "cancelled"}),
            ]
        )
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(websocket)
        )
        service._ready = True

        frames = [frame async for frame in service.run_tts("cancel", "ctx-cancel")]

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertFalse(service.ready)

    async def test_wait_ready_sets_model_reported_sample_rate(self):
        websocket = _FakeClientWebSocket(
            [json.dumps({"type": "health", "status": "ok", "sample_rate": 24_000})]
        )
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(websocket)
        )

        sample_rate = await service.wait_ready(timeout=0.5)

        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(service.sample_rate, 24_000)
        self.assertEqual(json.loads(websocket.sent[0]), {"type": "health"})

    async def test_warmup_consumes_pcm_and_releases_private_context(self):
        synthesis = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                ),
                b"\x01\x00\x02\x00",
                b"\x03\x00",
                json.dumps({"type": "done", "status": "completed"}),
            ]
        )
        release = _FakeClientWebSocket(
            [json.dumps({"type": "released", "context_id": "__warmup__-fixed"})]
        )
        service = VoxCPM2LocalTTSService(
            connector=_ConnectorQueue(synthesis, release)
        )

        with patch(
            "pipecat_dystream.voxcpm2_tts.uuid.uuid4",
            return_value=SimpleNamespace(hex="fixed"),
        ):
            chunks, pcm_bytes = await service.warmup("你好。", timeout=0.5)

        self.assertEqual((chunks, pcm_bytes), (2, 6))
        request = json.loads(synthesis.sent[0])
        self.assertEqual(request["text"], "你好。")
        self.assertEqual(request["context_id"], "__warmup__-fixed")
        self.assertEqual(
            json.loads(release.sent[0]),
            {
                "type": "release_context",
                "context_id": "__warmup__-fixed",
                "reason": "warmup",
            },
        )
        self.assertGreaterEqual(service.health_snapshot()["warmup_elapsed_ms"], 0)

        repeated = await service.warmup("不会再次合成。", timeout=0.5)
        self.assertEqual(repeated, (2, 6))
        self.assertEqual(len(synthesis.sent), 1)

    async def test_adapter_uses_wire_sample_rate_and_preserves_pcm(self):
        websocket = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                ),
                b"\x01\x00\x02\x00",
                json.dumps(
                    {
                        "type": "done",
                        "status": "completed",
                    }
                ),
            ]
        )
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(websocket)
        )
        service.start_ttfb_metrics = AsyncMock()

        frames = [frame async for frame in service.run_tts("你好", "ctx-1")]

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TTSAudioRawFrame)
        self.assertEqual(frames[0].audio, b"\x01\x00\x02\x00")
        self.assertEqual(frames[0].sample_rate, 24_000)
        request = json.loads(websocket.sent[0])
        self.assertEqual(request["type"], "synthesize")
        self.assertEqual(request["context_id"], "ctx-1")
        self.assertEqual(request["text"], "你好")

    async def test_interruption_sends_cancel_and_drops_late_audio(self):
        websocket = _FakeClientWebSocket(
            [
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 24_000,
                        "channels": 1,
                        "sample_width": 2,
                    }
                )
            ]
        )
        release = _FakeClientWebSocket(
            [json.dumps({"type": "released", "context_id": "ctx-2"})]
        )
        service = VoxCPM2LocalTTSService(
            connector=_ConnectorQueue(websocket, release)
        )
        service.start_ttfb_metrics = AsyncMock()
        frames = []

        async def collect():
            async for frame in service.run_tts("长回复", "ctx-2"):
                frames.append(frame)

        task = asyncio.create_task(collect())
        for _ in range(20):
            if service._active.get("ctx-2"):
                break
            await asyncio.sleep(0)
        request_id = next(iter(service._active["ctx-2"]))
        await service.on_audio_context_interrupted("ctx-2")
        for _ in range(20):
            if websocket.sent and release.sent:
                break
            await asyncio.sleep(0)
        await websocket._messages.put(b"\x09\x00")
        await websocket._messages.put(
            json.dumps(
                {"type": "done", "request_id": request_id, "status": "cancelled"}
            )
        )
        await task

        cancel = json.loads(websocket.sent[-1])
        self.assertEqual(
            cancel,
            {
                "type": "cancel",
                "request_id": request_id,
                "context_id": "ctx-2",
                "discard_context": True,
            },
        )
        self.assertEqual(
            json.loads(release.sent[0]),
            {
                "type": "release_context",
                "context_id": "ctx-2",
                "reason": "interrupted",
            },
        )
        self.assertFalse(any(isinstance(frame, TTSAudioRawFrame) for frame in frames))

    async def test_context_release_does_not_block_interruption_hook(self):
        release = _FakeClientWebSocket([])
        service = VoxCPM2LocalTTSService(
            connector=lambda uri: _FakeConnection(release)
        )

        await asyncio.wait_for(
            service.on_audio_context_interrupted("ctx-fast"),
            timeout=0.05,
        )

        self.assertEqual(len(service._release_tasks), 1)
        task = next(iter(service._release_tasks))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_two_sentences_share_one_pipecat_audio_context(self):
        health = _FakeClientWebSocket(
            [json.dumps({"type": "health", "status": "ok", "sample_rate": 24_000})]
        )

        def synthesis_socket(pcm):
            return _FakeClientWebSocket(
                [
                    json.dumps(
                        {
                            "type": "start",
                            "sample_rate": 24_000,
                            "channels": 1,
                            "sample_width": 2,
                        }
                    ),
                    pcm,
                    json.dumps({"type": "done", "status": "completed"}),
                ]
            )

        connector = _ConnectorQueue(
            health,
            synthesis_socket(b"\x01\x00"),
            synthesis_socket(b"\x02\x00"),
        )
        service = VoxCPM2LocalTTSService(
            connector=connector,
            task_manager=TaskManager(),
        )
        clock = SystemClock()
        clock.start()
        service._clock = clock
        service.push_frame = AsyncMock()
        await service.start(
            StartFrame(audio_in_sample_rate=16_000, audio_out_sample_rate=16_000)
        )

        await service.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        await service.process_frame(TextFrame("第一句。"), FrameDirection.DOWNSTREAM)
        await service.process_frame(TextFrame("第二句。"), FrameDirection.DOWNSTREAM)
        await service.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await service.stop(EndFrame())

        pushed = [call.args[0] for call in service.push_frame.await_args_list]
        starts = [frame for frame in pushed if isinstance(frame, TTSStartedFrame)]
        stops = [frame for frame in pushed if isinstance(frame, TTSStoppedFrame)]
        audio = [frame for frame in pushed if isinstance(frame, TTSAudioRawFrame)]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(stops), 1)
        self.assertEqual(b"".join(frame.audio for frame in audio), b"\x01\x00\x02\x00")
        self.assertEqual(starts[0].context_id, stops[0].context_id)


if __name__ == "__main__":
    unittest.main()
