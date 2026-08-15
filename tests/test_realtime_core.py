import asyncio
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np


WORK_DIR = Path(__file__).resolve().parents[1] / "work"
sys.path.insert(0, str(WORK_DIR))

import server_mse


class FakeManager:
    def __init__(self, args):
        self.audio_q = queue.Queue()
        self.frame_q = queue.Queue()

    def start(self):
        pass

    def queues(self):
        return self.audio_q, self.frame_q, 1

    def health(self):
        return {
            "generation": 1,
            "motion_alive": True,
            "render_alive": True,
            "motion_exitcode": None,
            "render_exitcode": None,
        }

    def stop(self):
        pass


def make_bare_engine():
    engine = object.__new__(server_mse.RealtimeMSEEngine)
    engine.args = SimpleNamespace(hop_ms=200)
    engine.media_clients = set()
    engine._assistant_lock = threading.Lock()
    engine._speaker_chunks = deque()
    engine._speaker_head_offset = 0
    engine._speaker_samples = 0
    engine._user_chunks = deque()
    engine._user_head_offset = 0
    engine._user_samples = 0
    engine._last_user_voice_ts = 0.0
    engine._user_last_rms = 0.0
    engine.listener_virtual_only = False
    engine._state = engine.ASSISTANT_ACTIVE
    engine._turn_id = 1
    engine._stream_generation = 0
    engine._last_tts_audio_ts = 0.0
    engine._tail_samples_remaining = 0
    engine._interrupt_bridge_audio = np.zeros(0, dtype=np.float32)
    engine._interrupt_bridge_generation = -1
    engine._interrupt_grace_active = False
    engine._tts_reset_requested = threading.Event()
    engine._assistant_turn_started_at = time.monotonic()
    engine._first_motion_submit_turn = -1
    engine._first_rendered_frame_turn = -1
    engine.assistant_media_drain_sec = 1.2
    engine._assistant_media_pending_until = 0.0
    engine._media_control_seq = 0
    engine.log = lambda message: None
    return engine


class RealtimeCoreTests(unittest.TestCase):
    def test_turn_capture_keeps_exact_first_and_last_dystream_frames(self):
        engine = server_mse.RealtimeMSEEngine.__new__(
            server_mse.RealtimeMSEEngine
        )
        with tempfile.TemporaryDirectory() as tmp:
            engine.turn_frame_capture_dir = Path(tmp)
            engine._turn_capture_lock = threading.Lock()
            engine._turn_capture_active_key = None
            engine._turn_capture_last_frame = None
            engine._turn_capture_write_q = queue.Queue(maxsize=8)
            engine.log = lambda message: None
            first = np.full((3, 4, 3), 11, dtype=np.uint8)
            last = np.full((3, 4, 3), 222, dtype=np.uint8)

            engine._capture_assistant_frame(2, 5, first)
            engine._capture_assistant_frame(2, 5, last)
            engine._finish_assistant_frame_capture("test")

            first_path, first_saved = engine._turn_capture_write_q.get_nowait()
            last_path, last_saved = engine._turn_capture_write_q.get_nowait()
            self.assertTrue(first_path.name.endswith("-first.png"))
            self.assertTrue(last_path.name.endswith("-last.png"))
            np.testing.assert_array_equal(first_saved, first)
            np.testing.assert_array_equal(last_saved, last)

    def test_assistant_turn_start_is_a_typed_media_control(self):
        engine = make_bare_engine()
        controls = []
        engine._broadcast_media_control = controls.append

        turn_id = engine.begin_assistant_turn()

        self.assertEqual(turn_id, 2)
        self.assertEqual(
            controls,
            [
                {
                    "type": "assistant_turn_started",
                    "generation": 0,
                    "turn_id": 2,
                }
            ],
        )
        self.assertEqual(
            engine.assistant_turn_control_snapshot(),
            {
                "type": "assistant_turn_started",
                "generation": 0,
                "turn_id": 2,
            },
        )
        engine._state = engine.WARMUP_IDLE
        self.assertIsNone(engine.assistant_turn_control_snapshot())

    def test_emergency_stream_reset_advances_generation_before_restart(self):
        engine = make_bare_engine()
        engine._stream_generation = 3
        engine.audio_q = queue.Queue()
        engine.frame_q = queue.Queue()
        engine.manager = SimpleNamespace(motion_q=queue.Queue())
        calls = []

        class FakeClient:
            accepted_generation = 0

            def advance_generation(self, generation):
                self.accepted_generation = generation
                calls.append(("advance", generation))
                return 0

            def request_stream_reset(self):
                calls.append(("reset", self.accepted_generation))
                return 0

        engine.media_clients = {FakeClient()}
        old_value = os.environ.get("ENGINE_TTS_STREAM_RESET")
        os.environ["ENGINE_TTS_STREAM_RESET"] = "1"
        try:
            engine._apply_live_reset(
                np.zeros(0, dtype=np.float32),
                [],
                [],
            )
        finally:
            if old_value is None:
                os.environ.pop("ENGINE_TTS_STREAM_RESET", None)
            else:
                os.environ["ENGINE_TTS_STREAM_RESET"] = old_value

        self.assertEqual(calls, [("advance", 3), ("reset", 3)])

    def test_pipecat_launcher_disables_custom_idle_face_controls(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "start_pipecat_mse.sh"
        ).read_text(encoding="utf-8")
        for setting in (
            "DYSTREAM_LISTENING_CONTROLLER=0",
            "DYSTREAM_LISTENING_FACE_CONTROL=0",
            "DYSTREAM_LISTENING_NOD=0",
            "DYSTREAM_LISTENING_BLINK=0",
        ):
            self.assertIn(setting, launcher)

    def test_low_latency_media_config_uses_short_gop_without_turn_seeks(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "scripts" / "start_pipecat_mse.sh").read_text(
            encoding="utf-8"
        )
        frontend = (root / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn('PIPE_GOP="${PIPE_GOP:-3}"', launcher)
        self.assertIn("const START_BUFFER_SEC = 0.60;", frontend)
        self.assertIn("const TARGET_LATENCY_SEC = 0.48;", frontend)
        self.assertIn("const HANDOFF_TARGET_LATENCY_SEC = 0.08;", frontend)
        self.assertIn("const LOW_BUFFER_ENTER_SEC = 0.32;", frontend)
        self.assertIn("const LOW_BUFFER_EXIT_SEC = 0.48;", frontend)
        self.assertIn("const LOW_BUFFER_PLAYBACK_RATE = 0.96;", frontend)
        self.assertIn("slot.lowBufferRecovery = true;", frontend)
        self.assertIn("slot.lowBufferRecovery = false;", frontend)
        self.assertIn("const HARD_CATCHUP_SEC = 2.50;", frontend)
        self.assertNotIn("jumpNearLiveTail('assistant-active')", frontend)
        self.assertNotIn("jumpNearLiveTail('tts-start')", frontend)

    def test_avatar_stage_does_not_upscale_native_512_video(self):
        frontend = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        # Global border-box sizing plus the 1 px border on both sides gives
        # the video content an exact desktop maximum of 512 CSS pixels.
        self.assertIn("width: min(514px, 100%);", frontend)
        self.assertIn("margin-inline: auto;", frontend)

    def test_live_worker_logs_are_throttled_but_slow_hops_stay_visible(self):
        worker = (
            Path(__file__).resolve().parents[1]
            / "dual_gpu_mic_realtime_ui_v8_playbuffer.py"
        ).read_text(encoding="utf-8")

        self.assertIn('os.getenv("DYSTREAM_LIVE_LOG_EVERY", "5")', worker)
        self.assertGreaterEqual(worker.count("% live_log_every == 0"), 3)
        self.assertIn("t1 - t0 >= 0.2", worker)
        self.assertIn('float(item["motion_total"]) >= 0.2', worker)

    def test_frontend_batches_exact_40ms_pcm_and_clears_lifecycle_residue(self):
        frontend = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("const MIC_SEND_SAMPLES = 640;", frontend)
        self.assertIn("let micPending16 = new Float32Array(0);", frontend)
        self.assertIn("function sendMic16k(x16)", frontend)
        self.assertIn("offset + MIC_SEND_SAMPLES", frontend)
        self.assertIn("micPending16 = merged.slice(offset);", frontend)
        self.assertNotIn("off += 320", frontend)
        self.assertGreaterEqual(frontend.count("resetMicPending16();"), 4)

    def test_mse_teardown_disposes_both_live_slots(self):
        frontend = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        teardown = frontend.split("function teardownMSE()", 1)[1].split(
            "function buildMSE", 1
        )[0]
        dispose = frontend.split("function disposeSlot(slot)", 1)[1].split(
            "function activateSlot", 1
        )[0]

        self.assertIn("mseGeneration += 1;", teardown)
        self.assertIn("Array.from(slotByVideo.values())", teardown)
        self.assertIn("slots.forEach(disposeSlot);", teardown)
        self.assertIn("setLiveVisible(false, 'mse reset');", teardown)
        self.assertIn("slot.pendingSegments = [];", dispose)
        self.assertIn("slot.sourceBuffer.abort();", dispose)
        self.assertIn("slot.video.pause();", dispose)
        self.assertIn("slot.video.playbackRate = 1.0;", dispose)
        self.assertIn("slot.video.removeAttribute('src');", dispose)
        self.assertIn("URL.revokeObjectURL(slot.objectUrl);", dispose)

    def test_mse_callbacks_are_scoped_to_the_current_generation(self):
        frontend = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        setup = frontend.split("function buildMSE(mime, replacing)", 1)[1].split(
            "function setupMSE", 1
        )[0]

        self.assertIn("generation: mseGeneration", setup)
        self.assertIn("const nextMediaSource = new MediaSource();", setup)
        self.assertGreaterEqual(setup.count("isCurrentSlot(slot)"), 5)
        self.assertIn("slot.mediaSource !== nextMediaSource", setup)
        self.assertIn("slot.sourceBuffer !== nextSourceBuffer", setup)

    def test_frontend_start_can_timeout_and_be_cancelled_while_connecting(self):
        frontend = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("let micLifecycleGeneration = 0;", frontend)
        self.assertIn(
            "async function waitForMicOpen(ws, generation, timeoutMs = 5000)",
            frontend,
        )
        self.assertIn("mic websocket open timeout", frontend)
        self.assertIn("micLifecycleGeneration += 1;", frontend)
        self.assertIn(
            "oldMicWs.readyState < WebSocket.CLOSING",
            frontend,
        )
        self.assertIn(
            "if (generation === micLifecycleGeneration) startInProgress = false;",
            frontend,
        )

    def test_frontend_media_callbacks_ignore_superseded_websockets(self):
        frontend = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        connect = frontend.split("function connectMedia()", 1)[1].split(
            "function connectMicWS", 1
        )[0]

        self.assertIn("const generation = ++mediaWsGeneration;", connect)
        self.assertGreaterEqual(
            connect.count(
                "mediaWs !== ws || generation !== mediaWsGeneration"
            ),
            4,
        )
        self.assertIn("mediaWs = null;", connect)

    def test_active_interrupt_keeps_continuous_encoder_and_media_clock(self):
        root = Path(__file__).resolve().parents[1]
        server = (root / "server_mse.py").read_text(encoding="utf-8")
        launcher = (root / "scripts" / "start_pipecat_mse.sh").read_text(
            encoding="utf-8"
        )
        frontend = (root / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn(
            'ENGINE_TTS_STREAM_RESET="${ENGINE_TTS_STREAM_RESET:-0}"',
            launcher,
        )
        self.assertIn(
            '"ENGINE_TTS_STREAM_RESET",\n            "0",',
            server,
        )
        self.assertIn("client.advance_generation(stream_generation)", server)
        self.assertIn("client.request_stream_reset()", server)
        self.assertIn("obj.type === 'stream_reset'", frontend)
        self.assertIn("handoffMSE();", frontend)
        self.assertIn("obj.type === 'assistant_media_boundary'", frontend)
        self.assertIn("obj.type === 'assistant_media_ended'", frontend)
        self.assertIn("obj.type === 'assistant_turn_started'", frontend)
        self.assertIn("maybeReleaseLiveAudioAtBoundary", frontend)
        self.assertIn("maybeReturnToListenerAtBoundary", frontend)
        self.assertIn("deferredAssistantBoundaries", frontend)
        self.assertIn(
            "boundary.turnId !== activeAssistantTurn.turnId",
            frontend,
        )
        self.assertIn("assistant_interrupted", frontend)
        self.assertIn("assistant_turn_control_snapshot", server)
        response_created_branch = frontend.split(
            "message.includes('[PIPECAT S2S] response created')",
            1,
        )[1].split("else if", 1)[0]
        self.assertNotIn(
            "assistantResponseArmed = true",
            response_created_branch,
        )
        server_log_handler = frontend.split(
            "function handleServerLog(message)", 1
        )[1].split("function handleMediaControl", 1)[0]
        self.assertNotIn(
            "resetAssistantAudioGate(",
            server_log_handler,
        )
        self.assertIn('id="videoA"', frontend)
        self.assertIn('id="videoB"', frontend)
        live_video_lines = [
            line for line in frontend.splitlines()
            if 'id="videoA"' in line or 'id="videoB"' in line
        ]
        self.assertTrue(live_video_lines)
        self.assertTrue(all("autoplay" not in line for line in live_video_lines))
        self.assertIn("function activateSlot(slot, reason)", frontend)
        self.assertIn("requestVideoFrameCallback", frontend)
        self.assertNotIn("oldActive.video.pause();", frontend)
        self.assertNotIn("activeSlot.video.pause();", frontend)
        self.assertNotIn(
            "releaseLiveAudio('current-generation rendered frame')",
            frontend,
        )
        activate_slot = frontend.split(
            "function activateSlot(slot, reason)", 1
        )[1].split("function armSlotActivation", 1)[0]
        self.assertNotIn("setLiveVisible(true", activate_slot)
        start_handler = frontend.split("startBtn.onclick", 1)[1].split(
            "stopBtn.onclick", 1
        )[0]
        self.assertNotIn("setLiveVisible(true", start_handler)
        self.assertIn("setLiveVisible(true, 'assistant media started')", frontend)
        reset_handler = frontend.split(
            "obj.type === 'stream_reset'", 1
        )[1].split("obj && obj.message", 1)[0]
        self.assertNotIn("setupMSE(", reset_handler)
        self.assertNotIn("setLiveVisible(false", reset_handler)
        self.assertIn(
            'ENGINE_LISTENER_AUDIO="${ENGINE_LISTENER_AUDIO:-1}"',
            launcher,
        )
        self.assertNotIn("request_stream_reset()", server.split(
            "def begin_assistant_turn(self):", 1
        )[1].split("def enqueue_speaker_audio", 1)[0])

    def test_encoder_reset_notifies_new_epoch_before_stopping_old_ffmpeg(self):
        server = (
            Path(__file__).resolve().parents[1] / "server_mse.py"
        ).read_text(encoding="utf-8")
        media_client = server.split("class MediaPipeClient:", 1)[1].split(
            "class RealtimeMSEEngine:", 1
        )[0]
        run_reset = media_client.split("def _run(self):", 1)[1].split(
            "try:\n                item = self.segment_q.get", 1
        )[0]

        self.assertLess(
            run_reset.index("self._notify_stream_reset()"),
            run_reset.index("self._stop_ffmpeg()"),
        )
        self.assertIn(
            'json.dumps({"type": "stream_reset", "epoch": epoch})',
            server,
        )

    def test_media_generation_fence_keeps_encoded_output_and_av_pairs_atomic(self):
        server = (
            Path(__file__).resolve().parents[1] / "server_mse.py"
        ).read_text(encoding="utf-8")
        fence = server.split("def advance_generation(self, generation", 1)[1].split(
            "def media_clock_snapshot", 1
        )[0]

        self.assertIn("with self._generation_lock:", fence)
        self.assertIn("self.segment_q", fence)
        self.assertIn("self.av_pair_q", fence)
        self.assertNotIn("self.out_q", fence)
        self.assertIn("def _av_dispatcher(", server)
        self.assertIn("self._wait_for_writer_jobs(", server)
        self.assertIn(
            "await asyncio.to_thread(engine.unregister_media_client, client)",
            server,
        )

    def test_idle_uses_history_compaction_instead_of_anchor_reset(self):
        root = Path(__file__).resolve().parents[1]
        server = (root / "server_mse.py").read_text(encoding="utf-8")
        worker = (
            root / "dual_gpu_mic_realtime_ui_v8_playbuffer.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"type": "compact"', server)
        self.assertIn("def compact_stream_history():", worker)
        self.assertIn("past_motion_preserved=1", worker)
        self.assertNotIn("[ENGINE IDLE] model state reset", server)

    def test_segment_frames_aligns_to_pipe_frame_stride(self):
        old_stride = os.environ.get("PIPE_FRAME_STRIDE")
        try:
            os.environ["PIPE_FRAME_STRIDE"] = "2"
            self.assertEqual(
                server_mse.RealtimeMSEEngine._segment_frames_aligned_to_pipe_stride(5),
                (4, 2),
            )
            self.assertEqual(
                server_mse.RealtimeMSEEngine._segment_frames_aligned_to_pipe_stride(4),
                (4, 2),
            )
        finally:
            if old_stride is None:
                os.environ.pop("PIPE_FRAME_STRIDE", None)
            else:
                os.environ["PIPE_FRAME_STRIDE"] = old_stride

    def test_engine_feed_thread_stays_alive_after_init(self):
        original_manager = server_mse.omni.DyStreamWorkerManager
        old_feed_idle = os.environ.get("ENGINE_FEED_IDLE")
        os.environ["ENGINE_FEED_IDLE"] = "0"
        server_mse.omni.DyStreamWorkerManager = FakeManager
        loop = asyncio.new_event_loop()
        args = SimpleNamespace(
            sample=1,
            hop_ms=200,
            denoising_steps=1,
            motion_gpu=0,
            render_gpu=1,
            feature_lag_frames=3,
            port=6008,
            segment_frames=5,
        )
        try:
            engine = server_mse.RealtimeMSEEngine(args, loop)
            time.sleep(0.05)
            self.assertTrue(engine.feed_thread.is_alive())
            starting_health = engine.health_snapshot()
            self.assertFalse(starting_health["frame_ready"])
            self.assertEqual(starting_health["status"], "degraded")
            self.assertEqual(engine.audio_inflight_high_water_samples, 16_000)

            engine.latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
            ready_health = engine.health_snapshot()
            self.assertTrue(ready_health["frame_ready"])
            self.assertEqual(ready_health["status"], "ok")
            engine.shutdown()
        finally:
            loop.close()
            server_mse.omni.DyStreamWorkerManager = original_manager
            if old_feed_idle is None:
                os.environ.pop("ENGINE_FEED_IDLE", None)
            else:
                os.environ["ENGINE_FEED_IDLE"] = old_feed_idle

    def test_tts_samples_keep_fifo_order_when_head_is_split(self):
        engine = make_bare_engine()
        engine.enqueue_speaker_audio(np.full(5000, 1.0, dtype=np.float32))
        engine.enqueue_speaker_audio(np.full(1000, 2.0, dtype=np.float32))

        first = engine._pop_speaker_samples(3200, pad=False)
        second = engine._pop_speaker_samples(3200, pad=False)

        np.testing.assert_array_equal(first, np.full(3200, 1.0, dtype=np.float32))
        np.testing.assert_array_equal(second[:1800], np.full(1800, 1.0, dtype=np.float32))
        np.testing.assert_array_equal(second[1800:], np.full(1000, 2.0, dtype=np.float32))

    def test_partial_active_audio_is_not_padded_by_sample_buffer(self):
        engine = make_bare_engine()
        engine.enqueue_speaker_audio(np.full(1000, 0.25, dtype=np.float32))

        samples = engine._pop_speaker_samples(3200, pad=False)

        self.assertEqual(len(samples), 1000)
        self.assertTrue(np.all(samples == np.float32(0.25)))

    def test_partial_speaker_hop_switches_directly_to_listener_branch(self):
        for real_samples in (1, 1377, 3199):
            with self.subTest(real_samples=real_samples):
                engine = make_bare_engine()
                engine.user_speaking_rms = 0.1
                engine._listener_virtual_audio = np.full(
                    6400,
                    0.5,
                    dtype=np.float32,
                )
                engine._listener_virtual_cursor = 0
                engine._listener_other_source = "zero"
                engine._listener_transition_samples = 1280
                engine._user_chunks.append(
                    np.full(3200, 0.75, dtype=np.float32)
                )
                engine._user_samples = 3200

                other = engine._route_partial_speaker_hop_other(
                    3200,
                    real_samples,
                )
                speaker = np.concatenate(
                    [
                        np.ones(real_samples, dtype=np.float32),
                        np.zeros(3200 - real_samples, dtype=np.float32),
                    ]
                )

                np.testing.assert_array_equal(
                    other[:real_samples],
                    np.zeros(real_samples, dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    other[real_samples:],
                    np.full(3200 - real_samples, 0.5, dtype=np.float32),
                )
                self.assertFalse(
                    np.any(
                        (speaker[real_samples:] == 0)
                        & (other[real_samples:] == 0)
                    )
                )
                self.assertEqual(engine._listener_other_source, "virtual")
                self.assertEqual(
                    engine._listener_virtual_cursor,
                    3200 - real_samples,
                )
                self.assertEqual(engine._user_samples, 3200)

                next_virtual = engine._next_virtual_listener_hop(8)
                np.testing.assert_array_equal(
                    next_virtual,
                    np.full(8, 0.5, dtype=np.float32),
                )
                self.assertEqual(
                    engine._listener_virtual_cursor,
                    3208 - real_samples,
                )

    def test_normal_listener_source_change_keeps_existing_crossfade(self):
        engine = make_bare_engine()
        engine.user_speaking_rms = 0.1
        engine._listener_virtual_audio = np.full(
            6400,
            0.5,
            dtype=np.float32,
        )
        engine._listener_virtual_cursor = 0
        engine._listener_other_source = "zero"
        engine._listener_transition_samples = 1280

        other, mode, _ = engine._pop_listener_other_hop(3200)

        self.assertEqual(mode, "LISTENER_VIRTUAL")
        self.assertAlmostEqual(float(other[0]), 0.0, places=6)
        self.assertAlmostEqual(float(other[1279]), 0.5, places=6)
        self.assertTrue(np.all(other[1280:] == np.float32(0.5)))

    def test_listener_virtual_only_ignores_loud_mic_and_drains_it(self):
        engine = make_bare_engine()
        engine.listener_virtual_only = True
        engine.user_speaking_rms = 0.006
        engine._listener_virtual_audio = np.full(
            6400,
            0.5,
            dtype=np.float32,
        )
        engine._listener_virtual_cursor = 0
        engine._listener_other_source = "virtual"
        engine._listener_transition_samples = 1280
        engine._user_chunks.append(
            np.full(3200, 0.75, dtype=np.float32)
        )
        engine._user_samples = 3200

        other, mode, other_rms = engine._pop_listener_other_hop(3200)

        self.assertEqual(mode, "LISTENER_VIRTUAL")
        self.assertEqual(engine._listener_other_source, "virtual")
        self.assertEqual(engine._user_samples, 0)
        np.testing.assert_array_equal(
            other,
            np.full(3200, 0.5, dtype=np.float32),
        )
        self.assertAlmostEqual(other_rms, 0.5, places=6)

    def test_listener_default_still_uses_loud_mic(self):
        engine = make_bare_engine()
        engine.user_speaking_rms = 0.006
        engine._listener_virtual_audio = np.full(
            6400,
            0.5,
            dtype=np.float32,
        )
        engine._listener_virtual_cursor = 0
        engine._listener_other_source = "mic"
        engine._listener_transition_samples = 1280
        engine._user_chunks.append(
            np.full(3200, 0.75, dtype=np.float32)
        )
        engine._user_samples = 3200

        other, mode, other_rms = engine._pop_listener_other_hop(3200)

        self.assertEqual(mode, "USER_SPEAKING")
        self.assertEqual(engine._listener_other_source, "mic")
        self.assertEqual(engine._user_samples, 0)
        np.testing.assert_array_equal(
            other,
            np.full(3200, 0.75, dtype=np.float32),
        )
        self.assertAlmostEqual(other_rms, 0.75, places=6)

    def test_interrupt_preserves_generation_and_enqueues_graceful_tail(self):
        engine = make_bare_engine()
        engine.enqueue_speaker_audio(np.ones(1600, dtype=np.float32))

        dropped = engine.interrupt_assistant()

        self.assertEqual(dropped, 1)
        self.assertEqual(engine._state, engine.ASSISTANT_TAIL)
        self.assertEqual(engine._turn_id, 1)
        self.assertEqual(engine._stream_generation, 0)
        self.assertEqual(engine._speaker_samples, 6400)
        self.assertFalse(engine._tts_reset_requested.is_set())
        self.assertTrue(engine._interrupt_grace_active)
        graceful_tail = engine._pop_speaker_samples(6400, pad=False)
        self.assertGreater(graceful_tail[0], 0.99)
        self.assertEqual(graceful_tail[-1], 0.0)
        self.assertEqual(len(engine._interrupt_bridge_audio), 0)
        self.assertEqual(engine._interrupt_bridge_generation, -1)

    def test_interrupt_grace_uses_next_unconsumed_prefix_and_fades_media_safe(self):
        engine = make_bare_engine()
        engine._speaker_chunks.append(
            np.arange(5000, dtype=np.float32)
        )
        engine._speaker_head_offset = 1000
        engine._speaker_samples = 4000

        engine.interrupt_assistant()
        graceful_tail = engine._pop_speaker_samples(6400, pad=False)

        self.assertEqual(len(graceful_tail), 6400)
        self.assertEqual(graceful_tail[0], 1000.0)
        self.assertAlmostEqual(float(graceful_tail[-1]), 0.0, places=6)
        self.assertTrue(np.all(np.isfinite(graceful_tail)))
        self.assertEqual(len(engine._pop_speaker_samples(3200, pad=False)), 0)
        self.assertEqual(engine._stream_generation, 0)

    def test_zero_source_interrupt_still_produces_a_silent_grace_window(self):
        engine = make_bare_engine()

        engine.interrupt_assistant()
        graceful_tail = engine._pop_speaker_samples(6400, pad=False)

        np.testing.assert_array_equal(
            graceful_tail,
            np.zeros(6400, dtype=np.float32),
        )

    def test_normal_turn_preserves_continuous_stream_generation(self):
        engine = make_bare_engine()
        engine._state = engine.WARMUP_IDLE
        engine._turn_id = 3
        engine._stream_generation = 7
        engine._tts_reset_requested.clear()

        turn_id = engine.begin_assistant_turn()

        self.assertEqual(engine._state, engine.ASSISTANT_ACTIVE)
        self.assertEqual(engine._turn_id, 4)
        self.assertEqual(turn_id, 4)
        self.assertEqual(engine._stream_generation, 7)
        self.assertFalse(engine._tts_reset_requested.is_set())

    def test_assistant_output_pending_covers_active_audio_and_tail(self):
        engine = make_bare_engine()
        self.assertTrue(engine.assistant_output_pending())

        engine._state = engine.ASSISTANT_TAIL
        engine._speaker_samples = 0
        engine._tail_samples_remaining = 6400
        self.assertTrue(engine.assistant_output_pending())

        engine._state = engine.WARMUP_IDLE
        engine._tail_samples_remaining = 0
        self.assertFalse(engine.assistant_output_pending())

    def test_assistant_output_pending_covers_downstream_media_drain_window(self):
        engine = make_bare_engine()
        engine._state = engine.WARMUP_IDLE
        engine._speaker_samples = 0
        engine._tail_samples_remaining = 0

        engine._mark_assistant_media_pending()

        self.assertTrue(engine.assistant_output_pending())
        engine.interrupt_assistant()
        self.assertTrue(engine.assistant_output_pending())
        engine._pop_speaker_samples(6400, pad=False)
        engine._state = engine.WARMUP_IDLE
        engine._assistant_media_pending_until = 0.0
        self.assertFalse(engine.assistant_output_pending())

    def test_pipeline_milestones_are_logged_once_for_current_turn_only(self):
        engine = make_bare_engine()
        messages = []
        engine.log = messages.append
        engine._assistant_turn_started_at = time.monotonic() - 0.01

        engine._mark_pipeline_milestone(0, "motion_submit")
        engine._mark_pipeline_milestone(1, "motion_submit")
        engine._mark_pipeline_milestone(1, "motion_submit")
        engine._mark_pipeline_milestone(1, "rendered_frame")
        engine._mark_pipeline_milestone(1, "rendered_frame")

        self.assertEqual(len(messages), 2)
        self.assertIn("milestone=motion_submit", messages[0])
        self.assertIn("milestone=rendered_frame", messages[1])

    def test_renderer_source_motion_is_process_stable_across_generations(self):
        worker = (
            Path(__file__).resolve().parents[1]
            / "dual_gpu_mic_realtime_ui_v8_playbuffer.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(worker.count("render_src_motion = None"), 1)
        self.assertNotIn("render_src_generation", worker)
        self.assertNotIn("or produced_start == 0", worker)
        self.assertIn('"turn_id": current_turn_id', worker)
        self.assertIn('"turn_id": turn_id', worker)
        self.assertIn('"mode": mode', worker)


class MediaPipeResetTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, client_id=0):
        messages = []

        class FakeEngine:
            loop = asyncio.get_running_loop()
            args = SimpleNamespace(segment_frames=5)

            @staticmethod
            def log(message):
                messages.append(message)

        return server_mse.MediaPipeClient(FakeEngine(), client_id), messages

    async def test_old_encoder_epoch_bytes_are_filtered_after_reset(self):
        class FakeEngine:
            loop = asyncio.get_running_loop()
            args = SimpleNamespace(segment_frames=5)

            @staticmethod
            def log(message):
                pass

        client = server_mse.MediaPipeClient(FakeEngine(), 0)
        client._stream_epoch = 2

        self.assertIsNone(
            client.filter_output_item(("media", 1, b"old-epoch"))
        )
        self.assertEqual(
            client.filter_output_item(("media", 2, b"current-epoch")),
            b"current-epoch",
        )
        self.assertEqual(
            client.filter_output_item('{"type": "stream_reset", "epoch": 2}'),
            '{"type": "stream_reset", "epoch": 2}',
        )

    async def test_encoder_failure_requests_owner_thread_recovery_once(self):
        client, messages = self.make_client(9)
        io_stop = threading.Event()

        self.assertTrue(
            client._request_encoder_recovery("test failure", io_stop)
        )
        self.assertTrue(client._reset_requested.is_set())
        self.assertFalse(client._reset_done.is_set())
        self.assertFalse(
            client._request_encoder_recovery("duplicate failure", io_stop)
        )
        self.assertEqual(
            sum("[PIPE RECOVERY]" in message for message in messages),
            1,
        )

        client._reset_requested.clear()
        client._reset_in_progress.set()
        self.assertFalse(
            client._request_encoder_recovery("reentrant failure", io_stop)
        )
        client._reset_in_progress.clear()
        io_stop.set()
        self.assertFalse(
            client._request_encoder_recovery("normal stop", io_stop)
        )

    async def test_generation_fence_drops_only_unclaimed_raw_input(self):
        client, _ = self.make_client(10)
        frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        audio = np.zeros(640, dtype=np.float32)
        encoded_piece = ("media", 0, b"partial-moof-or-mdat")
        client.out_q.put_nowait(encoded_piece)

        self.assertTrue(client.push_segment(frames, audio, generation=0))
        self.assertTrue(client._enqueue_av_unit(0, b"old-video", b"old-audio"))

        dropped = client.advance_generation(1)

        self.assertEqual(dropped, 2)
        self.assertTrue(client.segment_q.empty())
        self.assertTrue(client.av_pair_q.empty())
        self.assertEqual(client.out_q.get_nowait(), encoded_piece)
        self.assertFalse(client.push_segment(frames, audio, generation=0))
        self.assertTrue(client.push_segment(frames, audio, generation=1))
        queued_generation, _, _, _ = client.segment_q.get_nowait()
        self.assertEqual(queued_generation, 1)
        clock = client.media_clock_snapshot()
        self.assertEqual(clock["stream_epoch"], 0)
        self.assertEqual(clock["accepted_generation"], 1)
        self.assertEqual(clock["epoch_media_units"], 0)

    async def test_assistant_boundary_uses_continuous_media_unit_clock(self):
        client, _ = self.make_client(12)
        client.advance_generation(3)

        client._emit_assistant_media_boundary(
            {
                "generation": 3,
                "turn_id": 9,
                "source_frame_offset": 1,
            },
            unit_start=10,
            stream_epoch=0,
        )
        message = json.loads(
            await asyncio.wait_for(client.out_q.get(), timeout=1.0)
        )

        self.assertEqual(message["type"], "assistant_media_boundary")
        self.assertEqual(message["stream_epoch"], 0)
        self.assertEqual(message["generation"], 3)
        self.assertEqual(message["turn_id"], 9)
        self.assertEqual(message["start_media_unit"], 10)
        self.assertAlmostEqual(message["safe_media_time_s"], 0.84)

    async def test_assistant_start_and_end_are_paired_on_source_frame_clock(self):
        client, _ = self.make_client(14)
        client.proc = SimpleNamespace(
            stdin=object(),
            poll=lambda: None,
            returncode=None,
        )
        client.audio_w_fd = 1
        enqueued = []

        def record(generation, frame_bytes, audio_bytes, boundaries):
            enqueued.append((generation, boundaries))
            return True

        client._enqueue_av_unit = record
        frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        audio = np.zeros(4 * 640, dtype=np.float32)
        meta = [
            {"generation": 0, "turn_id": 0, "mode": "WARMUP_IDLE"},
            {"generation": 0, "turn_id": 7, "mode": "ASSISTANT_ACTIVE"},
            {"generation": 0, "turn_id": 7, "mode": "ASSISTANT_TAIL"},
            {"generation": 0, "turn_id": 7, "mode": "WARMUP_IDLE"},
        ]

        client._write_to_ffmpeg(0, frames, audio, meta)

        self.assertEqual(len(enqueued), 2)
        self.assertEqual(
            enqueued[0][1],
            [{
                "type": "assistant_media_boundary",
                "generation": 0,
                "turn_id": 7,
                "source_frame_offset": 1,
            }],
        )
        self.assertEqual(
            enqueued[1][1],
            [{
                "type": "assistant_media_ended",
                "generation": 0,
                "turn_id": 7,
                "source_frame_offset": 1,
            }],
        )

    async def test_end_boundary_keeps_its_typed_event_name(self):
        client, _ = self.make_client(15)
        client._emit_assistant_media_boundary(
            {
                "type": "assistant_media_ended",
                "generation": 0,
                "turn_id": 4,
                "source_frame_offset": 0,
            },
            unit_start=5,
            stream_epoch=0,
        )
        message = json.loads(
            await asyncio.wait_for(client.out_q.get(), timeout=1.0)
        )

        self.assertEqual(message["type"], "assistant_media_ended")
        self.assertAlmostEqual(message["safe_media_time_s"], 0.4)

    async def test_old_claimed_pair_cannot_emit_boundary_on_new_epoch(self):
        client, messages = self.make_client(13)
        client.advance_generation(2)
        client._stream_epoch = 1

        client._emit_assistant_media_boundary(
            {
                "generation": 2,
                "turn_id": 11,
                "source_frame_offset": 0,
            },
            unit_start=0,
            stream_epoch=0,
        )

        self.assertTrue(client.out_q.empty())
        self.assertTrue(
            any("stale suppressed" in message for message in messages)
        )

    async def test_claimed_pair_finishes_both_writers_before_generation_fence(self):
        client, _ = self.make_client(11)
        io_stop = client._io_stop
        release_claimed = threading.Event()
        video_started = threading.Event()
        audio_started = threading.Event()
        video_seen = []
        audio_seen = []

        def fake_writer(job_q, seen, started):
            while not io_stop.is_set():
                try:
                    job = job_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if job is None:
                    break
                seen.append((job["unit_id"], job["data"]))
                if job["unit_id"] == 0:
                    started.set()
                    release_claimed.wait(timeout=2.0)
                job["done"].set()

        video_thread = threading.Thread(
            target=fake_writer,
            args=(client.video_job_q, video_seen, video_started),
            daemon=True,
        )
        audio_thread = threading.Thread(
            target=fake_writer,
            args=(client.audio_job_q, audio_seen, audio_started),
            daemon=True,
        )
        dispatcher_thread = threading.Thread(
            target=client._av_dispatcher,
            args=(
                client.av_pair_q,
                client.video_job_q,
                client.audio_job_q,
                io_stop,
            ),
            daemon=True,
        )

        self.assertTrue(client._enqueue_av_unit(0, b"video-0", b"audio-0"))
        video_thread.start()
        audio_thread.start()
        dispatcher_thread.start()
        try:
            self.assertTrue(video_started.wait(timeout=1.0))
            self.assertTrue(audio_started.wait(timeout=1.0))
            self.assertEqual(client._claimed_av_unit, (0, 0))

            # This second old pair is still unclaimed and must be removed.
            self.assertTrue(client._enqueue_av_unit(0, b"video-1", b"audio-1"))
            self.assertEqual(client.advance_generation(1), 1)
            self.assertTrue(client._enqueue_av_unit(1, b"video-2", b"audio-2"))
            release_claimed.set()

            deadline = time.monotonic() + 2.0
            boundary = None
            while time.monotonic() < deadline:
                boundary = client.generation_boundary_snapshot(
                    generation=1,
                    guard_units=2,
                )
                if boundary is not None:
                    break
                await asyncio.sleep(0.01)

            self.assertIsNotNone(boundary)
            self.assertEqual(video_seen, [(0, b"video-0"), (2, b"video-2")])
            self.assertEqual(audio_seen, [(0, b"audio-0"), (2, b"audio-2")])
            self.assertEqual(boundary["start_media_unit"], 1)
            self.assertEqual(boundary["safe_media_unit"], 3)
            self.assertAlmostEqual(boundary["safe_media_time_s"], 0.24)
            clock = client.media_clock_snapshot()
            self.assertEqual(clock["epoch_media_units"], 2)
            self.assertIsNone(client._claimed_av_unit)
        finally:
            release_claimed.set()
            io_stop.set()
            for q in (
                client.av_pair_q,
                client.video_job_q,
                client.audio_job_q,
            ):
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            client.running.clear()
            for thread in (dispatcher_thread, video_thread, audio_thread):
                thread.join(timeout=1.0)

    async def test_stream_reset_reaches_frontend_queue(self):
        messages = []

        class FakeEngine:
            loop = asyncio.get_running_loop()
            args = SimpleNamespace(segment_frames=5)

            @staticmethod
            def log(message):
                messages.append(message)

        client = server_mse.MediaPipeClient(FakeEngine(), 1)
        client.start()
        try:
            await asyncio.to_thread(client.request_stream_reset)
            message = await asyncio.wait_for(client.out_q.get(), timeout=1.0)
            self.assertEqual(message, '{"type": "stream_reset", "epoch": 1}')
            self.assertTrue(any("encoder reset epoch=1" in item for item in messages))
        finally:
            client.stop()
            client.thread.join(timeout=1.0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
    async def test_dual_pipe_ffmpeg_emits_fmp4_and_can_reset(self):
        messages = []

        class FakeEngine:
            loop = asyncio.get_running_loop()
            args = SimpleNamespace(segment_frames=5)

            @staticmethod
            def log(message):
                messages.append(message)

        client = server_mse.MediaPipeClient(FakeEngine(), 2)
        client.start()
        frames = np.zeros((20, 64, 64, 3), dtype=np.uint8)
        audio = np.zeros(20 * 640, dtype=np.float32)
        try:
            client.push_segment(frames, audio)
            received_bytes = False
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                raw_item = await asyncio.wait_for(client.out_q.get(), timeout=5.0)
                item = client.filter_output_item(raw_item)
                if isinstance(item, bytes) and item:
                    received_bytes = True
                    break
            self.assertTrue(received_bytes, messages)

            await asyncio.to_thread(client.request_stream_reset)
            reset_message = None
            deadline = asyncio.get_running_loop().time() + 1.0
            while asyncio.get_running_loop().time() < deadline:
                raw_item = await asyncio.wait_for(client.out_q.get(), timeout=1.0)
                item = client.filter_output_item(raw_item)
                if item is not None:
                    reset_message = item
                    break
            self.assertEqual(
                reset_message,
                '{"type": "stream_reset", "epoch": 1}',
            )
        finally:
            client.stop()
            client.thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
