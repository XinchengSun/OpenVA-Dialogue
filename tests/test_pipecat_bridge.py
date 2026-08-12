import asyncio
import os
import unittest
from unittest.mock import patch

import numpy as np
from pipecat.frames.frames import OutputImageRawFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_dystream.audio import float32_to_pcm16le, pcm16le_to_float32_mono
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

from pipecat_dystream.bot import create_services, create_user_aggregator_params
from pipecat_dystream.bridge import DyStreamAvatarProcessor, DyStreamOutputClient
from pipecat_dystream.funasr_compat import RawPCMFunASRSTTService


class _FakeEngine:
    def __init__(self):
        self.output_clients = set()
        self.begin_calls = 0
        self.interrupt_calls = 0
        self.cancel_prepared_calls = 0

    def register_output_client(self, client):
        self.output_clients.add(client)
        client.start()

    def unregister_output_client(self, client):
        self.output_clients.discard(client)
        client.stop()

    def begin_assistant_turn(self):
        self.begin_calls += 1

    def interrupt_assistant(self):
        self.interrupt_calls += 1

    def cancel_prepared_assistant(self):
        self.cancel_prepared_calls += 1
        return True


class AudioConversionTests(unittest.TestCase):
    def test_stereo_48k_pcm_is_downmixed_and_resampled_to_16k(self):
        frames = 480
        left = np.full(frames, 8192, dtype="<i2")
        right = np.full(frames, -8192, dtype="<i2")
        stereo = np.column_stack([left, right]).reshape(-1).tobytes()

        output = pcm16le_to_float32_mono(
            stereo, sample_rate=48_000, num_channels=2
        )

        self.assertEqual(len(output), 160)
        self.assertLess(float(np.max(np.abs(output))), 1e-5)

    def test_float_conversion_clips_to_pcm16(self):
        encoded = float32_to_pcm16le(np.array([-2.0, 0.0, 2.0], dtype=np.float32))
        decoded = np.frombuffer(encoded, dtype="<i2")
        np.testing.assert_array_equal(decoded, np.array([-32767, 0, 32767], dtype=np.int16))


class FunASRCompatibilityTests(unittest.TestCase):
    def test_segmented_funasr_receives_raw_pcm_not_a_wav_container(self):
        service = object.__new__(RawPCMFunASRSTTService)
        self.assertFalse(service.wants_wav_segments)


class ServiceConstructionTests(unittest.TestCase):
    @patch("pipecat_dystream.qwen3_tts.Qwen3RealtimeTTSService")
    @patch("pipecat_dystream.resilient_llm.RecoveringOpenAILLMService")
    @patch("pipecat_dystream.funasr_compat.RawPCMFunASRSTTService")
    def test_create_services_uses_compat_funasr_settings(
        self, stt_service, _llm_service, _tts_service
    ):
        env = {
            "PIPECAT_LLM_API_KEY": "test-key",
            "PIPECAT_LLM_MODEL": "test-model",
            "PIPECAT_LLM_ENABLE_THINKING": "false",
            "PIPECAT_TTS_MODEL": "test-tts-model",
            "PIPECAT_TTS_VOICE": "test-voice",
        }
        with patch.dict(os.environ, env, clear=False):
            stt, _, _ = create_services()

        self.assertIs(stt, stt_service.return_value)
        stt_service.assert_called_once()
        self.assertIs(
            stt_service.call_args.kwargs["settings"], stt_service.Settings.return_value
        )

        _llm_service.Settings.assert_called_once()
        self.assertEqual(
            _llm_service.Settings.call_args.kwargs["extra"],
            {
                "extra_body": {"enable_thinking": False},
            },
        )
        llm_settings = _llm_service.Settings.call_args.kwargs
        self.assertEqual(llm_settings["max_tokens"], 256)
        self.assertNotIn("\u5341\u4e94", llm_settings["system_instruction"])
        self.assertIn("\u56de\u7b54\u8981\u5b8c\u6574", llm_settings["system_instruction"])
        self.assertIn("\u4e0d\u8981\u8d85\u8fc7\u56db\u53e5", llm_settings["system_instruction"])
        _tts_service.assert_called_once_with(
            api_key="test-key",
            model="test-tts-model",
            voice="test-voice",
            sample_rate=16_000,
            language_type="Chinese",
        )

    @patch("pipecat_dystream.bot.SileroVADAnalyzer")
    def test_user_turn_uses_short_speech_timeout_instead_of_smart_turn(self, vad):
        env = {
            "PIPECAT_USER_SPEECH_TIMEOUT_SEC": "0.25",
            "PIPECAT_USER_TURN_STOP_TIMEOUT_SEC": "1.5",
        }
        with patch.dict(os.environ, env, clear=False):
            params = create_user_aggregator_params()

        self.assertIs(params.vad_analyzer, vad.return_value)
        self.assertEqual(params.user_turn_stop_timeout, 1.5)
        self.assertEqual(len(params.user_turn_strategies.stop), 1)
        strategy = params.user_turn_strategies.stop[0]
        self.assertIsInstance(strategy, SpeechTimeoutUserTurnStopStrategy)
        self.assertEqual(strategy._user_speech_timeout, 0.25)

class AvatarProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_path_invalidates_late_output_and_clears_engine(self):
        engine = _FakeEngine()
        processor = DyStreamAvatarProcessor(engine, task_manager=TaskManager())
        processor._register_client()
        client = processor.output_client
        self.assertIsNotNone(client)

        processor._begin_turn("ctx-1")
        self.assertEqual(engine.begin_calls, 1)
        processor._interrupt_turn()
        self.assertEqual(engine.interrupt_calls, 1)

        frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        self.assertFalse(client.push_segment(frames, np.zeros(640, dtype=np.float32)))

        await processor._unregister_client()
        self.assertFalse(engine.output_clients)

    async def test_interrupt_without_tts_cancels_engine_preparation(self):
        engine = _FakeEngine()
        processor = DyStreamAvatarProcessor(engine, task_manager=TaskManager())

        processor._interrupt_turn()

        self.assertEqual(engine.cancel_prepared_calls, 1)
        self.assertEqual(engine.interrupt_calls, 0)


class OutputClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_image_then_matching_40ms_audio_for_each_frame(self):
        emitted = []

        async def emit(frame):
            emitted.append(frame)

        client = DyStreamOutputClient(
            asyncio.get_running_loop(), emit, max_segments=2
        )
        client.start()
        client.begin_turn("ctx-1")
        frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        frames[1, :, :, :] = 255
        audio = np.concatenate(
            [
                np.full(640, 0.25, dtype=np.float32),
                np.full(640, -0.25, dtype=np.float32),
            ]
        )
        try:
            client.finish_turn(TTSStoppedFrame(context_id="ctx-1"), expected_segments=1)
            self.assertTrue(client.push_segment(frames, audio))
            for _ in range(100):
                if len(emitted) == 5:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(len(emitted), 5)
            self.assertIsInstance(emitted[0], OutputImageRawFrame)
            self.assertTrue(emitted[0].sync_with_audio)
            self.assertIsInstance(emitted[1], TTSAudioRawFrame)
            self.assertEqual(emitted[1].context_id, "ctx-1")
            self.assertEqual(len(emitted[1].audio), 1280)
            self.assertIsInstance(emitted[2], OutputImageRawFrame)
            self.assertIsInstance(emitted[3], TTSAudioRawFrame)
            self.assertIsInstance(emitted[4], TTSStoppedFrame)
        finally:
            client.stop()
            await client.wait_stopped()

    async def test_interrupt_rejects_late_segments_until_next_turn(self):
        async def emit(_frame):
            return None

        client = DyStreamOutputClient(asyncio.get_running_loop(), emit)
        client.start()
        frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        audio = np.zeros(640, dtype=np.float32)
        try:
            client.begin_turn("ctx-1")
            client.interrupt()
            self.assertFalse(client.push_segment(frames, audio))
            client.begin_turn("ctx-2")
            self.assertTrue(client.push_segment(frames, audio))
        finally:
            client.stop()
            await client.wait_stopped()


if __name__ == "__main__":
    unittest.main()
