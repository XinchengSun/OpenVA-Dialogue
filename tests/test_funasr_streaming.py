import unittest
from unittest.mock import AsyncMock

import numpy as np
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_dystream.funasr_streaming import ParaformerStreamingSTTService


class _FakeProvider:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def generate(self, audio: np.ndarray, **kwargs):
        self.calls.append(
            {
                "audio": audio.copy(),
                "cache": kwargs["cache"],
                "is_final": kwargs["is_final"],
                "chunk_size": kwargs["chunk_size"],
            }
        )
        kwargs["cache"]["calls"] = kwargs["cache"].get("calls", 0) + 1
        return next(self.results)


class ParaformerStreamingSTTTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = _FakeProvider(["ni", "hao"])
        self.service = ParaformerStreamingSTTService(
            provider=self.provider,
            chunk_size=(0, 2, 1),
            pre_roll_secs=0.30,
            task_manager=TaskManager(),
        )
        self.service.push_frame = AsyncMock()
        clock = SystemClock()
        clock.start()
        self.service._clock = clock
        await self.service.start(
            StartFrame(audio_in_sample_rate=16_000, audio_out_sample_rate=16_000)
        )
        self.service.push_frame.reset_mock()

    async def asyncTearDown(self):
        await self.service.cancel(CancelFrame())

    @staticmethod
    def _audio(sample_count, value=1, sample_rate=16_000, channels=1):
        pcm = np.full(sample_count, value, dtype="<i2").tobytes()
        return InputAudioRawFrame(
            pcm,
            sample_rate=sample_rate,
            num_channels=channels,
        )

    def _pushed(self):
        return [call.args[0] for call in self.service.push_frame.await_args_list]

    def _pushed_with_directions(self):
        return [
            (
                call.args[0],
                call.args[1] if len(call.args) > 1 else FrameDirection.DOWNSTREAM,
            )
            for call in self.service.push_frame.await_args_list
        ]

    async def _start_speaking(self, direction=FrameDirection.UPSTREAM):
        await self.service.process_frame(VADUserStartedSpeakingFrame(), direction)

    async def _stop_speaking(self, direction=FrameDirection.UPSTREAM):
        await self.service.process_frame(VADUserStoppedSpeakingFrame(), direction)

    async def test_idle_audio_never_calls_provider_and_is_passed_downstream(self):
        for _ in range(50):
            await self.service.process_frame(
                self._audio(160), FrameDirection.DOWNSTREAM
            )

        self.assertEqual(self.provider.calls, [])
        self.assertEqual(len(self.service._pre_roll_buffer), 4_800 * 2)
        audio_frames = [
            (frame, direction)
            for frame, direction in self._pushed_with_directions()
            if isinstance(frame, InputAudioRawFrame)
        ]
        self.assertEqual(len(audio_frames), 50)
        self.assertTrue(
            all(direction is FrameDirection.DOWNSTREAM for _, direction in audio_frames)
        )

    async def test_vad_start_replays_pre_roll_once_then_stop_finalizes_and_idles(self):
        # chunk_size[1] == 2 means 1920 samples; leave 320 samples for final.
        frame = self._audio(2240, value=7)
        await self.service.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(self.provider.calls, [])

        await self._start_speaking()

        self.assertEqual(len(self.provider.calls), 1)
        self.assertFalse(self.provider.calls[0]["is_final"])
        self.assertEqual(self.provider.calls[0]["audio"].size, 1920)
        self.assertTrue(np.all(self.provider.calls[0]["audio"] > 0))
        interim = [f for f in self._pushed() if isinstance(f, InterimTranscriptionFrame)]
        self.assertEqual([frame.text for frame in interim], ["ni"])

        await self._stop_speaking()

        self.assertEqual(len(self.provider.calls), 2)
        self.assertIs(self.provider.calls[0]["cache"], self.provider.calls[1]["cache"])
        self.assertTrue(self.provider.calls[1]["is_final"])
        self.assertEqual(self.provider.calls[1]["audio"].size, 320)
        final = [f for f in self._pushed() if isinstance(f, TranscriptionFrame)]
        self.assertEqual([frame.text for frame in final], ["nihao"])
        self.assertTrue(final[0].finalized)
        self.assertFalse(self.service._vad_active)

        # More idle audio must not reuse the closed cache or call Paraformer.
        await self.service.process_frame(self._audio(1920), FrameDirection.DOWNSTREAM)
        self.assertEqual(len(self.provider.calls), 2)

        directions = self._pushed_with_directions()
        self.assertIn(
            (next(f for f in self._pushed() if isinstance(f, VADUserStartedSpeakingFrame)),
             FrameDirection.UPSTREAM),
            directions,
        )
        self.assertIn(
            (next(f for f in self._pushed() if isinstance(f, VADUserStoppedSpeakingFrame)),
             FrameDirection.UPSTREAM),
            directions,
        )

    async def test_barge_in_interruption_preserves_first_syllable(self):
        self.provider = _FakeProvider(["first"])
        self.service._provider = self.provider
        await self.service.process_frame(
            self._audio(100, value=123), FrameDirection.DOWNSTREAM
        )
        await self._start_speaking()
        first_cache = self.service._cache

        await self.service.process_frame(
            InterruptionFrame(), FrameDirection.DOWNSTREAM
        )

        self.assertIs(self.service._cache, first_cache)
        self.assertEqual(len(self.service._audio_buffer), 200)

        await self._stop_speaking()
        self.assertEqual(self.provider.calls[0]["audio"].size, 100)
        self.assertTrue(np.all(self.provider.calls[0]["audio"] > 0))
        self.assertTrue(self.provider.calls[0]["is_final"])
        final = [f for f in self._pushed() if isinstance(f, TranscriptionFrame)]
        self.assertEqual(final[-1].text, "first")

    async def test_exact_boundary_sends_empty_final_flush(self):
        self.provider = _FakeProvider(["chunk", "flush"])
        self.service._provider = self.provider
        await self.service.process_frame(
            self._audio(1920), FrameDirection.DOWNSTREAM
        )
        await self._start_speaking()
        await self._stop_speaking()

        self.assertEqual(self.provider.calls[1]["audio"].size, 0)
        self.assertTrue(self.provider.calls[1]["is_final"])
        final = [f for f in self._pushed() if isinstance(f, TranscriptionFrame)]
        self.assertEqual(final[-1].text, "chunkflush")

    async def test_only_upstream_vad_boundaries_control_the_gate(self):
        self.provider = _FakeProvider(["final"])
        self.service._provider = self.provider
        await self.service.process_frame(
            self._audio(100), FrameDirection.DOWNSTREAM
        )

        await self._start_speaking(FrameDirection.DOWNSTREAM)
        self.assertFalse(self.service._vad_active)
        await self._start_speaking(FrameDirection.UPSTREAM)
        self.assertTrue(self.service._vad_active)

        await self._stop_speaking(FrameDirection.DOWNSTREAM)
        self.assertTrue(self.service._vad_active)
        self.assertEqual(self.provider.calls, [])
        await self._stop_speaking(FrameDirection.UPSTREAM)
        self.assertFalse(self.service._vad_active)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertTrue(self.provider.calls[0]["is_final"])

    async def test_cancel_discards_active_audio_without_inference(self):
        await self.service.process_frame(
            self._audio(100), FrameDirection.DOWNSTREAM
        )
        await self._start_speaking()
        await self.service.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(self.provider.calls, [])
        self.assertFalse(self.service._vad_active)
        self.assertEqual(self.service._audio_buffer, bytearray())
        self.assertEqual(self.service._pre_roll_buffer, bytearray())

    async def test_end_finalizes_active_audio_and_clears_state(self):
        self.provider = _FakeProvider(["end"])
        self.service._provider = self.provider
        await self.service.process_frame(
            self._audio(100), FrameDirection.DOWNSTREAM
        )
        await self._start_speaking()
        await self.service.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(len(self.provider.calls), 1)
        self.assertTrue(self.provider.calls[0]["is_final"])
        self.assertFalse(self.service._vad_active)
        self.assertEqual(self.service._pre_roll_buffer, bytearray())

    async def test_rejects_non_16k_mono_without_calling_provider(self):
        await self.service.process_frame(
            self._audio(160, sample_rate=48_000, channels=2),
            FrameDirection.DOWNSTREAM,
        )

        self.assertEqual(self.provider.calls, [])
        self.assertTrue(any(isinstance(f, ErrorFrame) for f in self._pushed()))
        audio_directions = [
            direction
            for frame, direction in self._pushed_with_directions()
            if isinstance(frame, InputAudioRawFrame)
        ]
        self.assertEqual(audio_directions, [FrameDirection.DOWNSTREAM])

    async def test_rejects_disabling_passthrough_needed_by_downstream_vad(self):
        with self.assertRaisesRegex(ValueError, "downstream user aggregator owns VAD"):
            ParaformerStreamingSTTService(
                provider=self.provider,
                audio_passthrough=False,
                task_manager=TaskManager(),
            )


if __name__ == "__main__":
    unittest.main()
