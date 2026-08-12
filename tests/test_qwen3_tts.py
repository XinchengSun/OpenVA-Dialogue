import asyncio
import base64
import threading
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    CancelFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.clocks.system_clock import SystemClock
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_dystream.qwen3_tts import Qwen3RealtimeTTSService


class _FakeProvider:
    def __init__(self, callback, *, fail_connect=False):
        self.callback = callback
        self.fail_connect = fail_connect
        self.appended_text = []
        self.commit_calls = 0
        self.cancel_calls = 0
        self.close_calls = 0
        self.update_kwargs = None

    def connect(self):
        self.callback.on_open()
        if self.fail_connect:
            raise TimeoutError("simulated half-open websocket")

    def update_session(self, **kwargs):
        self.update_kwargs = kwargs
        self.callback.on_event({"type": "session.updated"})

    def append_text(self, text):
        self.appended_text.append(text)

    def commit(self):
        self.commit_calls += 1

    def cancel_response(self):
        self.cancel_calls += 1

    def close(self):
        self.close_calls += 1
        self.callback.on_close(None, None)

    def emit_created(self, response_id):
        self.callback.on_event(
            {
                "type": "response.created",
                "response": {"id": response_id},
            }
        )

    def emit_audio(self, response_id, audio):
        self.callback.on_event(
            {
                "type": "response.audio.delta",
                "response_id": response_id,
                "delta": base64.b64encode(audio).decode("ascii"),
            }
        )

    def emit_done(self, response_id, status="completed"):
        self.callback.on_event(
            {
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "status": status,
                },
            }
        )


class _FakeProviderFactory:
    def __init__(self, *, fail_first=False):
        self.providers = []
        self.fail_first = fail_first

    def __call__(self, callback):
        provider = _FakeProvider(
            callback,
            fail_connect=self.fail_first and not self.providers,
        )
        self.providers.append(provider)
        return provider


class _BlockingProvider(_FakeProvider):
    def __init__(self, callback):
        super().__init__(callback)
        self.connect_started = threading.Event()
        self.connect_release = threading.Event()

    def connect(self):
        self.connect_started.set()
        self.connect_release.wait(timeout=5.0)
        self.callback.on_open()

    def close(self):
        self.close_calls += 1
        self.connect_release.set()
        self.callback.on_close(None, None)


class _BlockingProviderFactory:
    def __init__(self):
        self.provider = None

    def __call__(self, callback):
        self.provider = _BlockingProvider(callback)
        return self.provider


async def _settle_callbacks():
    # SDK callbacks use call_soon_threadsafe and then create an asyncio task.
    for _ in range(6):
        await asyncio.sleep(0)


class Qwen3RealtimeTTSServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.factory = _FakeProviderFactory()
        self.service = Qwen3RealtimeTTSService(
            api_key="test-key",
            provider_factory=self.factory,
            task_manager=TaskManager(),
        )
        clock = SystemClock()
        clock.start()
        self.service._clock = clock
        self.service.push_frame = AsyncMock()
        await self.service.start(
            StartFrame(
                audio_in_sample_rate=16_000,
                audio_out_sample_rate=16_000,
            )
        )
        await _settle_callbacks()

    async def asyncTearDown(self):
        if not self.service._shutting_down:
            await self.service.cancel(CancelFrame())

    def _pushed_frames(self):
        return [call.args[0] for call in self.service.push_frame.await_args_list]

    async def _request(self, text="你好。", context_id="ctx-1"):
        yielded = [
            frame
            async for frame in self.service.run_tts(
                text,
                context_id,
            )
        ]
        await _settle_callbacks()
        return yielded

    async def test_start_opens_one_persistent_16k_commit_connection(self):
        self.assertTrue(self.service.ready)
        self.assertEqual(len(self.factory.providers), 1)
        provider = self.factory.providers[0]
        self.assertEqual(provider.update_kwargs["mode"], "commit")
        self.assertEqual(provider.update_kwargs["sample_rate"], 16_000)
        self.assertEqual(provider.update_kwargs["audio_format"], "pcm")

    async def test_streams_all_pcm_chunks_in_order_and_stops_after_done(self):
        provider = self.factory.providers[0]
        self.assertEqual(await self._request(), [None])
        await self.service.flush_audio("ctx-1")

        provider.emit_created("resp-1")
        provider.emit_audio("resp-1", b"\x01\x00\x02\x00")
        provider.emit_audio("resp-1", b"\x03\x00\x04\x00")
        await _settle_callbacks()

        frames_before_done = self._pushed_frames()
        self.assertTrue(any(isinstance(frame, TTSStartedFrame) for frame in frames_before_done))
        self.assertFalse(any(isinstance(frame, TTSStoppedFrame) for frame in frames_before_done))

        provider.emit_done("resp-1")
        await _settle_callbacks()

        frames = self._pushed_frames()
        audio_frames = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
        self.assertEqual(
            b"".join(frame.audio for frame in audio_frames),
            b"\x01\x00\x02\x00\x03\x00\x04\x00",
        )
        self.assertEqual([frame.context_id for frame in audio_frames], ["ctx-1", "ctx-1"])
        self.assertTrue(any(isinstance(frame, TTSStoppedFrame) for frame in frames))

    async def test_response_done_does_not_close_context_until_turn_flush(self):
        provider = self.factory.providers[0]
        await self._request()
        provider.emit_created("resp-1")
        provider.emit_done("resp-1")
        await _settle_callbacks()

        self.assertFalse(
            any(isinstance(frame, TTSStoppedFrame) for frame in self._pushed_frames())
        )

        await self.service.flush_audio("ctx-1")
        await _settle_callbacks()
        self.assertTrue(
            any(isinstance(frame, TTSStoppedFrame) for frame in self._pushed_frames())
        )

    async def test_interruption_drops_old_chunks_and_reconnects_in_background(self):
        old_provider = self.factory.providers[0]
        await self._request()
        old_provider.emit_created("resp-old")
        await _settle_callbacks()
        pushed_before_interrupt = len(self._pushed_frames())

        await self.service.on_audio_context_interrupted("ctx-1")
        old_provider.emit_audio("resp-old", b"\x01\x00\x02\x00")
        await self.service.wait_ready(timeout=1.0)
        await _settle_callbacks()

        late_frames = self._pushed_frames()[pushed_before_interrupt:]
        self.assertFalse(any(isinstance(frame, TTSAudioRawFrame) for frame in late_frames))
        self.assertEqual(old_provider.cancel_calls, 1)
        self.assertEqual(old_provider.close_calls, 1)
        self.assertGreaterEqual(len(self.factory.providers), 2)
        self.assertTrue(self.service.ready)

    async def test_two_sentences_share_one_context_until_both_responses_finish(self):
        provider = self.factory.providers[0]
        await self._request("第一句。")
        await self._request("第二句。")
        await self.service.flush_audio("ctx-1")

        provider.emit_created("resp-1")
        provider.emit_created("resp-2")
        provider.emit_done("resp-1")
        await _settle_callbacks()
        self.assertFalse(
            any(isinstance(frame, TTSStoppedFrame) for frame in self._pushed_frames())
        )

        provider.emit_done("resp-2")
        await _settle_callbacks()
        stopped = [
            frame for frame in self._pushed_frames() if isinstance(frame, TTSStoppedFrame)
        ]
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0].context_id, "ctx-1")


class Qwen3ConnectionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_half_open_attempt_is_closed_and_late_callbacks_are_stale(self):
        factory = _FakeProviderFactory(fail_first=True)
        service = Qwen3RealtimeTTSService(
            api_key="test-key",
            provider_factory=factory,
            task_manager=TaskManager(),
        )
        clock = SystemClock()
        clock.start()
        service._clock = clock
        service.push_frame = AsyncMock()

        with patch.dict(
            "os.environ",
            {"PIPECAT_TTS_CONNECT_ATTEMPTS": "2"},
            clear=False,
        ):
            await service.start(
                StartFrame(
                    audio_in_sample_rate=16_000,
                    audio_out_sample_rate=16_000,
                )
            )
        await _settle_callbacks()

        self.assertEqual(len(factory.providers), 2)
        failed_provider, active_provider = factory.providers
        self.assertEqual(failed_provider.close_calls, 1)
        self.assertIs(service._provider, active_provider)
        self.assertTrue(service.ready)

        # Reproduce the real SDK race: its failed thread reports ready/close
        # after the retry has already installed a new healthy provider.
        failed_provider.callback.on_event({"type": "session.updated"})
        failed_provider.callback.on_close(1006, "late old close")
        await _settle_callbacks()
        self.assertIs(service._provider, active_provider)
        self.assertTrue(service.ready)

        await service.cancel(CancelFrame())

    async def test_cancelled_connect_closes_handle_and_late_ready_is_ignored(self):
        factory = _BlockingProviderFactory()
        service = Qwen3RealtimeTTSService(
            api_key="test-key",
            provider_factory=factory,
            task_manager=TaskManager(),
        )
        clock = SystemClock()
        clock.start()
        service._clock = clock
        service.push_frame = AsyncMock()

        start_task = asyncio.create_task(
            service.start(
                StartFrame(
                    audio_in_sample_rate=16_000,
                    audio_out_sample_rate=16_000,
                )
            )
        )
        while factory.provider is None:
            await asyncio.sleep(0)
        await asyncio.to_thread(factory.provider.connect_started.wait, 1.0)

        start_task.cancel()
        await asyncio.gather(start_task, return_exceptions=True)
        await _settle_callbacks()

        self.assertEqual(factory.provider.close_calls, 1)
        self.assertIsNone(service._provider)
        self.assertFalse(service.ready)

        await service.cancel(CancelFrame())


if __name__ == "__main__":
    unittest.main()
