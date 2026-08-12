import asyncio
import base64
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_dystream.qwen_audio_s2s import (
    QwenAudioRealtimeS2SProcessor,
    create_qwen_audio_s2s_from_env,
)


_CLOSED = object()


class _FakeWebSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.block_audio_send: asyncio.Event | None = None
        self.audio_send_started = asyncio.Event()
        self.block_close: asyncio.Event | None = None
        self.close_started = asyncio.Event()

    async def send(self, message):
        if self.closed:
            raise ConnectionError("fake websocket closed")
        event = json.loads(message)
        if (
            event.get("type") == "input_audio_buffer.append"
            and self.block_audio_send is not None
        ):
            self.audio_send_started.set()
            await self.block_audio_send.wait()
        if self.closed:
            raise ConnectionError("fake websocket closed")
        self.sent.append(event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.incoming.get()
        if message is _CLOSED:
            raise StopAsyncIteration
        return json.dumps(message)

    async def emit(self, event):
        await self.incoming.put(event)

    async def disconnect(self):
        await self.incoming.put(_CLOSED)

    async def close(self):
        self.close_started.set()
        if self.block_close is not None:
            await self.block_close.wait()
        if not self.closed:
            self.closed = True
            await self.incoming.put(_CLOSED)


class _FakeConnectFactory:
    def __init__(self, *websockets):
        self.websockets = list(websockets)
        self.calls = []

    async def __call__(self, url, headers):
        self.calls.append((url, headers))
        if not self.websockets:
            raise ConnectionError("no fake websocket available")
        return self.websockets.pop(0)


async def _settle():
    for _ in range(8):
        await asyncio.sleep(0)


async def _wait_until(predicate, timeout=1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


class QwenAudioRealtimeS2STests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = _FakeWebSocket()
        self.factory = _FakeConnectFactory(self.websocket)
        self.logs = []
        self.processor = QwenAudioRealtimeS2SProcessor(
            api_key="test-key",
            base_url="wss://workspace.invalid/api-ws/v1/realtime",
            connect_factory=self.factory,
            log=self.logs.append,
            reconnect_initial_sec=0.001,
            reconnect_max_sec=0.002,
            shutdown_timeout_sec=0.02,
            task_manager=TaskManager(),
        )
        clock = SystemClock()
        clock.start()
        self.processor._clock = clock
        self.processor.push_frame = AsyncMock()
        await self.processor.process_frame(
            StartFrame(
                audio_in_sample_rate=16_000,
                audio_out_sample_rate=16_000,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await _settle()

    async def asyncTearDown(self):
        await self.processor.process_frame(
            CancelFrame(),
            FrameDirection.DOWNSTREAM,
        )

    def _pushed_frames(self):
        return [call.args[0] for call in self.processor.push_frame.await_args_list]

    async def _mark_ready(self):
        await self.websocket.emit({"type": "session.updated"})
        await self.processor.wait_ready(timeout=1.0)
        await _settle()

    async def test_waits_for_session_updated_and_sends_smart_turn_config(self):
        self.assertFalse(self.processor.ready)
        self.assertEqual(len(self.factory.calls), 1)
        url, headers = self.factory.calls[0]
        self.assertIn("model=qwen-audio-3.0-realtime-flash", url)
        self.assertEqual(headers["Authorization"], "Bearer test-key")

        update = self.websocket.sent[0]
        self.assertEqual(update["type"], "session.update")
        self.assertEqual(update["session"]["voice"], "longanqian")
        self.assertEqual(
            update["session"]["turn_detection"],
            {"type": "smart_turn"},
        )
        self.assertEqual(update["session"]["input_audio_format"], "pcm")
        self.assertEqual(update["session"]["output_audio_format"], "pcm")
        self.assertEqual(update["session"]["max_history_turns"], 10)

        await self._mark_ready()
        self.assertTrue(self.processor.ready)

    async def test_server_vad_sends_only_its_configurable_parameters(self):
        processor = QwenAudioRealtimeS2SProcessor(
            api_key="test-key",
            base_url="wss://workspace.invalid/api-ws/v1/realtime",
            turn_detection="server_vad",
            threshold=0.2,
            silence_duration_ms=400,
            task_manager=TaskManager(),
        )
        self.assertEqual(
            processor._session_update_event()["session"]["turn_detection"],
            {
                "type": "server_vad",
                "threshold": 0.2,
                "silence_duration_ms": 400,
            },
        )

    async def test_streams_16k_pcm_without_buffering_for_replay(self):
        await self._mark_ready()
        pcm = b"\x01\x00\x02\x00"
        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=pcm,
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await _settle()
        append = self.websocket.sent[-1]
        self.assertEqual(append["type"], "input_audio_buffer.append")
        self.assertEqual(base64.b64decode(append["audio"]), pcm)

    async def test_microphone_backlog_keeps_live_tail_without_reconnecting(self):
        await self._mark_ready()
        send_gate = asyncio.Event()
        self.websocket.block_audio_send = send_gate
        chunks = [
            bytes([index]) * 3200
            for index in range(5)
        ]

        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=chunks[0],
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await asyncio.wait_for(
            self.websocket.audio_send_started.wait(),
            timeout=1.0,
        )
        for chunk in chunks[1:]:
            await self.processor.process_frame(
                InputAudioRawFrame(
                    audio=chunk,
                    sample_rate=16_000,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )

        self.assertTrue(self.processor.ready)
        self.assertFalse(self.websocket.closed)
        self.assertEqual(len(self.factory.calls), 1)
        self.assertEqual(self.processor._microphone_queue.qsize(), 3)

        send_gate.set()
        await _wait_until(
            lambda: self.processor._microphone_queue.qsize() == 0
        )
        await _settle()
        appended = [
            base64.b64decode(event["audio"])
            for event in self.websocket.sent
            if event.get("type") == "input_audio_buffer.append"
        ]
        self.assertEqual(appended, [chunks[0], chunks[2], chunks[3], chunks[4]])

    async def test_microphone_send_timeout_recovers_stalled_websocket(self):
        replacement = _FakeWebSocket()
        self.factory.websockets.append(replacement)
        await self._mark_ready()
        old_epoch = self.processor._connection_epoch
        self.processor._microphone_send_timeout_sec = 0.01
        self.websocket.block_audio_send = asyncio.Event()

        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=bytes(3200),
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await asyncio.wait_for(
            self.websocket.audio_send_started.wait(),
            timeout=1.0,
        )
        await _wait_until(
            lambda: self.processor._connection_epoch > old_epoch
        )

        self.assertTrue(self.websocket.closed)
        self.assertFalse(self.processor.ready)
        self.assertTrue(
            any(
                "microphone send failed" in message
                and "TimeoutError" in message
                for message in self.logs
            )
        )

        await replacement.emit({"type": "session.updated"})
        await self.processor.wait_ready(timeout=1.0)
        self.assertTrue(self.processor.ready)

    async def test_response_done_is_the_only_normal_stop_boundary(self):
        await self._mark_ready()
        audio = b"\x01\x00\x02\x00"
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-1"}}
        )
        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "delta": base64.b64encode(audio).decode("ascii"),
            }
        )
        await self.websocket.emit(
            {"type": "response.audio.done", "response_id": "resp-1"}
        )
        await _settle()

        frames = self._pushed_frames()
        starts = [frame for frame in frames if isinstance(frame, TTSStartedFrame)]
        chunks = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
        stops = [frame for frame in frames if isinstance(frame, TTSStoppedFrame)]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].audio, audio)
        self.assertEqual(chunks[0].sample_rate, 24_000)
        self.assertEqual(chunks[0].context_id, starts[0].context_id)
        self.assertEqual(stops, [])

        await self.websocket.emit(
            {
                "type": "response.done",
                "response": {"id": "resp-1", "status": "completed"},
            }
        )
        await _settle()
        stops = [
            frame
            for frame in self._pushed_frames()
            if isinstance(frame, TTSStoppedFrame)
        ]
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0].context_id, starts[0].context_id)

    async def test_two_complete_turns_share_one_connection_without_boundary_leak(self):
        await self._mark_ready()
        for turn in (1, 2):
            response_id = f"resp-{turn}"
            audio = bytes((turn, 0))
            await self.websocket.emit(
                {"type": "input_audio_buffer.speech_started"}
            )
            await self.websocket.emit(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "reason": "turn_valid",
                }
            )
            await self.websocket.emit(
                {
                    "type": "response.created",
                    "response": {"id": response_id},
                }
            )
            await self.websocket.emit(
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(audio).decode("ascii"),
                }
            )
            await self.websocket.emit(
                {
                    "type": "response.audio.done",
                    "response_id": response_id,
                }
            )
            await self.websocket.emit(
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                    },
                }
            )
            await _settle()

        frames = self._pushed_frames()
        starts = [frame for frame in frames if isinstance(frame, TTSStartedFrame)]
        chunks = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
        stops = [frame for frame in frames if isinstance(frame, TTSStoppedFrame)]
        self.assertEqual(len(self.factory.calls), 1)
        self.assertEqual([frame.audio for frame in chunks], [b"\x01\x00", b"\x02\x00"])
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(stops), 2)
        self.assertNotEqual(starts[0].context_id, starts[1].context_id)
        self.assertEqual(
            [frame.context_id for frame in chunks],
            [frame.context_id for frame in starts],
        )
        self.assertEqual(
            [frame.context_id for frame in stops],
            [frame.context_id for frame in starts],
        )

    async def test_completed_response_id_cannot_restart_from_late_audio(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-done"}}
        )
        await self.websocket.emit(
            {
                "type": "response.done",
                "response": {"id": "resp-done", "status": "completed"},
            }
        )
        await _settle()
        before_late = len(self._pushed_frames())

        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-done",
                "delta": base64.b64encode(b"\x06\x00").decode("ascii"),
            }
        )
        await _settle()

        late = self._pushed_frames()[before_late:]
        self.assertFalse(any(isinstance(frame, TTSStartedFrame) for frame in late))
        self.assertFalse(any(isinstance(frame, TTSAudioRawFrame) for frame in late))

    async def test_audio_delta_without_created_synthesizes_start(self):
        await self._mark_ready()
        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-fallback",
                "delta": base64.b64encode(b"\x03\x00").decode("ascii"),
            }
        )
        await _settle()
        relevant = [
            frame
            for frame in self._pushed_frames()
            if isinstance(frame, (TTSStartedFrame, TTSAudioRawFrame))
        ]
        self.assertEqual(len(relevant), 2)
        self.assertIsInstance(relevant[0], TTSStartedFrame)
        self.assertIsInstance(relevant[1], TTSAudioRawFrame)
        self.assertEqual(relevant[0].context_id, relevant[1].context_id)

    async def test_external_interrupt_cancels_and_filters_late_audio(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-old"}}
        )
        await _settle()
        await self.processor.process_frame(
            InterruptionFrame(),
            FrameDirection.DOWNSTREAM,
        )
        await _settle()
        self.assertEqual(self.websocket.sent[-1]["type"], "response.cancel")
        before_late = len(self._pushed_frames())

        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-old",
                "delta": base64.b64encode(b"\x04\x00").decode("ascii"),
            }
        )
        await self.websocket.emit(
            {"type": "response.done", "response": {"id": "resp-old"}}
        )
        await _settle()
        late = self._pushed_frames()[before_late:]
        self.assertFalse(any(isinstance(frame, TTSAudioRawFrame) for frame in late))
        self.assertFalse(any(isinstance(frame, TTSStoppedFrame) for frame in late))

    async def test_interrupt_before_response_created_blocks_old_response(self):
        await self._mark_ready()
        await self.processor.process_frame(
            InterruptionFrame(),
            FrameDirection.DOWNSTREAM,
        )
        await _settle()

        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-old"}}
        )
        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-old",
                "delta": base64.b64encode(b"\x04\x00").decode("ascii"),
            }
        )
        await _settle()
        self.assertFalse(
            any(
                isinstance(frame, TTSAudioRawFrame)
                and frame.audio == b"\x04\x00"
                for frame in self._pushed_frames()
            )
        )

        await self.websocket.emit({"type": "input_audio_buffer.speech_started"})
        await self.websocket.emit(
            {"type": "input_audio_buffer.speech_stopped", "reason": "turn_valid"}
        )
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-new"}}
        )
        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-new",
                "delta": base64.b64encode(b"\x05\x00").decode("ascii"),
            }
        )
        await _settle()
        self.assertTrue(
            any(
                isinstance(frame, TTSAudioRawFrame)
                and frame.audio == b"\x05\x00"
                for frame in self._pushed_frames()
            )
        )

    async def test_blocked_microphone_send_does_not_delay_local_interruption(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-active"}}
        )
        await _settle()
        send_gate = asyncio.Event()
        self.websocket.block_audio_send = send_gate
        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=b"\x07\x00",
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await asyncio.wait_for(self.websocket.audio_send_started.wait(), timeout=1.0)

        await asyncio.wait_for(
            self.processor.process_frame(
                InterruptionFrame(),
                FrameDirection.DOWNSTREAM,
            ),
            timeout=0.1,
        )
        self.assertTrue(
            any(
                isinstance(frame, InterruptionFrame)
                for frame in self._pushed_frames()
            )
        )
        send_gate.set()
        await _settle()

    async def test_old_cancel_is_dropped_after_new_turn_response_starts(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-old"}}
        )
        await _settle()

        send_gate = asyncio.Event()
        self.websocket.block_audio_send = send_gate
        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=b"\x07\x00",
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await asyncio.wait_for(self.websocket.audio_send_started.wait(), timeout=1.0)
        await self.processor.process_frame(
            InterruptionFrame(),
            FrameDirection.DOWNSTREAM,
        )

        await self.websocket.emit({"type": "input_audio_buffer.speech_started"})
        await self.websocket.emit(
            {"type": "input_audio_buffer.speech_stopped", "reason": "turn_valid"}
        )
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-new"}}
        )
        await _wait_until(lambda: self.processor._response_id == "resp-new")
        send_gate.set()
        await _settle()

        self.assertFalse(
            any(event["type"] == "response.cancel" for event in self.websocket.sent)
        )
        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-new",
                "delta": base64.b64encode(b"\x08\x00").decode("ascii"),
            }
        )
        await self.websocket.emit(
            {
                "type": "response.done",
                "response": {"id": "resp-new", "status": "completed"},
            }
        )
        await _settle()
        self.assertTrue(
            any(
                isinstance(frame, TTSAudioRawFrame)
                and frame.audio == b"\x08\x00"
                for frame in self._pushed_frames()
            )
        )
        self.assertTrue(
            any(isinstance(frame, TTSStoppedFrame) for frame in self._pushed_frames())
        )

    async def test_cancel_waiting_at_speech_stop_is_dropped_before_new_response(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-old"}}
        )
        await _settle()

        send_gate = asyncio.Event()
        self.websocket.block_audio_send = send_gate
        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=b"\x07\x00",
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await asyncio.wait_for(self.websocket.audio_send_started.wait(), timeout=1.0)
        await self.websocket.emit({"type": "input_audio_buffer.speech_started"})
        await self.websocket.emit(
            {"type": "input_audio_buffer.speech_stopped", "reason": "turn_valid"}
        )
        await _wait_until(lambda: self.processor._speech_stopped_at > 0.0)

        send_gate.set()
        await _settle()
        self.assertFalse(
            any(event["type"] == "response.cancel" for event in self.websocket.sent)
        )

        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-new"}}
        )
        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-new",
                "delta": base64.b64encode(b"\x08\x00").decode("ascii"),
            }
        )
        await self.websocket.emit(
            {
                "type": "response.done",
                "response": {"id": "resp-new", "status": "completed"},
            }
        )
        await _settle()
        self.assertTrue(
            any(
                isinstance(frame, TTSAudioRawFrame)
                and frame.audio == b"\x08\x00"
                for frame in self._pushed_frames()
            )
        )

    async def test_shutdown_is_bounded_and_survives_caller_cancellation(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-active"}}
        )
        await _wait_until(lambda: self.processor._response_started)
        send_gate = asyncio.Event()
        close_gate = asyncio.Event()
        self.websocket.block_audio_send = send_gate
        self.websocket.block_close = close_gate
        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=b"\x09\x00",
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await asyncio.wait_for(self.websocket.audio_send_started.wait(), timeout=1.0)
        connection_task = self.processor._connection_task
        sender_task = self.processor._microphone_sender_task

        first_caller = asyncio.create_task(self.processor._shutdown())
        await asyncio.wait_for(self.websocket.close_started.wait(), timeout=1.0)
        self.assertFalse(self.processor.health_snapshot()["response_active"])
        first_caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_caller

        await asyncio.wait_for(self.processor._shutdown(), timeout=0.5)
        await asyncio.wait_for(self.processor._shutdown(), timeout=0.1)
        self.assertTrue(connection_task.done())
        self.assertTrue(sender_task.done())
        self.assertTrue(self.processor._shutdown_task.done())
        self.assertIsNone(self.processor._connection_task)
        self.assertIsNone(self.processor._microphone_sender_task)
        self.assertIsNone(self.processor._websocket)
        self.assertEqual(self.processor._microphone_queue.qsize(), 0)
        self.assertFalse(
            any(not task.done() for task in self.processor._auxiliary_tasks)
        )

    async def test_speech_started_cancels_active_response_and_interrupts_sink(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-1"}}
        )
        await self.websocket.emit({"type": "input_audio_buffer.speech_started"})
        await _settle()
        self.assertTrue(
            any(event["type"] == "response.cancel" for event in self.websocket.sent)
        )
        self.assertTrue(
            any(
                isinstance(frame, InterruptionFrame)
                for frame in self._pushed_frames()
            )
        )

    async def test_external_and_provider_interrupt_are_forwarded_once(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-1"}}
        )
        await _settle()
        await self.processor.process_frame(
            InterruptionFrame(),
            FrameDirection.DOWNSTREAM,
        )
        await self.websocket.emit({"type": "input_audio_buffer.speech_started"})
        await _settle()

        interruptions = [
            frame
            for frame in self._pushed_frames()
            if isinstance(frame, InterruptionFrame)
        ]
        self.assertEqual(len(interruptions), 1)

    async def test_terminal_response_cache_evicts_oldest_entry(self):
        epoch = self.processor._connection_epoch
        for index in range(65):
            self.processor._remember_terminal_response(epoch, f"resp-{index}")

        self.assertNotIn((epoch, "resp-0"), self.processor._rejected_responses)
        self.assertIn((epoch, "resp-1"), self.processor._rejected_responses)
        self.assertIn((epoch, "resp-64"), self.processor._rejected_responses)
        self.assertEqual(len(self.processor._rejected_responses), 64)

    async def test_failed_response_interrupts_instead_of_ending_normally(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-failed"}}
        )
        await self.websocket.emit(
            {
                "type": "response.done",
                "response": {"id": "resp-failed", "status": "failed"},
            }
        )
        await _settle()

        frames = self._pushed_frames()
        self.assertTrue(any(isinstance(frame, InterruptionFrame) for frame in frames))
        self.assertFalse(any(isinstance(frame, TTSStoppedFrame) for frame in frames))

    async def test_stale_connection_events_are_dropped_and_old_mic_is_not_replayed(self):
        await self._mark_ready()
        old_epoch = self.processor._connection_epoch
        await self.websocket.disconnect()
        await _wait_until(
            lambda: self.processor._connection_epoch > old_epoch
            or self.processor._websocket is None
        )

        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=b"\x09\x00",
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        sent_before = len(self.websocket.sent)
        await self.processor._handle_provider_event(
            {
                "type": "response.audio.delta",
                "response_id": "stale",
                "delta": base64.b64encode(b"\x09\x00").decode("ascii"),
            },
            old_epoch,
        )
        await _settle()
        self.assertEqual(len(self.websocket.sent), sent_before)
        self.assertFalse(
            any(
                isinstance(frame, TTSAudioRawFrame)
                and frame.audio == b"\x09\x00"
                for frame in self._pushed_frames()
            )
        )

    async def test_disconnect_reconnects_and_does_not_replay_dropped_microphone(self):
        replacement = _FakeWebSocket()
        self.factory.websockets.append(replacement)
        await self._mark_ready()
        await self.websocket.disconnect()
        await asyncio.sleep(0.02)
        await _settle()

        self.assertGreaterEqual(len(self.factory.calls), 2)
        self.assertEqual(replacement.sent[0]["type"], "session.update")
        dropped_pcm = b"\x08\x00"
        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=dropped_pcm,
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        self.assertFalse(
            any(
                event.get("type") == "input_audio_buffer.append"
                and base64.b64decode(event["audio"]) == dropped_pcm
                for event in replacement.sent
            )
        )

        await replacement.emit({"type": "session.updated"})
        await self.processor.wait_ready(timeout=1.0)
        fresh_pcm = b"\x07\x00"
        await self.processor.process_frame(
            InputAudioRawFrame(
                audio=fresh_pcm,
                sample_rate=16_000,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await _settle()
        appended = [
            event
            for event in replacement.sent
            if event.get("type") == "input_audio_buffer.append"
        ]
        self.assertEqual(len(appended), 1)
        self.assertEqual(base64.b64decode(appended[0]["audio"]), fresh_pcm)

    async def test_provider_idle_timeout_rolls_session_without_error_state(self):
        replacement = _FakeWebSocket()
        self.factory.websockets.append(replacement)
        await self._mark_ready()
        old_epoch = self.processor._connection_epoch

        await self.websocket.emit(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "response_idle_timeout",
                    "message": (
                        "Your session was closed because no response "
                        "was generated for 180 seconds."
                    ),
                },
            }
        )
        await _wait_until(
            lambda: self.processor._connection_epoch > old_epoch
        )
        self.assertFalse(self.processor.ready)
        self.assertIsNone(self.processor._last_error)
        self.assertTrue(self.websocket.closed)
        self.assertTrue(
            any(
                "provider idle limit reached" in message
                for message in self.logs
            )
        )
        self.assertFalse(
            any(
                "[PIPECAT S2S ERROR]" in message
                and "response_idle_timeout" in message
                for message in self.logs
            )
        )

        await replacement.emit({"type": "session.updated"})
        await self.processor.wait_ready(timeout=1.0)
        self.assertTrue(self.processor.ready)

    async def test_provider_error_interrupts_without_normal_stop(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-1"}}
        )
        await self.websocket.emit(
            {"type": "error", "error": {"message": "simulated"}}
        )
        await _settle()
        frames = self._pushed_frames()
        self.assertTrue(any(isinstance(frame, InterruptionFrame) for frame in frames))
        self.assertFalse(any(isinstance(frame, TTSStoppedFrame) for frame in frames))

    async def test_benign_response_cancel_error_does_not_cancel_active_response(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-1"}}
        )
        await self.websocket.emit(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "param": "response.cancel",
                    "message": "response already cancelled",
                },
            }
        )
        await self.websocket.emit(
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "delta": base64.b64encode(b"\x05\x00").decode("ascii"),
            }
        )
        await _settle()

        frames = self._pushed_frames()
        self.assertTrue(any(isinstance(frame, TTSAudioRawFrame) for frame in frames))
        self.assertFalse(any(isinstance(frame, InterruptionFrame) for frame in frames))

    async def test_non_cancel_invalid_request_is_visible_and_interrupts_active_response(self):
        await self._mark_ready()
        await self.websocket.emit(
            {"type": "response.created", "response": {"id": "resp-1"}}
        )
        await self.websocket.emit(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_value",
                    "param": "input_audio_buffer.append",
                    "message": "bad audio payload",
                },
            }
        )
        await _settle()

        self.assertIn(
            "bad audio payload",
            self.processor.health_snapshot()["last_error"],
        )
        frames = self._pushed_frames()
        self.assertTrue(any(isinstance(frame, InterruptionFrame) for frame in frames))
        self.assertFalse(any(isinstance(frame, TTSStoppedFrame) for frame in frames))


class QwenAudioFactoryTests(unittest.TestCase):
    def test_workspace_url_and_api_key_precedence(self):
        with patch.dict(
            os.environ,
            {
                "PIPECAT_S2S_API_KEY": "s2s-key",
                "DASHSCOPE_API_KEY": "dashscope-key",
                "PIPECAT_S2S_WORKSPACE_ID": "workspace-123",
            },
            clear=True,
        ):
            processor = create_qwen_audio_s2s_from_env()
        self.assertEqual(processor._api_key, "s2s-key")
        self.assertEqual(
            processor._base_url,
            "wss://workspace-123.cn-beijing.maas.aliyuncs.com"
            "/api-ws/v1/realtime",
        )

    def test_explicit_base_url_overrides_workspace(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "dashscope-key",
                "PIPECAT_S2S_WORKSPACE_ID": "workspace-123",
                "PIPECAT_S2S_BASE_URL": "wss://example.invalid/realtime",
            },
            clear=True,
        ):
            processor = create_qwen_audio_s2s_from_env()
        self.assertEqual(
            processor._base_url,
            "wss://example.invalid/realtime",
        )

    def test_existing_key_can_use_verified_compatibility_endpoint(self):
        with patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "dashscope-key"},
            clear=True,
        ):
            processor = create_qwen_audio_s2s_from_env()
        self.assertEqual(
            processor._base_url,
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        )
        self.assertEqual(processor._microphone_queue_max_chunks, 3)
        self.assertEqual(processor._microphone_send_timeout_sec, 1.0)

    def test_server_vad_reads_endpoint_parameters_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "dashscope-key",
                "PIPECAT_S2S_TURN_DETECTION": "server_vad",
                "PIPECAT_S2S_VAD_SILENCE_MS": "400",
                "PIPECAT_S2S_VAD_THRESHOLD": "0.2",
            },
            clear=True,
        ):
            processor = create_qwen_audio_s2s_from_env()
        self.assertEqual(processor._silence_duration_ms, 400)
        self.assertEqual(processor._threshold, 0.2)
        self.assertEqual(
            processor._session_update_event()["session"]["turn_detection"],
            {
                "type": "server_vad",
                "threshold": 0.2,
                "silence_duration_ms": 400,
            },
        )

    def test_vad_endpoint_parameters_are_range_checked(self):
        with self.assertRaisesRegex(ValueError, "silence_duration_ms"):
            QwenAudioRealtimeS2SProcessor(
                api_key="test-key",
                base_url="wss://workspace.invalid/api-ws/v1/realtime",
                silence_duration_ms=199,
            )
        with self.assertRaisesRegex(ValueError, "threshold"):
            QwenAudioRealtimeS2SProcessor(
                api_key="test-key",
                base_url="wss://workspace.invalid/api-ws/v1/realtime",
                threshold=1.1,
            )

    def test_manual_turn_detection_is_rejected(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "dashscope-key",
                "PIPECAT_S2S_TURN_DETECTION": "manual",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "manual mode is not implemented"):
                create_qwen_audio_s2s_from_env()
