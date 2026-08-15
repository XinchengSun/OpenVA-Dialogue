import asyncio
import os
import queue
import unittest
from types import SimpleNamespace
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import server_mse


class FakeLogEngine:
    def __init__(self):
        self.registered = []
        self.unregister_count = 0
        self.unregistered = asyncio.Event()

    def register_log_client(self, queue):
        self.registered.append(queue)

    def unregister_log_client(self, queue):
        self.registered.remove(queue)
        self.unregister_count += 1
        self.unregistered.set()


class FakeMediaClient:
    def __init__(self):
        self.out_q = asyncio.Queue()
        self.stopped = asyncio.Event()
        self.stop_reason = None

    def _signal_stopped(self, reason):
        self.stop_reason = reason
        self.stopped.set()

    @staticmethod
    def filter_output_item(item):
        return item


class FakeMediaEngine:
    def __init__(self, loop):
        self.loop = loop
        self.client = FakeMediaClient()
        self.unregister_count = 0
        self.unregistered = asyncio.Event()

    def register_media_client(self):
        return self.client

    def unregister_media_client(self, client):
        assert client is self.client
        self.unregister_count += 1
        self.loop.call_soon_threadsafe(self.unregistered.set)


class RealtimeWebSocketLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def make_pipe_client(self):
        messages = []
        engine = SimpleNamespace(
            loop=asyncio.get_running_loop(),
            args=SimpleNamespace(segment_frames=5),
            log=messages.append,
        )
        return server_mse.MediaPipeClient(engine, 99), messages

    async def test_full_media_output_queue_stops_client_without_dropping_bytes(self):
        client, messages = self.make_pipe_client()
        for index in range(client.out_q.maxsize):
            client.out_q.put_nowait(("media", 0, bytes([index % 256])))

        await asyncio.wait_for(
            asyncio.to_thread(client._put_bytes, b"new-media", 0),
            timeout=0.5,
        )
        await asyncio.wait_for(client.stopped.wait(), timeout=0.5)

        self.assertFalse(client.running.is_set())
        self.assertEqual(client.out_q.qsize(), client.out_q.maxsize)
        self.assertTrue(any("output queue blocked" in item for item in messages))

    async def test_full_output_queue_drops_logs_but_fails_critical_control(self):
        client, _ = self.make_pipe_client()
        for index in range(client.out_q.maxsize):
            client.out_q.put_nowait(f"queued-{index}")

        client.put_text("ordinary log")
        await asyncio.sleep(0)
        self.assertTrue(client.running.is_set())
        self.assertFalse(client.stopped.is_set())
        self.assertEqual(client.out_q.qsize(), client.out_q.maxsize)

        client.put_text("critical boundary", critical=True)
        await asyncio.wait_for(client.stopped.wait(), timeout=0.5)
        self.assertFalse(client.running.is_set())
        self.assertEqual(client.out_q.qsize(), client.out_q.maxsize)

    async def test_noncritical_logs_leave_half_the_output_queue_reserved(self):
        client, _ = self.make_pipe_client()
        reserve_threshold = max(1, client.out_q.maxsize // 2)

        for index in range(reserve_threshold + 5):
            client.put_text(f"ordinary-log-{index}")
        await asyncio.sleep(0)

        self.assertEqual(client.out_q.qsize(), reserve_threshold)
        self.assertTrue(client.running.is_set())

    async def test_sustained_av_pair_backpressure_stops_stale_connection(self):
        with mock.patch.dict(os.environ, {"PIPE_AV_FULL_FAIL_COUNT": "3"}):
            client, messages = self.make_pipe_client()
        while True:
            try:
                client.av_pair_q.put_nowait((0, 0, 0, b"video", b"audio", []))
            except queue.Full:
                break

        self.assertFalse(client._enqueue_av_unit(0, b"new", b"new"))
        self.assertTrue(client.running.is_set())
        self.assertFalse(client._enqueue_av_unit(0, b"new", b"new"))
        self.assertTrue(client.running.is_set())
        self.assertFalse(client._enqueue_av_unit(0, b"new", b"new"))
        await asyncio.wait_for(client.stopped.wait(), timeout=0.5)

        self.assertFalse(client.running.is_set())
        warnings = [item for item in messages if "AV pair queue full" in item]
        self.assertEqual(len(warnings), 2)

    async def test_transient_av_pair_backpressure_resets_after_one_success(self):
        with mock.patch.dict(os.environ, {"PIPE_AV_FULL_FAIL_COUNT": "2"}):
            client, _ = self.make_pipe_client()
        while True:
            try:
                client.av_pair_q.put_nowait((0, 0, 0, b"video", b"audio", []))
            except queue.Full:
                break

        self.assertFalse(client._enqueue_av_unit(0, b"new", b"new"))
        client.av_pair_q.get_nowait()
        self.assertTrue(client._enqueue_av_unit(0, b"new", b"new"))
        self.assertFalse(client._enqueue_av_unit(0, b"new", b"new"))

        self.assertTrue(client.running.is_set())
        self.assertFalse(client.stopped.is_set())

    async def test_media_send_has_a_hard_timeout(self):
        never = asyncio.Event()

        class BlockedWebSocket:
            async def send_bytes(self, _item):
                await never.wait()

        with self.assertRaises(asyncio.TimeoutError):
            await server_mse._send_media_ws_item(
                BlockedWebSocket(),
                b"media",
                timeout=0.01,
            )

    async def test_failed_media_close_aborts_transport_after_timeout(self):
        never = asyncio.Event()
        transport = mock.Mock()

        class BlockedWebSocket:
            async def close(self, **_kwargs):
                await never.wait()

        request = SimpleNamespace(transport=transport)
        await server_mse._close_failed_media_ws(
            request,
            BlockedWebSocket(),
            timeout=0.01,
        )

        transport.abort.assert_called_once_with()

    async def test_full_log_queue_drops_logs_without_pending_put_tasks(self):
        engine = object.__new__(server_mse.RealtimeMSEEngine)
        engine.loop = asyncio.get_running_loop()
        log_queue = asyncio.Queue(maxsize=1)
        log_queue.put_nowait({"message": "already queued"})
        engine.log_clients = {log_queue}
        engine.media_clients = set()

        with mock.patch("builtins.print"):
            engine.log("overflow log")
        await asyncio.sleep(0)

        self.assertEqual(log_queue.qsize(), 1)
        self.assertEqual(len(log_queue._putters), 0)

    async def test_media_socket_closes_when_encoder_client_stops(self):
        loop = asyncio.get_running_loop()
        engine = FakeMediaEngine(loop)
        app = web.Application()
        app["engine"] = engine
        app["open_websockets"] = set()
        app.router.add_get("/ws/media", server_mse.media_ws)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            websocket = await client.ws_connect("/ws/media")
            mime = await websocket.receive_json()
            connected = await websocket.receive_json()
            self.assertEqual(mime["type"], "mime")
            self.assertEqual(connected["message"], "[MEDIA] connected pipe encoder")

            engine.client.stopped.set()
            closed = await asyncio.wait_for(websocket.receive(), timeout=0.5)

            self.assertEqual(closed.type, aiohttp.WSMsgType.CLOSE)
            self.assertEqual(closed.data, aiohttp.WSCloseCode.INTERNAL_ERROR)
            await asyncio.wait_for(engine.unregistered.wait(), timeout=1.0)
            self.assertEqual(engine.unregister_count, 1)
            self.assertEqual(app["open_websockets"], set())
        finally:
            await client.close()

    async def test_media_socket_rearms_an_inflight_assistant_turn(self):
        loop = asyncio.get_running_loop()
        engine = FakeMediaEngine(loop)
        engine.assistant_turn_control_snapshot = lambda: {
            "type": "assistant_turn_started",
            "generation": 7,
            "turn_id": 12,
        }
        app = web.Application()
        app["engine"] = engine
        app["open_websockets"] = set()
        app.router.add_get("/ws/media", server_mse.media_ws)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            websocket = await client.ws_connect("/ws/media")
            await websocket.receive_json()
            await websocket.receive_json()

            current_turn = await websocket.receive_json()

            self.assertEqual(current_turn["type"], "assistant_turn_started")
            self.assertEqual(current_turn["generation"], 7)
            self.assertEqual(current_turn["turn_id"], 12)
            await websocket.close()
        finally:
            await client.close()

    async def test_media_send_timeout_closes_socket_and_stops_encoder_client(self):
        loop = asyncio.get_running_loop()
        engine = FakeMediaEngine(loop)
        app = web.Application()
        app["engine"] = engine
        app["open_websockets"] = set()
        app.router.add_get("/ws/media", server_mse.media_ws)
        client = TestClient(TestServer(app))
        await client.start_server()
        never = asyncio.Event()
        original_send_bytes = web.WebSocketResponse.send_bytes

        async def blocked_send_bytes(_ws, _data, *args, **kwargs):
            await never.wait()

        try:
            websocket = await client.ws_connect("/ws/media")
            await websocket.receive_json()
            await websocket.receive_json()
            with (
                mock.patch.dict(
                    os.environ,
                    {"MEDIA_WS_SEND_TIMEOUT_SEC": "0.05"},
                ),
                mock.patch.object(
                    web.WebSocketResponse,
                    "send_bytes",
                    blocked_send_bytes,
                ),
            ):
                # Reconnect so the handler reads the patched timeout value.
                await websocket.close()
                await asyncio.wait_for(engine.unregistered.wait(), timeout=1.0)
                engine = FakeMediaEngine(loop)
                app["engine"] = engine
                websocket = await client.ws_connect("/ws/media")
                await websocket.receive_json()
                await websocket.receive_json()
                engine.client.out_q.put_nowait(b"blocked media")
                closed = await asyncio.wait_for(websocket.receive(), timeout=0.5)

            self.assertEqual(closed.type, aiohttp.WSMsgType.CLOSE)
            self.assertEqual(closed.data, aiohttp.WSCloseCode.INTERNAL_ERROR)
            self.assertIn("TimeoutError", engine.client.stop_reason)
            await asyncio.wait_for(engine.unregistered.wait(), timeout=1.0)
        finally:
            web.WebSocketResponse.send_bytes = original_send_bytes
            await client.close()

    async def test_log_client_close_is_observed_without_waiting_for_a_log(self):
        engine = FakeLogEngine()
        app = web.Application()
        app["engine"] = engine
        app["open_websockets"] = set()
        app.router.add_get("/ws/logs", server_mse.logs_ws)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            websocket = await client.ws_connect("/ws/logs")
            connected = await websocket.receive_json()
            self.assertEqual(connected["message"], "[LOG] connected")
            self.assertEqual(len(app["open_websockets"]), 1)

            await websocket.close()
            await asyncio.wait_for(engine.unregistered.wait(), timeout=1.0)

            self.assertEqual(engine.unregister_count, 1)
            self.assertEqual(engine.registered, [])
            self.assertEqual(app["open_websockets"], set())
        finally:
            await client.close()

    async def test_server_shutdown_closes_every_registered_websocket(self):
        first = mock.AsyncMock()
        second = mock.AsyncMock()
        app = {"open_websockets": {first, second}}

        await server_mse.close_open_websockets(app)

        for websocket in (first, second):
            websocket.close.assert_awaited_once_with(
                code=aiohttp.WSCloseCode.GOING_AWAY,
                message=b"server shutdown",
            )


if __name__ == "__main__":
    unittest.main()
