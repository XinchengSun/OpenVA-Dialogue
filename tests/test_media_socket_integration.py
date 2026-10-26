"""Real localhost aiohttp WebSockets around production handlers; no GPU/FFmpeg.

Only the engine/encoder boundary is replaced. Socket send, receive, handshake,
peer closure, queue forwarding, and handler cleanup use real aiohttp code.
"""
import asyncio
import json
import threading
import unittest
from unittest.mock import patch

from aiohttp import web, WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from tests.test_media_backpressure import Client, NAMESPACE


class EngineBoundary:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.logs = set()
        self.media = set()
        self.log_closed = asyncio.Event()
        self.media_closed = asyncio.Event()

    def log(self, _message):
        pass

    def register_log_client(self, output):
        self.logs.add(output)

    def unregister_log_client(self, output):
        self.logs.remove(output)
        self.log_closed.set()

    def register_media_client(self):
        client = Client(self, 1)
        # The encoder boundary has a completed no-op worker so production stop()
        # can join it normally without launching FFmpeg or a model.
        client.thread = threading.Thread(target=lambda: None)
        client.thread.start()
        client.thread.join()
        self.media.add(client)
        return client

    def unregister_media_client(self, client):
        client.stop()
        self.media.remove(client)
        self.loop.call_soon_threadsafe(self.media_closed.set)


class MediaSocketIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.globals = patch.dict(NAMESPACE, web=web)
        self.globals.start()
        self.addCleanup(self.globals.stop)
        self.engine = EngineBoundary()
        application = web.Application()
        application["engine"] = self.engine
        application.router.add_get("/ws/logs", NAMESPACE["logs_ws"])
        application.router.add_get("/ws/media", NAMESPACE["media_ws"])
        self.client = TestClient(TestServer(application))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_quiet_log_peer_closes_and_unregisters_over_real_socket(self):
        socket = await self.client.ws_connect("/ws/logs")
        greeting = await asyncio.wait_for(socket.receive(), 2)
        self.assertEqual(greeting.type, WSMsgType.TEXT)
        self.assertEqual(len(self.engine.logs), 1)
        await asyncio.wait_for(socket.close(), 2)
        await asyncio.wait_for(self.engine.log_closed.wait(), 2)
        self.assertEqual(self.engine.logs, set())

    async def test_log_burst_keeps_control_and_binary_order_on_real_socket(self):
        socket = await self.client.ws_connect("/ws/media")
        await asyncio.wait_for(socket.receive(), 2)  # MIME greeting
        await asyncio.wait_for(socket.receive(), 2)  # Connected diagnostic
        client = next(iter(self.engine.media))
        for index in range(64):
            client.put_log(json.dumps({"type": "log", "message": str(index)}))
        admitted = client.put_text(json.dumps({"type": "assistant_media_boundary", "turn_id": 7}))
        self.assertTrue(admitted)
        self.assertTrue(client.running.is_set())
        seen_boundary = False
        for _ in range(65):
            message = await asyncio.wait_for(socket.receive(), 2)
            self.assertEqual(message.type, WSMsgType.TEXT)
            if json.loads(message.data).get("type") == "assistant_media_boundary":
                seen_boundary = True
                break
        self.assertTrue(seen_boundary)

        # Known synthetic bytes test transport order, not MP4 decoding validity.
        client.out_q.put_nowait(b"synthetic-frame-1")
        client.out_q.put_nowait(b"synthetic-frame-2")
        client.out_q.put_nowait(json.dumps({"type": "assistant_media_ended", "turn_id": 7}))
        received = []
        while len(received) < 3:
            message = await asyncio.wait_for(socket.receive(), 2)
            if message.type == WSMsgType.BINARY:
                received.append(message.data)
            elif message.type == WSMsgType.TEXT:
                payload = json.loads(message.data)
                if payload.get("type") != "log":
                    received.append(payload["type"])
            else:
                self.fail(f"unexpected socket state: {message.type}")
        self.assertEqual(received, [b"synthetic-frame-1", b"synthetic-frame-2", "assistant_media_ended"])
        await asyncio.wait_for(socket.close(), 2)
        await asyncio.wait_for(self.engine.media_closed.wait(), 2)
        self.assertEqual(self.engine.media, set())


if __name__ == "__main__":
    unittest.main()
