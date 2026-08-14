import asyncio
import unittest
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


class RealtimeWebSocketLifecycleTests(unittest.IsolatedAsyncioTestCase):
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
