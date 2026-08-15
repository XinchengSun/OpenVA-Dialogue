import json
import sys
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import probe_public_media


class PublicMediaProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_claims_one_id_before_opening_media(self):
        observed = {"status": [], "mic": [], "media": []}

        async def status(request):
            observed["status"].append(request.query.get("client_id"))
            return web.json_response({
                "service_ready": True,
                "phase": "ready",
                "conversation": {"state": "available"},
            })

        async def microphone(request):
            observed["mic"].append(request.query.get("client_id"))
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            await websocket.send_json({"type": "lease_granted"})
            async for _message in websocket:
                pass
            return websocket

        async def media(request):
            observed["media"].append(request.query.get("client_id"))
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            await websocket.send_json({
                "type": "mime",
                "mime": 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
            })
            payload = b"ftyp-moov-moof-mdat-moof-mdat" + (b"x" * 256_000)
            await websocket.send_bytes(payload)
            await websocket.close()
            return websocket

        app = web.Application()
        app.router.add_get("/api/realtime/status", status)
        app.router.add_get("/ws/mic", microphone)
        app.router.add_get("/ws/media", media)
        server = TestServer(app)
        await server.start_server()
        try:
            await probe_public_media.probe(
                str(server.make_url("/")).rstrip("/"),
                "not-a-real-token",
                2.0,
            )
        finally:
            await server.close()

        self.assertEqual(len(observed["status"]), 1)
        self.assertEqual(observed["status"], observed["mic"])
        self.assertEqual(observed["mic"], observed["media"])
        client_id = observed["status"][0]
        self.assertRegex(client_id, r"^[A-Za-z0-9_-]{16,64}$")

    async def test_busy_status_exits_without_attempting_takeover(self):
        websocket_attempts = []

        async def status(_request):
            return web.json_response({
                "service_ready": True,
                "phase": "ready",
                "conversation": {"state": "busy"},
            })

        async def unexpected_websocket(request):
            websocket_attempts.append(request.path)
            return web.Response(status=500)

        app = web.Application()
        app.router.add_get("/api/realtime/status", status)
        app.router.add_get("/ws/mic", unexpected_websocket)
        app.router.add_get("/ws/media", unexpected_websocket)
        server = TestServer(app)
        await server.start_server()
        try:
            with self.assertRaisesRegex(RuntimeError, "did not take over"):
                await probe_public_media.probe(
                    str(server.make_url("/")).rstrip("/"),
                    "not-a-real-token",
                    2.0,
                )
        finally:
            await server.close()

        self.assertEqual(websocket_attempts, [])

    def test_endpoint_urls_do_not_expose_identity_beyond_random_client_id(self):
        client_id = "test-client-123456"
        status = probe_public_media.endpoint_url(
            "https://example.test/base", "/api/realtime/status", client_id
        )
        media = probe_public_media.endpoint_url(
            "https://example.test/base", "/ws/media", client_id, websocket=True
        )
        self.assertEqual(
            status,
            "https://example.test/api/realtime/status?client_id=test-client-123456",
        )
        self.assertEqual(
            media,
            "wss://example.test/ws/media?client_id=test-client-123456",
        )


if __name__ == "__main__":
    unittest.main()
