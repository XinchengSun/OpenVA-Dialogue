import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import server_mse


CLIENT_A = "client-a-1234567890"
CLIENT_B = "client-b-1234567890"


class FakeEngine:
    def __init__(self):
        self.media_clients = set()
        self.user_audio = []
        self.messages = []

    def health_snapshot(self):
        return {
            "status": "ok",
            "launch_token": "must-not-be-public",
            "workers": {"secret": True},
        }

    def enqueue_user_audio(self, audio):
        self.user_audio.append(audio)

    def log(self, message):
        self.messages.append(message)

    def register_media_client(self):
        client = FakeMediaClient()
        self.media_clients.add(client)
        return client

    def unregister_media_client(self, client):
        self.media_clients.discard(client)


class FakeMediaClient:
    def __init__(self):
        self.out_q = asyncio.Queue()
        self.stopped = asyncio.Event()

    @staticmethod
    def filter_output_item(item):
        return item

    def _signal_stopped(self, _reason):
        self.stopped.set()


class FakeDialog:
    def __init__(self):
        self.closed = asyncio.Event()
        self.stop = asyncio.Event()
        self.task = asyncio.create_task(self.stop.wait())
        self.reset_count = 0
        self.interrupt_count = 0
        self.audio = []

    async def reset_dialog(self):
        self.reset_count += 1

    async def interrupt(self):
        self.interrupt_count += 1
        return 3

    async def send_audio(self, audio):
        self.audio.append(audio)
        return True

    def health_snapshot(self):
        return {
            "ready": True,
            "tts_active": False,
            "custom_cascade": {
                "tts": {
                    "active_requests": 0,
                    "bridge_uri": "ws://secret",
                    "model": "secret-model",
                    "voice": "secret-voice",
                }
            },
        }

    async def cleanup(self):
        self.stop.set()
        await asyncio.gather(self.task, return_exceptions=True)


class FakeAuxSocket:
    def __init__(self):
        self.closed = False
        self.close_calls = []

    async def close(self, **kwargs):
        self.close_calls.append(kwargs)
        self.closed = True


class ConversationLeaseUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_peer_cannot_spoof_loopback_host(self):
        request = SimpleNamespace(
            host="127.0.0.1:6008",
            remote="203.0.113.10",
            path="/",
            query={},
            cookies={},
            headers={},
        )
        handler = mock.AsyncMock(return_value=web.Response(status=204))
        with mock.patch.dict(os.environ, {"PUBLIC_ACCESS_TOKEN": "secret"}):
            response = await server_mse.public_access_gate(request, handler)

        self.assertEqual(response.status, 401)
        handler.assert_not_awaited()
        with self.assertRaises(web.HTTPForbidden):
            await server_mse.logs_ws(request)

    async def test_customization_paths_use_the_separate_admin_gate(self):
        request = SimpleNamespace(
            host="public.example",
            remote="203.0.113.10",
            path="/customize/login",
            query={},
            cookies={},
            headers={},
        )
        expected = web.Response(status=204)
        handler = mock.AsyncMock(return_value=expected)
        with mock.patch.dict(os.environ, {"PUBLIC_ACCESS_TOKEN": "secret"}):
            response = await server_mse.public_access_gate(request, handler)

        self.assertIs(response, expected)
        handler.assert_awaited_once_with(request)

    async def test_forwarded_public_peer_cannot_claim_loopback_bypass(self):
        request = SimpleNamespace(
            host="127.0.0.1:6008",
            remote="127.0.0.1",
            path="/",
            query={},
            cookies={},
            headers={"CF-Connecting-IP": "203.0.113.11"},
        )
        handler = mock.AsyncMock(return_value=web.Response(status=204))
        with mock.patch.dict(os.environ, {"PUBLIC_ACCESS_TOKEN": "secret"}):
            response = await server_mse.public_access_gate(request, handler)

        self.assertEqual(response.status, 401)
        handler.assert_not_awaited()
        with self.assertRaises(web.HTTPForbidden):
            await server_mse.logs_ws(request)

    async def test_one_owner_and_atomic_aux_release(self):
        lease = server_mse.ConversationLease(timeout_sec=20)
        first = object()
        second = object()
        epoch = await lease.try_acquire(first, CLIENT_A)

        self.assertIsNotNone(epoch)
        self.assertIsNone(await lease.try_acquire(second, CLIENT_B))
        self.assertTrue(lease.snapshot(CLIENT_A)["owner_self"])
        self.assertFalse(lease.snapshot(CLIENT_B)["owner_self"])

        auxiliary = FakeAuxSocket()
        self.assertTrue(await lease.bind_aux(CLIENT_A, auxiliary))
        release_sockets = await lease.begin_release(first, epoch)
        self.assertEqual(release_sockets, [auxiliary])
        self.assertFalse(await lease.bind_aux(CLIENT_A, FakeAuxSocket()))

        await server_mse._close_lease_sockets(release_sockets)
        self.assertTrue(auxiliary.closed)
        self.assertTrue(await lease.finish_release(first, epoch))
        self.assertFalse(lease.active)

    async def test_stale_owner_cannot_touch_or_release_new_owner(self):
        lease = server_mse.ConversationLease(timeout_sec=20)
        first = object()
        second = object()
        first_epoch = await lease.try_acquire(first, CLIENT_A)
        self.assertEqual(await lease.begin_release(first, first_epoch), [])
        self.assertTrue(await lease.finish_release(first, first_epoch))
        second_epoch = await lease.try_acquire(second, CLIENT_B)

        self.assertFalse(await lease.touch(first, first_epoch))
        self.assertIsNone(await lease.begin_release(first, first_epoch))
        self.assertTrue(await lease.is_owner(second, second_epoch))

    async def test_release_and_auxiliary_bind_are_serialized(self):
        lease = server_mse.ConversationLease(timeout_sec=20)
        owner = object()
        epoch = await lease.try_acquire(owner, CLIENT_A)
        auxiliary = FakeAuxSocket()

        # Hold the lock so both operations queue in a deterministic order.
        await lease._lock.acquire()
        release_task = asyncio.create_task(lease.begin_release(owner, epoch))
        await asyncio.sleep(0)
        bind_task = asyncio.create_task(lease.bind_aux(CLIENT_A, auxiliary))
        lease._lock.release()

        self.assertEqual(await release_task, [])
        self.assertFalse(await bind_task)
        self.assertEqual(auxiliary.close_calls, [])


class ConversationLeaseHTTPTests(unittest.IsolatedAsyncioTestCase):
    PUBLIC_HEADERS = {"Host": "public.example", "Origin": "https://public.example"}

    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.dialog = FakeDialog()
        self.lease = server_mse.ConversationLease(timeout_sec=20)
        self.app = web.Application()
        self.app["engine"] = self.engine
        self.app["dialog_backend_name"] = "pipecat"
        self.app["dialog_session"] = self.dialog
        self.app["conversation_lease"] = self.lease
        self.app["mic_owner_lock"] = asyncio.Lock()
        self.app["mic_input_lock"] = asyncio.Lock()
        self.app["workload_admission_lock"] = asyncio.Lock()
        self.app["customization_prepare_lock"] = asyncio.Lock()
        self.app["customization_activation_start_lock"] = asyncio.Lock()
        self.app["customization_activation_state"] = {"job_id": None}
        self.app["active_mic_ws"] = None
        self.app["mic_cleanup_tasks"] = set()
        self.app["open_websockets"] = set()
        self.app.router.add_get("/ws/mic", server_mse.mic_ws)
        self.app.router.add_get("/ws/media", server_mse.media_ws)
        self.app.router.add_get("/ws/logs", server_mse.logs_ws)
        self.app.router.add_get("/api/realtime/status", server_mse.realtime_status)
        self.app.router.add_get("/health", server_mse.health)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.dialog.cleanup()

    async def wait_for_lease_release(self):
        for _ in range(100):
            if not self.lease.active:
                return
            await asyncio.sleep(0.01)
        self.fail("microphone lease was not released")

    async def test_second_microphone_is_busy_without_replacing_owner(self):
        first = await self.client.ws_connect(f"/ws/mic?client_id={CLIENT_A}")
        granted = await first.receive_json()
        self.assertEqual(granted["type"], "lease_granted")
        self.assertEqual(self.dialog.reset_count, 1)

        second = await self.client.ws_connect(f"/ws/mic?client_id={CLIENT_B}")
        busy = await second.receive_json()
        self.assertEqual(busy["type"], "lease_busy")
        self.assertFalse(first.closed)
        self.assertEqual(self.dialog.interrupt_count, 0)

        await first.send_bytes(b"owner-audio")
        for _ in range(100):
            if self.dialog.audio:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.dialog.audio, [b"owner-audio"])

        await first.send_json({"type": "lease_heartbeat"})
        self.assertEqual((await first.receive_json())["type"], "pong")
        await first.close()
        await self.wait_for_lease_release()
        self.assertEqual(self.dialog.interrupt_count, 1)

        third = await self.client.ws_connect(f"/ws/mic?client_id={CLIENT_B}")
        self.assertEqual((await third.receive_json())["type"], "lease_granted")
        await third.close()

    async def test_customization_blocks_microphone_admission(self):
        await self.app["customization_prepare_lock"].acquire()
        try:
            websocket = await self.client.ws_connect(f"/ws/mic?client_id={CLIENT_A}")
            busy = await websocket.receive_json()
            self.assertEqual(busy["type"], "service_busy")
            self.assertEqual(busy["reason"], "customizing")
            self.assertFalse(self.lease.active)
            self.assertEqual(self.dialog.reset_count, 0)
        finally:
            self.app["customization_prepare_lock"].release()

    async def test_persisted_customization_state_blocks_after_restart(self):
        with mock.patch.object(
            server_mse,
            "runtime_customization_workload_active",
            return_value=True,
        ):
            websocket = await self.client.ws_connect(
                f"/ws/mic?client_id={CLIENT_A}"
            )
            busy = await websocket.receive_json()
        self.assertEqual(busy["type"], "service_busy")
        self.assertEqual(busy["reason"], "customizing")
        self.assertFalse(self.lease.active)

    async def test_public_media_requires_current_owner_and_logs_are_local_only(self):
        with self.assertRaises(aiohttp.WSServerHandshakeError) as media_error:
            await self.client.ws_connect(
                f"/ws/media?client_id={CLIENT_A}",
                headers=self.PUBLIC_HEADERS,
            )
        self.assertEqual(media_error.exception.status, 409)

        with self.assertRaises(aiohttp.WSServerHandshakeError) as log_error:
            await self.client.ws_connect(
                "/ws/logs",
                headers={"Host": "public.example"},
            )
        self.assertEqual(log_error.exception.status, 403)

    async def test_public_media_binds_only_after_matching_lease_grant(self):
        microphone = await self.client.ws_connect(
            f"/ws/mic?client_id={CLIENT_A}",
            headers=self.PUBLIC_HEADERS,
        )
        self.assertEqual((await microphone.receive_json())["type"], "lease_granted")

        with self.assertRaises(aiohttp.WSServerHandshakeError) as other_error:
            await self.client.ws_connect(
                f"/ws/media?client_id={CLIENT_B}",
                headers=self.PUBLIC_HEADERS,
            )
        self.assertEqual(other_error.exception.status, 409)

        media = await self.client.ws_connect(
            f"/ws/media?client_id={CLIENT_A}",
            headers=self.PUBLIC_HEADERS,
        )
        self.assertEqual((await media.receive_json())["type"], "mime")
        self.assertEqual((await media.receive_json())["type"], "log")

        await microphone.close()
        closed = await asyncio.wait_for(media.receive(), timeout=1.0)
        self.assertIn(closed.type, {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED})
        await self.wait_for_lease_release()

    async def test_status_is_client_aware_and_strictly_sanitized(self):
        available = await self.client.get(
            f"/api/realtime/status?client_id={CLIENT_A}",
            headers={"Host": "public.example"},
        )
        self.assertEqual(available.status, 200)
        self.assertEqual((await available.json())["conversation"]["state"], "available")

        owner = object()
        epoch = await self.lease.try_acquire(owner, CLIENT_A)
        own_response = await self.client.get(
            f"/api/realtime/status?client_id={CLIENT_A}",
            headers={"Host": "public.example"},
        )
        own_payload = await own_response.json()
        self.assertEqual(own_payload["conversation"]["state"], "yours")
        self.assertNotIn("client_id", json.dumps(own_payload))

        other_response = await self.client.get(
            f"/api/realtime/status?client_id={CLIENT_B}",
            headers={"Host": "public.example"},
        )
        other_payload = await other_response.json()
        self.assertEqual(other_payload["conversation"]["state"], "busy")
        encoded = json.dumps(other_payload)
        for forbidden in (
            "launch_token",
            "bridge_uri",
            "secret-model",
            "secret-voice",
            "workers",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(other_response.headers["Cache-Control"], "no-store")

        public_health = await self.client.get(
            "/health",
            headers={"Host": "public.example"},
        )
        self.assertEqual(public_health.status, 200)
        health_payload = await public_health.json()
        self.assertNotIn("launch_token", json.dumps(health_payload))

        local_health = await self.client.get("/health")
        self.assertEqual(local_health.status, 200)
        self.assertEqual((await local_health.json())["launch_token"], "must-not-be-public")

        self.assertEqual(await self.lease.begin_release(owner, epoch), [])
        await self.lease.finish_release(owner, epoch)

    async def test_public_status_requires_valid_client_id(self):
        missing = await self.client.get(
            "/api/realtime/status",
            headers={"Host": "public.example"},
        )
        invalid = await self.client.get(
            "/api/realtime/status?client_id=short",
            headers={"Host": "public.example"},
        )
        self.assertEqual(missing.status, 400)
        self.assertEqual(invalid.status, 400)

        with self.assertRaises(aiohttp.WSServerHandshakeError) as mic_error:
            await self.client.ws_connect(
                "/ws/mic",
                headers=self.PUBLIC_HEADERS,
            )
        self.assertEqual(mic_error.exception.status, 400)

    async def test_public_websockets_reject_cross_origin_cookie_reuse(self):
        for path in (
            f"/ws/mic?client_id={CLIENT_A}",
            f"/ws/media?client_id={CLIENT_A}",
        ):
            with self.assertRaises(aiohttp.WSServerHandshakeError) as error:
                await self.client.ws_connect(
                    path,
                    headers={
                        "Host": "public.example",
                        "Origin": "https://attacker.trycloudflare.com",
                    },
                )
            self.assertEqual(error.exception.status, 403)
        self.assertFalse(self.lease.active)

    async def test_loopback_diagnostics_can_omit_client_id(self):
        status = await self.client.get("/api/realtime/status")
        self.assertEqual(status.status, 200)
        self.assertEqual((await status.json())["conversation"]["state"], "available")

        microphone = await self.client.ws_connect("/ws/mic")
        self.assertEqual((await microphone.receive_json())["type"], "lease_granted")
        await microphone.close()
        await self.wait_for_lease_release()

    async def test_status_reports_customization_phase(self):
        await self.app["customization_prepare_lock"].acquire()
        try:
            response = await self.client.get(
                f"/api/realtime/status?client_id={CLIENT_A}",
                headers={"Host": "public.example"},
            )
            payload = await response.json()
            self.assertEqual(payload["phase"], "customizing")
            self.assertEqual(payload["conversation"]["state"], "unavailable")
            self.assertFalse(payload["conversation"]["available"])
        finally:
            self.app["customization_prepare_lock"].release()


if __name__ == "__main__":
    unittest.main()
