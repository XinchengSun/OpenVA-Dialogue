"""Real HTTP/WebSocket checks for CPU gateway failure and recovery paths."""
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from scripts.serve_worker_proxy import create_app


class WorkerProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / 'availability.json'

    async def client(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def test_stopped_worker_is_visible_without_attempting_dead_upstream(self):
        self.state.write_text(json.dumps({'status': 'unavailable', 'reason': 'gpu_stopped',
                                         'message': '原 GPU 作业已停止。<test>'}), encoding='utf-8')
        proxy = await self.client(create_app('http://127.0.0.1:1', self.state))
        root = await proxy.get('/', headers={'Accept': 'text/html'})
        self.assertEqual(root.status, 503)
        self.assertIn('GPU 作业已停止', await root.text())
        self.assertIn('&lt;test&gt;', await root.text())
        health = await proxy.get('/health')
        self.assertEqual(health.status, 503)
        self.assertEqual((await health.json())['reason'], 'gpu_stopped')
        self.assertEqual(health.headers['Cache-Control'], 'no-store')
        media = await proxy.get('/ws/media', headers={'Upgrade': 'websocket'})
        self.assertEqual(media.status, 503)

    async def test_missing_or_invalid_record_fails_closed(self):
        proxy = await self.client(create_app('http://127.0.0.1:1', self.state))
        for content in (None, '{', '{}', '{"status":"ready","upstream":null}'):
            if content is not None:
                self.state.write_text(content, encoding='utf-8')
            response = await proxy.get('/health')
            self.assertEqual(response.status, 503)
            self.assertEqual((await response.json())['reason'], 'configuration_unavailable')

    async def test_ready_record_does_not_fake_worker_health(self):
        self.state.write_text(json.dumps({'status': 'ready', 'upstream': 'http://127.0.0.1:1'}))
        proxy = await self.client(create_app('http://127.0.0.1:1', self.state))
        response = await proxy.get('/health')
        self.assertEqual(response.status, 503)
        self.assertEqual((await response.json())['reason'], 'worker_unreachable')

    async def test_worker_http_error_and_websocket_rejection_remain_failures(self):
        worker = web.Application()
        async def unhealthy(request):
            return web.json_response({'ready': False}, status=503)
        worker.router.add_get('/health', unhealthy)
        backend = await self.client(worker)
        proxy = await self.client(create_app(str(backend.make_url('/'))))
        response = await proxy.get('/health')
        self.assertEqual(response.status, 503)
        self.assertEqual(await response.json(), {'ready': False})
        rejected = await proxy.get('/health', headers={
            'Upgrade': 'websocket', 'Connection': 'Upgrade'})
        self.assertEqual(rejected.status, 503)
        self.assertEqual((await rejected.json())['reason'], 'worker_unreachable')

    async def test_actual_http_and_websocket_recover_without_gateway_restart(self):
        worker = web.Application()

        async def health(request):
            self.assertNotIn('Cookie', request.headers)
            self.assertNotIn('Authorization', request.headers)
            return web.json_response({'worker': 'ready', 'query': request.query.get('sample')})

        async def media(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                if isinstance(message.data, bytes):
                    await ws.send_bytes(message.data)
                else:
                    await ws.send_str(message.data)
            return ws

        async def segments(request):
            response = web.StreamResponse(headers={'Content-Type': 'application/octet-stream'})
            await response.prepare(request)
            for _ in range(4):
                await response.write(bytes(range(256)))
            await response.write_eof()
            return response

        worker.router.add_get('/health', health)
        worker.router.add_get('/ws/media', media)
        worker.router.add_get('/segment', segments)
        backend = await self.client(worker)
        self.state.write_text('{"status":"unavailable","reason":"gpu_stopped","message":"离线"}',
                              encoding='utf-8')
        proxy = await self.client(create_app('http://127.0.0.1:1', self.state))
        self.assertEqual((await proxy.get('/health')).status, 503)
        self.state.write_text(json.dumps({'status': 'ready', 'upstream': str(backend.make_url('/'))}))
        response = await proxy.get('/health?sample=中文', headers={
            'Cookie': 'private=example', 'Authorization': 'Bearer example'})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {'worker': 'ready', 'query': '中文'})
        segment = await proxy.get('/segment')
        self.assertEqual(await segment.read(), bytes(range(256)) * 4)
        async with proxy.ws_connect('/ws/media') as ws:
            await ws.send_str('hello')
            self.assertEqual((await ws.receive(timeout=2)).data, 'hello')
            await ws.send_bytes(b'\x00\x01media')
            self.assertEqual((await ws.receive(timeout=2)).data, b'\x00\x01media')


if __name__ == '__main__':
    unittest.main()
