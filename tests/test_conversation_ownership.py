"""Real microphone sockets: history survives reconnect, never crosses tabs."""
import ast
import asyncio
import __future__
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
import uuid
import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

tree = ast.parse((Path(__file__).resolve().parents[1]/'server_mse.py').read_text(encoding='utf-8'))
namespace = dict(asyncio=asyncio, aiohttp=aiohttp, web=web, json=json, uuid=uuid)
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, 'name', '') == 'mic_ws'],
                        type_ignores=[]), '<production mic handler>', 'exec',
             flags=__future__.annotations.compiler_flag), namespace)

def session():
    return SimpleNamespace(closed=asyncio.Event(), task=SimpleNamespace(done=lambda: False),
                           interrupt=AsyncMock(return_value=0), send_audio=AsyncMock(return_value=True),
                           fresh_conversation=AsyncMock())

class ConversationOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.first = session(); self.second = session()
        self.first.fresh_conversation.return_value = self.second
        self.app = web.Application()
        self.app.update(engine=SimpleNamespace(log=Mock(), enqueue_user_audio=Mock()),
                        dialog_backend_name='pipecat', dialog_session=self.first,
                        mic_owner_lock=asyncio.Lock(), mic_input_lock=asyncio.Lock(), mic_close_tasks=set())
        self.app.router.add_get('/ws/mic', namespace['mic_ws'])
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def connect(self, identity):
        ws = await self.client.ws_connect('/ws/mic')
        await ws.send_json({'type':'config', 'conversation_id':identity})
        self.assertEqual((await ws.receive_json(timeout=2))['type'], 'ok')
        return ws

    async def test_same_conversation_reconnect_and_multiple_questions_keep_pipeline(self):
        first = await self.connect('tab-a')
        await first.send_bytes(b'\0\0' * 640)
        await first.send_bytes(b'\1\0' * 640)
        await first.send_json({'type':'ping'})
        await first.receive_json(timeout=2)
        self.assertEqual(self.first.send_audio.await_count, 2)
        await first.close()
        second = await self.connect('tab-a')
        self.first.fresh_conversation.assert_not_awaited()
        self.assertIs(self.app['dialog_session'], self.first)
        await second.close()

    async def test_other_conversation_gets_fresh_pipeline_before_audio(self):
        first = await self.connect('tab-a')
        second = await self.connect('tab-b')
        await second.send_bytes(b'\0\0' * 640)
        await second.send_json({'type':'ping'})
        await second.receive_json(timeout=2)
        self.first.fresh_conversation.assert_awaited_once()
        self.assertIs(self.app['dialog_session'], self.second)
        self.second.send_audio.assert_awaited_once()
        await first.close()
        await second.close()
