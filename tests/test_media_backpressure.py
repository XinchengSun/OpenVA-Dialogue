"""CPU behavioral tests of the actual media transport, without GPU imports."""

import ast
import asyncio
import __future__
import json
import math
import os
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import aiohttp
from latency_trace import trace, media_fields


SOURCE = Path(__file__).resolve().parents[1] / "server_mse.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
NAMESPACE = dict(asyncio=asyncio, threading=threading, queue=queue, os=os,
                 json=json, aiohttp=aiohttp, math=math, trace=trace, media_fields=media_fields)
NAMES = {"_BoundedLoopQueueWriter", "MediaPipeClient", "_forward_websocket_queue",
         "_websocket_send_timeout", "_send_websocket_item", "media_ws", "logs_ws"}
exec(compile(ast.Module(body=[node for node in TREE.body
                             if getattr(node, "name", None) in NAMES], type_ignores=[]),
             str(SOURCE), "exec", flags=__future__.annotations.compiler_flag), NAMESPACE)
Writer = NAMESPACE["_BoundedLoopQueueWriter"]
Client = NAMESPACE["MediaPipeClient"]
forward = NAMESPACE["_forward_websocket_queue"]


class HeldLoop:
    def __init__(self):
        self.callbacks = []

    def call_soon_threadsafe(self, callback):
        self.callbacks.append(callback)

    def flush(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            callback()


class QueueAdmissionTests(unittest.TestCase):
    def test_log_storm_bounds_callbacks_before_event_loop_runs(self):
        loop = HeldLoop()
        output = asyncio.Queue(maxsize=64)
        writer = Writer(loop, output)
        accepted = sum(writer.submit(str(i), best_effort=True) for i in range(10000))
        self.assertEqual(accepted, 64)
        self.assertEqual(len(loop.callbacks), 64)
        self.assertEqual(writer.snapshot()["dropped_best_effort"], 9936)
        self.assertEqual(writer.snapshot()["pending_callbacks"], 64)
        loop.flush()
        self.assertEqual(writer.snapshot()["pending_callbacks"], 0)
        self.assertEqual(output.qsize(), 64)
        for i in range(10000):
            writer.submit(str(i), best_effort=True)
        loop.flush()
        self.assertEqual(output.qsize(), 64)
        self.assertFalse(output._putters, "logs must not leave suspended queue.put tasks")

    def test_log_overflow_keeps_encoded_bytes_and_critical_overflow_stops_client(self):
        loop = HeldLoop()
        running = threading.Event()
        running.set()
        output = asyncio.Queue(maxsize=1)
        encoded = ("media", 0, b"partial-moof-or-mdat")
        output.put_nowait(encoded)
        writer = Writer(loop, output, on_overflow=running.clear)
        writer.submit("diagnostic log", best_effort=True)
        loop.flush()
        self.assertTrue(running.is_set())
        writer.submit("assistant_media_boundary")
        loop.flush()
        self.assertFalse(running.is_set())
        self.assertEqual(output.get_nowait(), encoded)

    def test_close_rejects_new_and_already_scheduled_messages(self):
        loop = HeldLoop()
        output = asyncio.Queue(maxsize=2)
        writer = Writer(loop, output)
        self.assertTrue(writer.submit("old control"))
        writer.close()
        self.assertFalse(writer.submit("new control"))
        loop.flush()
        self.assertTrue(output.empty())

    def test_stalled_loop_control_overflow_is_fail_safe(self):
        loop = HeldLoop()
        overflow = []
        writer = Writer(loop, asyncio.Queue(maxsize=1), lambda: overflow.append(True))
        self.assertTrue(writer.submit("first"))
        self.assertFalse(writer.submit("second"))
        self.assertEqual(overflow, [True])


class FakeSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.block_send = False
        self.send_cancelled = False

    async def prepare(self, request):
        return None

    async def receive(self):
        return await self.incoming.get()

    async def send_bytes(self, item):
        if self.block_send:
            try:
                await asyncio.Event().wait()
            finally:
                self.send_cancelled = True
        self.sent.append(item)

    async def send_str(self, item):
        await self.send_bytes(item)

    def disconnect(self):
        self.incoming.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))


class WebSocketLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_consumer_survives_log_burst_and_receives_boundary(self):
        ws = FakeSocket()
        engine = SimpleNamespace(loop=asyncio.get_running_loop(), log=lambda message: None)
        client = Client(engine, 1)
        forwarding = asyncio.create_task(forward(
            ws, client.out_q, client.filter_output_item, client.running.is_set,
            diagnostic_queue=client.log_q,
        ))
        await asyncio.sleep(0)
        for index in range(10000):
            client.put_log(f"diagnostic {index}")
        # The socket can consume, but the event loop has not run log callbacks.
        self.assertEqual(client.out_q.qsize(), 0)
        self.assertTrue(client.put_text("assistant_media_boundary"))
        for _ in range(100):
            if "assistant_media_boundary" in ws.sent:
                break
            await asyncio.sleep(0.001)
        self.assertTrue(client.running.is_set())
        self.assertIn("assistant_media_boundary", ws.sent)
        self.assertEqual(client.transport_snapshot()["rejected_controls"], 0)
        self.assertGreater(client.transport_snapshot()["dropped_best_effort"], 0)
        ws.disconnect()
        await asyncio.wait_for(forwarding, 1)
        self.assertFalse(client.log_q._getters)

    async def test_diagnostics_never_reorder_critical_fifo(self):
        ws = FakeSocket()
        output, diagnostic = asyncio.Queue(), asyncio.Queue()
        for item in [b"init", "boundary", b"moof", b"mdat"]:
            output.put_nowait(item)
        for item in ["log one", "log two"]:
            diagnostic.put_nowait(item)
        task = asyncio.create_task(forward(ws, output, lambda item: item, diagnostic_queue=diagnostic))
        for _ in range(100):
            if len(ws.sent) == 6:
                break
            await asyncio.sleep(0.001)
        ws.disconnect()
        await asyncio.wait_for(task, 1)
        self.assertEqual([item for item in ws.sent if item not in ("log one", "log two")],
                         [b"init", "boundary", b"moof", b"mdat"])
        self.assertEqual(ws.sent[0], b"init")

    async def test_idle_disconnect_releases_waiting_queue_getter(self):
        ws = FakeSocket()
        output = asyncio.Queue()
        task = asyncio.create_task(forward(ws, output, lambda item: item))
        await asyncio.sleep(0)
        ws.disconnect()
        await asyncio.wait_for(task, 1)
        self.assertFalse(output._getters)
        self.assertFalse(ws.incoming._getters)

    async def test_stream_preserves_order_and_filters_stale_epoch(self):
        ws = FakeSocket()
        engine = SimpleNamespace(loop=asyncio.get_running_loop(), log=lambda message: None)
        client = Client(engine, 1)
        client._stream_epoch = 2
        for item in [("media", 1, b"stale"), "control", ("media", 2, b"moof"),
                     ("media", 2, b"mdat")]:
            client.out_q.put_nowait(item)
        task = asyncio.create_task(forward(ws, client.out_q, client.filter_output_item))
        for _ in range(100):
            if len(ws.sent) == 3:
                break
            await asyncio.sleep(0.001)
        ws.disconnect()
        await asyncio.wait_for(task, 1)
        self.assertEqual(ws.sent, ["control", b"moof", b"mdat"])

    async def test_control_overflow_stops_only_offending_media_client(self):
        engine = SimpleNamespace(loop=asyncio.get_running_loop(), log=lambda message: None)
        slow, healthy = Client(engine, 1), Client(engine, 2)
        for _ in range(slow.out_q.maxsize):
            slow.out_q.put_nowait(b"encoded")
        slow.put_text("critical boundary")
        healthy.put_text("critical boundary")
        await asyncio.sleep(0)
        self.assertFalse(slow.running.is_set())
        self.assertTrue(healthy.running.is_set())
        self.assertEqual(await healthy.out_q.get(), "critical boundary")
        self.assertEqual(slow.transport_snapshot()["rejected_controls"], 1)
        self.assertEqual(healthy.transport_snapshot()["rejected_controls"], 0)

    async def test_reset_notification_timeout_stops_client_and_cancels_write(self):
        engine = SimpleNamespace(loop=asyncio.get_running_loop(), log=lambda message: None)
        client = Client(engine, 1)
        cancelled = []

        def stalled_submit(coro, loop):
            coro.close()

            def expired(timeout):
                raise TimeoutError("event loop stalled")

            return SimpleNamespace(result=expired, cancel=lambda: cancelled.append(True))

        proxy = SimpleNamespace(**vars(asyncio))
        proxy.run_coroutine_threadsafe = stalled_submit
        with patch.dict(NAMESPACE, asyncio=proxy):
            client._notify_stream_reset()
        self.assertEqual(cancelled, [True])
        self.assertFalse(client.running.is_set())

    async def test_stalled_send_is_cancelled_and_both_tasks_are_joined(self):
        ws = FakeSocket()
        ws.block_send = True
        output = asyncio.Queue()
        output.put_nowait(b"media")
        real_wait_for = asyncio.wait_for

        async def fast_timeout(awaitable, timeout):
            return await real_wait_for(awaitable, min(timeout, 0.02))

        proxy = SimpleNamespace(**vars(asyncio))
        proxy.wait_for = fast_timeout
        with patch.dict(NAMESPACE, asyncio=proxy):
            with self.assertRaises(asyncio.TimeoutError):
                await forward(ws, output, lambda item: item, send_timeout=5.0)
        self.assertTrue(ws.send_cancelled)
        self.assertFalse(output._getters)
        self.assertFalse(ws.incoming._getters)

    def test_unmeasured_send_deadline_is_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(NAMESPACE["_websocket_send_timeout"](), 0)
        for value in ("nan", "inf", "-1"):
            with patch.dict(os.environ, {"PIPE_WEBSOCKET_SEND_TIMEOUT_SEC": value}):
                with self.assertRaises(ValueError):
                    NAMESPACE["_websocket_send_timeout"]()

    async def test_logs_handler_unregisters_on_quiet_peer_close(self):
        ws = FakeSocket()
        registered, unregistered = [], []
        engine = SimpleNamespace(register_log_client=registered.append,
                                 unregister_log_client=unregistered.append)
        request = SimpleNamespace(app={"engine": engine})
        with patch.dict(NAMESPACE, web=SimpleNamespace(WebSocketResponse=lambda **kw: ws)):
            task = asyncio.create_task(NAMESPACE["logs_ws"](request))
            await asyncio.sleep(0)
            ws.disconnect()
            await asyncio.wait_for(task, 1)
        self.assertEqual(registered, unregistered)
        self.assertEqual(len(registered), 1)

    async def test_media_initial_send_failure_unregisters_encoder(self):
        ws = FakeSocket()

        async def broken_send(item):
            raise ConnectionResetError("peer closed during greeting")

        ws.send_str = broken_send
        client = object()
        unregistered = []
        engine = SimpleNamespace(register_media_client=lambda: client,
                                 unregister_media_client=unregistered.append)
        request = SimpleNamespace(app={"engine": engine})
        with patch.dict(NAMESPACE, web=SimpleNamespace(WebSocketResponse=lambda **kw: ws)):
            await NAMESPACE["media_ws"](request)
        self.assertEqual(unregistered, [client])

    async def test_encoded_put_timeout_cancels_pending_queue_write(self):
        engine = SimpleNamespace(loop=asyncio.get_running_loop(), log=lambda message: None)
        client = Client(engine, 2)
        client.out_q = asyncio.Queue(maxsize=1)
        client.out_q.put_nowait(b"existing")
        real_submit = asyncio.run_coroutine_threadsafe

        def short_submit(coro, loop):
            future = real_submit(coro, loop)
            return SimpleNamespace(result=lambda timeout: future.result(timeout=0.02),
                                   cancel=future.cancel)

        proxy = SimpleNamespace(**vars(asyncio))
        proxy.run_coroutine_threadsafe = short_submit
        with patch.dict(NAMESPACE, asyncio=proxy):
            await asyncio.to_thread(client._put_bytes, b"late bytes", 0)
        await asyncio.sleep(0)
        self.assertFalse(client.running.is_set())
        self.assertFalse(client.out_q._putters)
        self.assertEqual(client.out_q.get_nowait(), b"existing")
        await asyncio.sleep(0)
        self.assertTrue(client.out_q.empty())


if __name__ == "__main__":
    unittest.main()
