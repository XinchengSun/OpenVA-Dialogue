import argparse
import audioop
import asyncio
import hmac
import json
import math
import os
import queue
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Set

import aiohttp
from aiohttp import web
import numpy as np
from PIL import Image
import soundfile as sf
from dotenv import load_dotenv
from scipy.signal import resample_poly
import websockets

from customization_runtime import (
    CUSTOMIZATION_FORWARDED_HEADERS,
    MAX_REQUEST_BYTES,
    customization_workload_active as runtime_customization_workload_active,
    is_customization_path,
    register_customization_routes,
)
from pipecat_dystream.public_access import (
    ACCESS_COOKIE_NAME,
    RedactedAccessLogger,
    is_loopback_host,
    token_matches,
)

# Run this file from the repository root so local DyStream demo imports resolve.
import omni_avatar_interactive_v2 as omni

load_dotenv()

ENGINE_VERSION = "FLASHAV2AV_0.1.0"


# -------------------------
# Doubao / SeedDuplex binary protocol
# -------------------------

class DoubaoRealtimeProtocol:
    FULL_CLIENT_REQUEST = 0x1
    AUDIO_ONLY_REQUEST = 0x2
    FULL_SERVER_RESPONSE = 0x9
    AUDIO_ONLY_RESPONSE = 0xB
    ERROR_INFORMATION = 0xF

    FLAG_EVENT = 0x4
    FLAG_ERROR_CODE = 0xF

    SER_RAW = 0x0
    SER_JSON = 0x1
    COMP_NONE = 0x0

    EV_START_CONNECTION = 1
    EV_FINISH_CONNECTION = 2
    EV_START_SESSION = 100
    EV_FINISH_SESSION = 102
    EV_TASK_REQUEST = 200
    EV_END_ASR = 400
    EV_CLIENT_INTERRUPT = 515

    EV_CONNECTION_STARTED = 50
    EV_CONNECTION_FAILED = 51
    EV_CONNECTION_FINISHED = 52
    EV_SESSION_STARTED = 150
    EV_SESSION_FINISHED = 152
    EV_SESSION_FAILED = 153
    EV_TTS_SENTENCE_START = 350
    EV_TTS_SENTENCE_END = 351
    EV_TTS_RESPONSE = 352
    EV_TTS_ENDED = 359
    EV_ASR_INFO = 450
    EV_ASR_RESPONSE = 451
    EV_ASR_ENDED = 459
    EV_CHAT_RESPONSE = 550
    EV_CHAT_ENDED = 559
    EV_DIALOG_COMMON_ERROR = 599

    def __init__(self, log: Callable[[str], None]):
        self.ws_url = os.getenv("SEEDUPLEX_WS_URL", "wss://openspeech.bytedance.com/api/v3/realtime/dialogue")
        self.app_id = os.getenv("SEEDUPLEX_APP_ID", "")
        self.access_key = os.getenv("SEEDUPLEX_ACCESS_KEY", "")
        self.app_key = os.getenv("SEEDUPLEX_APP_KEY", "")
        self.resource_id = os.getenv("SEEDUPLEX_RESOURCE_ID", "volc.speech.dialog")
        self.speaker = os.getenv("SEEDUPLEX_SPEAKER", "zh_female_vv_jupiter_bigtts")
        self.model = os.getenv("SEEDUPLEX_MODEL", "1.2.1.1")
        self.log = log

        if not self.app_id or not self.access_key:
            raise RuntimeError("missing SEEDUPLEX_APP_ID or SEEDUPLEX_ACCESS_KEY in .env")

    def headers(self) -> Dict[str, str]:
        return {
            "X-Api-App-ID": self.app_id,
            "X-Api-App-Key": self.app_key,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    @staticmethod
    def make_header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
        return bytes([
            0x11,  # protocol v1, header size 4 bytes
            ((message_type & 0x0F) << 4) | (flags & 0x0F),
            ((serialization & 0x0F) << 4) | (compression & 0x0F),
            0x00,
        ])

    def pack_event(
        self,
        message_type: int,
        event_id: int,
        payload: bytes,
        session_id: Optional[str] = None,
        serialization: int = SER_JSON,
    ) -> bytes:
        parts = [
            self.make_header(message_type, self.FLAG_EVENT, serialization, self.COMP_NONE),
            struct.pack(">I", event_id),
        ]
        if session_id is not None:
            sid = session_id.encode("utf-8")
            parts.append(struct.pack(">I", len(sid)))
            parts.append(sid)
        parts.append(struct.pack(">I", len(payload)))
        parts.append(payload)
        return b"".join(parts)

    def pack_json_event(self, event_id: int, obj: Dict[str, Any], session_id: Optional[str] = None) -> bytes:
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.pack_event(self.FULL_CLIENT_REQUEST, event_id, body, session_id, self.SER_JSON)

    def pack_audio_event(self, audio_bytes: bytes, session_id: str) -> bytes:
        return self.pack_event(self.AUDIO_ONLY_REQUEST, self.EV_TASK_REQUEST, audio_bytes, session_id, self.SER_RAW)

    def parse_frame(self, data: bytes) -> Dict[str, Any]:
        if len(data) < 4:
            return {"event_id": None, "message_type": None, "payload": b"", "json": None, "text": "short frame"}

        b0, b1, b2, _ = data[:4]
        header_size = (b0 & 0x0F) * 4
        message_type = b1 >> 4
        flags = b1 & 0x0F
        serialization = b2 >> 4

        offset = header_size
        event_id = None
        code = None
        session = None

        if message_type == self.ERROR_INFORMATION:
            if len(data) >= offset + 4:
                code = struct.unpack(">I", data[offset:offset + 4])[0]
                offset += 4
        elif flags == self.FLAG_EVENT:
            if len(data) >= offset + 4:
                event_id = struct.unpack(">I", data[offset:offset + 4])[0]
                offset += 4
            if event_id not in (self.EV_CONNECTION_STARTED, self.EV_CONNECTION_FAILED, self.EV_CONNECTION_FINISHED):
                if len(data) >= offset + 4:
                    sid_len = struct.unpack(">I", data[offset:offset + 4])[0]
                    offset += 4
                    if sid_len > 0 and len(data) >= offset + sid_len:
                        session = data[offset:offset + sid_len].decode("utf-8", errors="replace")
                        offset += sid_len

        payload = b""
        if len(data) >= offset + 4:
            payload_size = struct.unpack(">I", data[offset:offset + 4])[0]
            offset += 4
            payload = data[offset:offset + payload_size]

        js = None
        text = None
        if serialization == self.SER_JSON and payload:
            try:
                text = payload.decode("utf-8")
                js = json.loads(text)
            except Exception:
                text = payload.decode("utf-8", errors="replace")

        return {
            "message_type": message_type,
            "flags": flags,
            "event_id": event_id,
            "code": code,
            "session_id": session,
            "payload": payload,
            "json": js,
            "text": text,
        }

    def start_session_payload(self, instruction: str) -> Dict[str, Any]:
        payload = {
            "asr": {
                "audio_info": {"format": "pcm", "sample_rate": 16000, "channel": 1},
                "extra": {
                    "end_smooth_window_ms": int(os.getenv("SEEDUPLEX_END_SMOOTH_MS", "1500")),
                    "enable_custom_vad": False,
                    "enable_asr_twopass": False,
                },
            },
            "dialog": {
                "bot_name": "豆包",
                "system_role": instruction or "你是一个简洁自然的实时语音助手。",
                "speaking_style": "回答要完整、自然、口语化，默认一到三句；只有用户明确要求时才详细展开，不使用列表或序号。",
                "dialog_id": "",
                "extra": {
                    "strict_audit": True,
                    "model": self.model,
                },
            },
            "tts": {
                "speaker": self.speaker,
                "audio_config": {"channel": 1, "format": "pcm_s16le", "sample_rate": 24000},
                "extra": {},
            },
        }
        input_mod = os.getenv("SEEDUPLEX_INPUT_MOD", "").strip()
        if input_mod:
            payload["dialog"]["extra"]["input_mod"] = input_mod
        return payload

    @staticmethod
    def decode_tts_pcm24k_to_float16k(payload: bytes) -> np.ndarray:
        if not payload:
            return np.zeros(0, dtype=np.float32)
        x24 = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        if len(x24) == 0:
            return np.zeros(0, dtype=np.float32)
        return resample_poly(x24, 2, 3).astype(np.float32)

    async def run_utterance(
        self,
        pcm16_queue: asyncio.Queue,
        instruction: str,
        on_tts_audio_16k: Callable[[np.ndarray], None],
    ):
        """Stream one user utterance to Doubao and enqueue returned TTS audio chunks."""
        session_id = str(uuid.uuid4())
        session_started = asyncio.Event()
        tts_ended = asyncio.Event()

        self.log("[DOUBAO] connecting websocket")
        async with websockets.connect(
            self.ws_url,
            additional_headers=self.headers(),
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as ws:
            self.log("[DOUBAO] websocket connected")

            async def recv_loop():
                while True:
                    msg = await ws.recv()
                    if not isinstance(msg, bytes):
                        self.log(f"[DOUBAO] text frame: {msg[:200]}")
                        continue
                    fr = self.parse_frame(msg)
                    ev = fr.get("event_id")
                    mt = fr.get("message_type")
                    payload = fr.get("payload") or b""
                    js = fr.get("json")
                    self.log(f"[DOUBAO] recv event={ev} mt={mt} payload_bytes={len(payload)}")

                    if mt == self.ERROR_INFORMATION or ev in (self.EV_CONNECTION_FAILED, self.EV_SESSION_FAILED, self.EV_DIALOG_COMMON_ERROR):
                        self.log(f"[DOUBAO ERROR] {fr.get('text') or js}")
                        tts_ended.set()
                        return

                    if ev == self.EV_SESSION_STARTED:
                        session_started.set()

                    if ev == self.EV_ASR_RESPONSE and isinstance(js, dict):
                        self.log(f"[DOUBAO ASR] {js}")

                    if ev == self.EV_CHAT_RESPONSE and isinstance(js, dict):
                        content = js.get("content", "")
                        if content:
                            self.log(f"[DOUBAO TEXT] {content}")

                    if ev == self.EV_TTS_RESPONSE and payload:
                        x16 = self.decode_tts_pcm24k_to_float16k(payload)
                        if len(x16) > 0:
                            on_tts_audio_16k(x16)

                    if ev == self.EV_TTS_ENDED:
                        self.log("[DOUBAO] TTS ended")
                        tts_ended.set()
                        return

            recv_task = asyncio.create_task(recv_loop())

            await ws.send(self.pack_json_event(self.EV_START_CONNECTION, {}))
            await ws.send(self.pack_json_event(self.EV_START_SESSION, self.start_session_payload(instruction), session_id))
            self.log("[DOUBAO] StartConnection + StartSession sent")

            try:
                await asyncio.wait_for(session_started.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                self.log("[DOUBAO] SessionStarted timeout; continue")

            sent = 0
            while True:
                chunk = await pcm16_queue.get()
                if chunk is None:
                    break
                if chunk:
                    await ws.send(self.pack_audio_event(chunk, session_id))
                    sent += len(chunk)

            self.log(f"[DOUBAO] sent user audio bytes={sent}; sending EndASR")
            await ws.send(self.pack_json_event(self.EV_END_ASR, {}, session_id))

            try:
                await asyncio.wait_for(tts_ended.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                self.log("[DOUBAO ERROR] timeout waiting TTSEnded")

            try:
                await ws.send(self.pack_json_event(self.EV_FINISH_SESSION, {}, session_id))
                await ws.send(self.pack_json_event(self.EV_FINISH_CONNECTION, {}))
            except Exception:
                pass
            recv_task.cancel()


class DoubaoLiveSession:
    """
    One long-lived SeedDuplex session per browser microphone websocket.

    This replaces the old front-end VAD + one-Doubao-session-per-utterance flow.
    Browser audio is streamed continuously; Doubao server-side VAD decides turns.
    """

    def __init__(
        self,
        proto: DoubaoRealtimeProtocol,
        instruction: str,
        on_tts_start: Callable[[], None],
        on_tts_audio_16k: Callable[[np.ndarray], None],
        on_tts_end: Callable[[], None],
        log: Callable[[str], None],
    ):
        self.proto = proto
        self.instruction = instruction
        self.on_tts_start = on_tts_start
        self.on_tts_audio_16k = on_tts_audio_16k
        self.on_tts_end = on_tts_end
        self.log = log
        self.session_id = str(uuid.uuid4())
        self.audio_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=2048)
        self.stop_event = asyncio.Event()
        self.connected = asyncio.Event()
        self.closed = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.ws = None
        self.sent_audio_bytes = 0
        self._ratecv_state = None
        self._tts_active = False

    def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    async def send_audio(self, pcm16: bytes) -> bool:
        if not pcm16:
            return True
        if self.closed.is_set() or (self.task is not None and self.task.done()):
            self.log("[DOUBAO LIVE WARN] send_audio on closed session; need restart")
            return False
        try:
            self.audio_q.put_nowait(pcm16)
            return True
        except asyncio.QueueFull:
            self.log("[DOUBAO LIVE WARN] mic queue full; drop audio chunk")
            return True

    async def interrupt(self):
        if self.ws is not None:
            try:
                await self.ws.send(self.proto.pack_json_event(self.proto.EV_CLIENT_INTERRUPT, {}, self.session_id))
                self.log("[DOUBAO LIVE] ClientInterrupt sent")
            except Exception as e:
                self.log(f"[DOUBAO LIVE WARN] interrupt failed: {repr(e)}")

    async def close(self):
        self.stop_event.set()
        try:
            self.audio_q.put_nowait(None)
        except Exception:
            pass
        if self.task is not None:
            try:
                await asyncio.wait_for(self.closed.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self.task.cancel()

    async def _run(self):
        self.log("[DOUBAO LIVE] connecting websocket")
        try:
            async with websockets.connect(
                self.proto.ws_url,
                additional_headers=self.proto.headers(),
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as ws:
                self.ws = ws
                self.log("[DOUBAO LIVE] websocket connected")
                await ws.send(self.proto.pack_json_event(self.proto.EV_START_CONNECTION, {}))
                await ws.send(self.proto.pack_json_event(
                    self.proto.EV_START_SESSION,
                    self.proto.start_session_payload(self.instruction),
                    self.session_id,
                ))
                self.log("[DOUBAO LIVE] StartConnection + StartSession sent")

                recv_task = asyncio.create_task(self._recv_loop(ws))
                send_task = asyncio.create_task(self._send_loop(ws))
                done, pending = await asyncio.wait([recv_task, send_task], return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()

                try:
                    await ws.send(self.proto.pack_json_event(self.proto.EV_FINISH_SESSION, {}, self.session_id))
                    await ws.send(self.proto.pack_json_event(self.proto.EV_FINISH_CONNECTION, {}))
                except Exception:
                    pass
        except Exception as e:
            self.log(f"[DOUBAO LIVE ERROR] {repr(e)}")
        finally:
            self.ws = None
            self.closed.set()
            self.log(f"[DOUBAO LIVE] closed sent_audio_bytes={self.sent_audio_bytes}")

    async def _send_loop(self, ws):
        while not self.stop_event.is_set():
            chunk = await self.audio_q.get()
            if chunk is None:
                break
            if not chunk:
                continue
            await ws.send(self.proto.pack_audio_event(chunk, self.session_id))
            self.sent_audio_bytes += len(chunk)

    def _decode_tts_payload_stateful(self, payload: bytes) -> np.ndarray:
        """Convert Doubao 24k pcm_s16le TTS bytes to continuous 16k float32.

        This keeps resampler state across TTSResponse chunks, avoiding small
        discontinuities from resampling every websocket payload independently.
        """
        if not payload:
            return np.zeros(0, dtype=np.float32)
        try:
            out, self._ratecv_state = audioop.ratecv(payload, 2, 1, 24000, 16000, self._ratecv_state)
            if not out:
                return np.zeros(0, dtype=np.float32)
            return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0
        except Exception as e:
            self.log(f"[DOUBAO LIVE WARN] stateful resample failed; fallback resample_poly: {repr(e)}")
            return self.proto.decode_tts_pcm24k_to_float16k(payload)

    async def _recv_loop(self, ws):
        while not self.stop_event.is_set():
            msg = await ws.recv()
            if not isinstance(msg, bytes):
                self.log(f"[DOUBAO LIVE] text frame: {str(msg)[:200]}")
                continue
            fr = self.proto.parse_frame(msg)
            ev = fr.get("event_id")
            mt = fr.get("message_type")
            payload = fr.get("payload") or b""
            js = fr.get("json")

            if mt == self.proto.ERROR_INFORMATION or ev in (self.proto.EV_CONNECTION_FAILED, self.proto.EV_SESSION_FAILED, self.proto.EV_DIALOG_COMMON_ERROR):
                self.log(f"[DOUBAO LIVE ERROR] {fr.get('text') or js}")
                return
            if ev == self.proto.EV_SESSION_STARTED:
                self.connected.set()
                self.log("[DOUBAO LIVE] SessionStarted")
            elif ev == self.proto.EV_ASR_INFO:
                self.log("[DOUBAO ASR_INFO] speech detected")
            elif ev == self.proto.EV_ASR_RESPONSE and isinstance(js, dict):
                text = ""
                for r in (js.get("results") or []):
                    text = (r.get("text") or text or "").strip()
                if text:
                    self.log(f"[DOUBAO ASR] {text}")
            elif ev == self.proto.EV_ASR_ENDED:
                self.log("[DOUBAO ASR] ended")
            elif ev == self.proto.EV_CHAT_RESPONSE and isinstance(js, dict):
                content = js.get("content", "")
                if content:
                    self.log(f"[DOUBAO TEXT] {content}")
            elif ev == self.proto.EV_CHAT_ENDED:
                self.log("[DOUBAO TEXT] ended")
            elif ev == self.proto.EV_TTS_SENTENCE_START:
                self._ratecv_state = None
                if not self._tts_active:
                    self._tts_active = True
                    self.on_tts_start()
                self.log("[DOUBAO TTS] sentence start")
            elif ev == self.proto.EV_TTS_RESPONSE and payload:
                if not self._tts_active:
                    self._tts_active = True
                    self.on_tts_start()
                x16 = self._decode_tts_payload_stateful(payload)
                if len(x16) > 0:
                    self.log(f"[DOUBAO TTS AUDIO] payload_bytes={len(payload)} x16_samples={len(x16)}")
                    self.on_tts_audio_16k(x16)
            elif ev == self.proto.EV_TTS_ENDED:
                if self._tts_active:
                    self._tts_active = False
                    self.on_tts_end()
                self.log("[DOUBAO TTS] ended")


# -------------------------
# DyStream -> fMP4/MSE runtime
# -------------------------

class MediaPipeClient:
    """Per-browser continuous fMP4 encoder with separate video/audio writer threads.

    The previous pipe build wrote a whole video block to ffmpeg stdin and only
    then wrote the corresponding audio block. With 512x512 RGB frames, even one
    video frame is 786432 bytes. If that pipe blocks, audio never reaches
    ffmpeg, so ffmpeg cannot mux and stdout never produces MSE data.

    This version keeps the same long-lived ffmpeg process, but splits media into
    frame-sized A/V units and feeds video and audio through separate writer
    threads. That prevents video pipe backpressure from starving the audio pipe.
    """

    def __init__(self, engine: "RealtimeMSEEngine", client_id: int):
        self.engine = engine
        self.client_id = client_id
        self.loop = engine.loop
        self.segment_q: queue.Queue = queue.Queue(maxsize=int(os.getenv("PIPE_SEGMENT_Q", "2")))
        self.av_pair_q: queue.Queue = queue.Queue(maxsize=int(os.getenv("PIPE_AV_Q", "8")))
        # The dispatcher is the only producer for these single-job queues. It
        # waits for both acknowledgements before dispatching the next A/V unit,
        # so video and audio can never advance independently across a generation
        # fence.
        self.video_job_q: queue.Queue = queue.Queue(maxsize=1)
        self.audio_job_q: queue.Queue = queue.Queue(maxsize=1)
        self.out_q: asyncio.Queue = asyncio.Queue(
            maxsize=max(1, int(os.getenv("PIPE_OUT_Q", "16")))
        )
        self.stopped = asyncio.Event()
        self.stop_reason: Optional[str] = None
        self._lifecycle_lock = threading.Lock()
        self.running = threading.Event()
        self.running.set()
        self.thread = threading.Thread(target=self._thread_main, name=f"mse-pipe-client-{client_id}", daemon=True)
        self.proc: Optional[subprocess.Popen] = None
        self.audio_w_fd: Optional[int] = None
        self.width = None
        self.height = None
        self.enqueued_frames = 0
        self.video_written_frames = 0
        self.audio_written_frames = 0
        self.written_audio_samples = 0
        self.stdout_chunks = 0
        self._generation_lock = threading.Lock()
        self.accepted_generation = 0
        self._next_av_unit_id = 0
        self._consecutive_pair_full = 0
        self._pair_full_fail_count = max(
            1,
            int(os.getenv("PIPE_AV_FULL_FAIL_COUNT", "8")),
        )
        self._epoch_media_units = 0
        self._generation_start_units: Dict[int, int] = {}
        self._assistant_boundaries_scheduled: Set[tuple[str, int, int]] = set()
        self._active_assistant_media_key: Optional[tuple[int, int]] = None
        self._control_event_seq = 0
        self._claimed_av_unit = None
        self._io_stop = threading.Event()
        self._writer_threads_started = False
        self._io_threads = []
        self._reset_requested = threading.Event()
        self._reset_in_progress = threading.Event()
        self._reset_done = threading.Event()
        self._reset_done.set()
        self._stream_epoch = 0
        # Front-end preview is downsampled from DyStream 25fps.
        # stride=2 means 12.5fps output and 1280 audio samples per video frame.
        self.frame_stride = max(1, int(os.getenv("PIPE_FRAME_STRIDE", "2")))
        self.output_fps = 25.0 / float(self.frame_stride)
        self.samples_per_out_frame = 640 * self.frame_stride

    def start(self):
        self.thread.start()

    def _signal_stopped(self, reason: str):
        self.running.clear()
        with self._lifecycle_lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        try:
            self.loop.call_soon_threadsafe(self.stopped.set)
        except RuntimeError:
            pass

    def stop(self):
        self._signal_stopped("media client stopped")
        try:
            self.segment_q.put_nowait(None)
        except Exception:
            pass
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            self._stop_ffmpeg()

    def _put_text_nowait(self, text: str, critical: bool):
        if not self.running.is_set():
            return
        # Logs are best-effort. Keep capacity available for encoded media and
        # typed turn boundaries so a log burst cannot kill a healthy stream.
        if (
            not critical
            and self.out_q.qsize() >= max(1, self.out_q.maxsize // 2)
        ):
            return
        try:
            self.out_q.put_nowait(text)
        except asyncio.QueueFull:
            if critical:
                self.engine.log(
                    f"[PIPE WARN] client={self.client_id} critical media control queue blocked"
                )
                self._signal_stopped("critical media control queue blocked")

    def put_text(self, text: str, critical: bool = False):
        if not self.running.is_set():
            return
        try:
            self.loop.call_soon_threadsafe(
                self._put_text_nowait,
                text,
                critical,
            )
        except RuntimeError:
            if critical:
                self._signal_stopped("media event loop stopped")

    @staticmethod
    def _filter_queue(q: queue.Queue, keep: Callable[[Any], bool]) -> int:
        kept = []
        dropped = 0
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if keep(item):
                kept.append(item)
            else:
                dropped += 1
        for item in kept:
            q.put_nowait(item)
        return dropped

    def advance_generation(self, generation: int) -> int:
        """Fence unclaimed raw input without touching encoded fMP4 output.

        A pair already claimed by the dispatcher is intentionally allowed to
        finish on both ffmpeg inputs. Every later generation check happens at
        an A/V-pair boundary, so the media clocks remain aligned.
        """
        generation = int(generation)
        with self._generation_lock:
            if generation <= self.accepted_generation:
                return 0
            self.accepted_generation = generation
            dropped_segments = self._filter_queue(
                self.segment_q,
                lambda item: item is None or int(item[0]) >= generation,
            )
            dropped_pairs = self._filter_queue(
                self.av_pair_q,
                lambda item: item is None or int(item[1]) >= generation,
            )
            self._assistant_boundaries_scheduled = {
                key
                for key in self._assistant_boundaries_scheduled
                if key[1] >= generation
            }
            if (
                self._active_assistant_media_key is not None
                and self._active_assistant_media_key[0] < generation
            ):
                self._active_assistant_media_key = None
        dropped = dropped_segments + dropped_pairs
        self.engine.log(
            f"[PIPE FENCE] client={self.client_id} generation={generation} "
            f"dropped_segments={dropped_segments} dropped_pairs={dropped_pairs} "
            f"claimed={int(self._claimed_av_unit is not None)}"
        )
        return dropped

    def media_clock_snapshot(self) -> Dict[str, Any]:
        """Return the continuous clock for typed browser control messages."""
        with self._generation_lock:
            return {
                "stream_epoch": self._stream_epoch,
                "accepted_generation": self.accepted_generation,
                "epoch_media_units": self._epoch_media_units,
                "output_fps": self.output_fps,
                "samples_per_media_unit": self.samples_per_out_frame,
                "sample_rate": 16000,
            }

    def generation_boundary_snapshot(
        self,
        generation: int,
        guard_units: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Describe a generation boundary on the current ffmpeg epoch clock."""
        generation = int(generation)
        guard_units = max(0, int(guard_units))
        with self._generation_lock:
            start_unit = self._generation_start_units.get(generation)
            if start_unit is None:
                return None
            safe_unit = start_unit + guard_units
            return {
                "stream_epoch": self._stream_epoch,
                "generation": generation,
                "start_media_unit": start_unit,
                "safe_media_unit": safe_unit,
                "output_fps": self.output_fps,
                "samples_per_media_unit": self.samples_per_out_frame,
                "safe_media_time_s": (
                    safe_unit * self.samples_per_out_frame / 16000.0
                ),
            }

    def push_segment(
        self,
        frames: np.ndarray,
        audio: np.ndarray,
        generation: Optional[int] = None,
        audio_frame_meta: Optional[list[Dict[str, Any]]] = None,
    ):
        if not self.running.is_set():
            return False
        with self._generation_lock:
            if generation is None:
                generation = self.accepted_generation
            generation = int(generation)
            if generation != self.accepted_generation:
                return False
            item = (
                generation,
                frames.copy(),
                audio.copy(),
                list(audio_frame_meta or ()),
            )
            try:
                self.segment_q.put_nowait(item)
                return True
            except queue.Full:
                # Live stream: keep newest raw segment. Old segments are stale.
                try:
                    self.segment_q.get_nowait()
                except Exception:
                    pass
            try:
                self.segment_q.put_nowait(item)
                self.engine.log(f"[PIPE WARN] client={self.client_id} raw segment queue full; dropped oldest")
                return True
            except Exception:
                return False

    def clear_backlog(self):
        """Drop stale encoded-input queues for live response priority."""
        with self._generation_lock:
            dropped_segments = self._filter_queue(
                self.segment_q,
                lambda item: item is None,
            )
            dropped_pairs = self._filter_queue(
                self.av_pair_q,
                lambda item: item is None,
            )
        dropped = dropped_segments + dropped_pairs
        if dropped:
            self.engine.log(
                f"[PIPE LIVE] client={self.client_id} cleared stale pipe queues "
                f"segments={dropped_segments} pairs={dropped_pairs}"
            )
        return dropped

    def request_stream_reset(self):
        dropped = self.clear_backlog()
        if not self.running.is_set():
            return dropped
        with self._generation_lock:
            if (
                not self._reset_requested.is_set()
                and not self._reset_in_progress.is_set()
            ):
                self._reset_done.clear()
                self._reset_requested.set()
        if not self._reset_done.wait(timeout=2.0):
            self.engine.log(f"[PIPE WARN] client={self.client_id} encoder reset timed out")
        return dropped

    def _request_encoder_recovery(
        self,
        reason: str,
        io_stop: threading.Event,
    ) -> bool:
        """Ask the owner thread to rebuild a failed encoder without blocking I/O."""
        with self._generation_lock:
            if (
                not self.running.is_set()
                or io_stop.is_set()
                or self._reset_requested.is_set()
                or self._reset_in_progress.is_set()
            ):
                return False
            self._reset_done.clear()
            self._reset_requested.set()
        self.engine.log(
            f"[PIPE RECOVERY] client={self.client_id} requested reason={reason}"
        )
        return True

    async def _reset_output_queue(self, epoch: int):
        while True:
            try:
                self.out_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self.out_q.put(json.dumps({"type": "stream_reset", "epoch": epoch}))

    def _notify_stream_reset(self):
        epoch = self._stream_epoch
        fut = asyncio.run_coroutine_threadsafe(
            self._reset_output_queue(epoch),
            self.loop,
        )
        try:
            fut.result(timeout=2.0)
        except Exception as e:
            self.engine.log(f"[PIPE WARN] client={self.client_id} stream reset notify failed: {repr(e)}")

    def _put_bytes(self, data: bytes, epoch: int):
        if not data:
            return
        # Tag encoded bytes with their ffmpeg epoch. A blocked put from the old
        # process may complete after a stream reset; the websocket consumer can
        # then discard it instead of feeding old fMP4 bytes into the new MSE.
        item = ("media", epoch, data)
        fut = asyncio.run_coroutine_threadsafe(
            self._put_output_nowait(item),
            self.loop,
        )
        try:
            fut.result(timeout=1.0)
        except Exception:
            fut.cancel()
            self.engine.log(f"[PIPE WARN] client={self.client_id} websocket output queue blocked")
            self._signal_stopped("websocket output queue blocked")

    async def _put_output_nowait(self, item):
        self.out_q.put_nowait(item)

    def filter_output_item(self, item):
        if (
            isinstance(item, tuple)
            and len(item) == 3
            and item[0] == "media"
        ):
            _, epoch, data = item
            if epoch != self._stream_epoch:
                return None
            return data
        return item

    def _start_ffmpeg(self, w: int, h: int):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")

        audio_r, audio_w = os.pipe()
        child_audio_fd = audio_r

        gop = max(2, int(os.getenv("PIPE_GOP", str(max(2, int(round(self.output_fps * 0.5)))))))
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-thread_queue_size", os.getenv("FFMPEG_THREAD_QUEUE_SIZE", "16"),
            "-probesize", "32", "-analyzeduration", "0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{self.output_fps:g}", "-i", "pipe:0",
            "-thread_queue_size", os.getenv("FFMPEG_THREAD_QUEUE_SIZE", "16"),
            "-probesize", "32", "-analyzeduration", "0",
            "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", f"pipe:{child_audio_fd}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", os.getenv("FFMPEG_VIDEO_ENCODER", "libx264"),
            "-preset", os.getenv("FFMPEG_X264_PRESET", "ultrafast"),
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.1",
            "-b:v", os.getenv("FFMPEG_VIDEO_BITRATE", "1800k"),
            "-maxrate", os.getenv("FFMPEG_VIDEO_BITRATE", "1800k"),
            "-bufsize", os.getenv("FFMPEG_VIDEO_BUFSIZE", "3600k"),
            "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
            "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "64k",
            "-f", "mp4",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset+separate_moof",
            "-max_interleave_delta", "0",
            "-muxdelay", "0", "-muxpreload", "0", "-flush_packets", "1",
            "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(child_audio_fd,),
            bufsize=0,
        )
        self.proc = proc
        try:
            os.close(child_audio_fd)
        except Exception:
            pass
        self.audio_w_fd = audio_w
        epoch = self._stream_epoch
        av_pair_q = self.av_pair_q
        video_job_q = self.video_job_q
        audio_job_q = self.audio_job_q
        io_stop = self._io_stop
        with self._generation_lock:
            self._epoch_media_units = 0
            self._generation_start_units.clear()
            self._assistant_boundaries_scheduled.clear()
            self._active_assistant_media_key = None
            self._next_av_unit_id = 0
            self._claimed_av_unit = None
        self._io_threads = [
            threading.Thread(
                target=self._stdout_reader,
                args=(proc, epoch, io_stop),
                name=f"mse-stdout-{self.client_id}",
                daemon=True,
            ),
            threading.Thread(
                target=self._stderr_reader,
                args=(proc,),
                name=f"mse-stderr-{self.client_id}",
                daemon=True,
            ),
            threading.Thread(
                target=self._video_writer,
                args=(proc, video_job_q, io_stop),
                name=f"mse-video-writer-{self.client_id}",
                daemon=True,
            ),
            threading.Thread(
                target=self._audio_writer,
                args=(audio_w, audio_job_q, io_stop),
                name=f"mse-audio-writer-{self.client_id}",
                daemon=True,
            ),
            threading.Thread(
                target=self._av_dispatcher,
                args=(
                    av_pair_q,
                    video_job_q,
                    audio_job_q,
                    io_stop,
                ),
                name=f"mse-av-dispatcher-{self.client_id}",
                daemon=True,
            ),
        ]
        for thread in self._io_threads:
            thread.start()
        self._writer_threads_started = True
        encoder_name = os.getenv("FFMPEG_VIDEO_ENCODER", "libx264")
        self.engine.log(f"[PIPE] client={self.client_id} ffmpeg started {w}x{h} fps={self.output_fps:g} stride={self.frame_stride} gop={gop} encoder={encoder_name} dual_writer=1")

    def _stdout_reader(
        self,
        proc: subprocess.Popen,
        epoch: int,
        io_stop: threading.Event,
    ):
        assert proc.stdout is not None
        while self.running.is_set():
            try:
                data = proc.stdout.read(65536)
            except Exception as e:
                self.engine.log(f"[PIPE ERROR] client={self.client_id} stdout read failed: {repr(e)}")
                break
            if not data:
                break
            if epoch != self._stream_epoch:
                continue
            self.stdout_chunks += 1
            if self.stdout_chunks <= 5 or self.stdout_chunks % 20 == 0:
                self.engine.log(f"[PIPE-OUT] client={self.client_id} stdout chunk={self.stdout_chunks} bytes={len(data)}")
            self._put_bytes(data, epoch)
        self.engine.log(f"[PIPE] client={self.client_id} stdout reader ended")
        if epoch == self._stream_epoch:
            self._request_encoder_recovery(
                "unexpected stdout EOF",
                io_stop,
            )

    def _stderr_reader(self, proc: subprocess.Popen):
        assert proc.stderr is not None
        while self.running.is_set():
            try:
                line = proc.stderr.readline()
            except Exception:
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self.engine.log(f"[FFMPEG {self.client_id}] {text}")

    @staticmethod
    def _float_to_s16le(audio: np.ndarray) -> bytes:
        x = np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1.0, 1.0)
        return (x * 32767.0).astype("<i2").tobytes()

    @staticmethod
    def _write_all_fd(fd: int, data: bytes):
        view = memoryview(data)
        while len(view) > 0:
            n = os.write(fd, view)
            view = view[n:]

    def _video_writer(
        self,
        proc: subprocess.Popen,
        video_job_q: queue.Queue,
        io_stop: threading.Event,
    ):
        while self.running.is_set() and not io_stop.is_set():
            try:
                job = video_job_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            if proc.stdin is None:
                job["error"] = RuntimeError("ffmpeg video stdin is unavailable")
                job["done"].set()
                break
            try:
                proc.stdin.write(job["data"])
                # stdin is unbuffered (bufsize=0), flush is cheap but keeps behavior explicit.
                proc.stdin.flush()
                self.video_written_frames += 1
                if self.video_written_frames <= 3 or self.video_written_frames % 50 == 0:
                    self.engine.log(
                        f"[PIPE-VIDEO] client={self.client_id} wrote_frame={self.video_written_frames} "
                        f"unit={job['unit_id']} pairq={self.av_pair_q.qsize()}"
                    )
            except Exception as e:
                job["error"] = e
                is_normal_close = (not self.running.is_set()) or (isinstance(e, ValueError) and "closed file" in str(e)) or isinstance(e, BrokenPipeError)
                level = "PIPE" if is_normal_close else "PIPE ERROR"
                self.engine.log(f"[{level}] client={self.client_id} video writer closed: {repr(e)}")
            finally:
                job["done"].set()
            if job["error"] is not None:
                break
        self.engine.log(f"[PIPE] client={self.client_id} video writer ended")

    def _audio_writer(
        self,
        audio_w_fd: int,
        audio_job_q: queue.Queue,
        io_stop: threading.Event,
    ):
        while self.running.is_set() and not io_stop.is_set():
            try:
                job = audio_job_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                self._write_all_fd(audio_w_fd, job["data"])
                self.audio_written_frames += 1
                self.written_audio_samples += self.samples_per_out_frame
                if self.audio_written_frames <= 3 or self.audio_written_frames % 50 == 0:
                    self.engine.log(
                        f"[PIPE-AUDIO] client={self.client_id} wrote_frame={self.audio_written_frames} "
                        f"unit={job['unit_id']} audio_sec={self.written_audio_samples/16000:.2f} "
                        f"pairq={self.av_pair_q.qsize()}"
                    )
            except Exception as e:
                job["error"] = e
                is_normal_close = (not self.running.is_set()) or (isinstance(e, ValueError) and "closed file" in str(e)) or isinstance(e, BrokenPipeError)
                level = "PIPE" if is_normal_close else "PIPE ERROR"
                self.engine.log(f"[{level}] client={self.client_id} audio writer closed: {repr(e)}")
            finally:
                job["done"].set()
            if job["error"] is not None:
                break
        self.engine.log(f"[PIPE] client={self.client_id} audio writer ended")

    @staticmethod
    def _new_writer_job(unit_id: int, data: bytes) -> Dict[str, Any]:
        return {
            "unit_id": unit_id,
            "data": data,
            "done": threading.Event(),
            "error": None,
        }

    @staticmethod
    def _wait_for_writer_jobs(
        video_job: Dict[str, Any],
        audio_job: Dict[str, Any],
        io_stop: threading.Event,
    ) -> bool:
        while True:
            video_done = video_job["done"].is_set()
            audio_done = audio_job["done"].is_set()
            if video_done and audio_done:
                return True
            if io_stop.wait(timeout=0.01):
                return (
                    video_job["done"].is_set()
                    and audio_job["done"].is_set()
                )

    def _av_dispatcher(
        self,
        av_pair_q: queue.Queue,
        video_job_q: queue.Queue,
        audio_job_q: queue.Queue,
        io_stop: threading.Event,
    ):
        while self.running.is_set() and not io_stop.is_set():
            try:
                pair = av_pair_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if pair is None:
                break
            (
                pair_stream_epoch,
                generation,
                unit_id,
                frame_bytes,
                audio_bytes,
                boundaries,
            ) = pair
            with self._generation_lock:
                if generation != self.accepted_generation:
                    continue
                # Once claimed, generation advancement cannot split this pair.
                self._claimed_av_unit = (generation, unit_id)

            video_job = self._new_writer_job(unit_id, frame_bytes)
            audio_job = self._new_writer_job(unit_id, audio_bytes)
            try:
                video_job_q.put(video_job)
                audio_job_q.put(audio_job)
                completed = self._wait_for_writer_jobs(
                    video_job,
                    audio_job,
                    io_stop,
                )
                if not completed:
                    break
                if video_job["error"] is not None or audio_job["error"] is not None:
                    self.engine.log(
                        f"[PIPE ERROR] client={self.client_id} paired write failed "
                        f"unit={unit_id} video={video_job['error']!r} "
                        f"audio={audio_job['error']!r}"
                    )
                    self._request_encoder_recovery(
                        "paired writer failure",
                        io_stop,
                    )
                    io_stop.set()
                    break
                with self._generation_lock:
                    if pair_stream_epoch == self._stream_epoch:
                        unit_start = self._epoch_media_units
                        self._epoch_media_units += 1
                        self._generation_start_units.setdefault(
                            generation,
                            unit_start,
                        )
                    else:
                        unit_start = None
                    boundary_is_current = bool(
                        unit_start is not None
                        and generation == self.accepted_generation
                    )
                if boundaries and boundary_is_current:
                    for boundary in boundaries:
                        self._emit_assistant_media_boundary(
                            boundary,
                            unit_start,
                            pair_stream_epoch,
                        )
            finally:
                with self._generation_lock:
                    if self._claimed_av_unit == (generation, unit_id):
                        self._claimed_av_unit = None
        self.engine.log(f"[PIPE] client={self.client_id} AV dispatcher ended")

    def _enqueue_av_unit(
        self,
        generation: int,
        frame_bytes: bytes,
        audio_bytes: bytes,
        assistant_boundaries: Optional[list[Dict[str, Any]]] = None,
    ):
        if not self.running.is_set():
            return False
        with self._generation_lock:
            if generation != self.accepted_generation:
                return False
            if self.av_pair_q.full():
                self._consecutive_pair_full += 1
                if self._consecutive_pair_full in (
                    1,
                    self._pair_full_fail_count,
                ):
                    self.engine.log(
                        f"[PIPE WARN] client={self.client_id} AV pair queue full; "
                        "drop newest paired unit "
                        f"consecutive={self._consecutive_pair_full}/"
                        f"{self._pair_full_fail_count}"
                    )
                if self._consecutive_pair_full >= self._pair_full_fail_count:
                    self._signal_stopped("AV pair queue remained full")
                return False
            self._consecutive_pair_full = 0
            unit_id = self._next_av_unit_id
            self._next_av_unit_id += 1
            pair_stream_epoch = self._stream_epoch
            boundaries = []
            for boundary in assistant_boundaries or ():
                boundary_key = (
                    str(boundary.get("type", "assistant_media_boundary")),
                    generation,
                    int(boundary["turn_id"]),
                )
                if boundary_key in self._assistant_boundaries_scheduled:
                    continue
                self._assistant_boundaries_scheduled.add(boundary_key)
                boundaries.append(boundary)
            self.av_pair_q.put_nowait(
                (
                    pair_stream_epoch,
                    generation,
                    unit_id,
                    frame_bytes,
                    audio_bytes,
                    boundaries,
                )
            )
            return True

    def _emit_assistant_media_boundary(
        self,
        boundary: Dict[str, Any],
        unit_start: int,
        stream_epoch: int,
    ):
        source_frame_offset = max(
            0,
            int(boundary.get("source_frame_offset", 0)),
        )
        safe_media_time_s = (
            unit_start / self.output_fps
            + source_frame_offset / 25.0
        )
        with self._generation_lock:
            if (
                stream_epoch != self._stream_epoch
                or int(boundary["generation"]) != self.accepted_generation
            ):
                self.engine.log(
                    f"[PIPE BOUNDARY] client={self.client_id} stale suppressed "
                    f"epoch={stream_epoch}->{self._stream_epoch} "
                    f"generation={boundary['generation']}->{self.accepted_generation}"
                )
                return
            self._control_event_seq += 1
            event_seq = self._control_event_seq
        boundary_type = str(
            boundary.get("type", "assistant_media_boundary")
        )
        payload = {
            "type": boundary_type,
            "stream_epoch": stream_epoch,
            "generation": int(boundary["generation"]),
            "turn_id": int(boundary["turn_id"]),
            "start_media_unit": int(unit_start),
            "source_frame_offset": source_frame_offset,
            "output_fps": self.output_fps,
            "safe_media_time_s": safe_media_time_s,
            "event_seq": event_seq,
        }
        self.put_text(json.dumps(payload, ensure_ascii=False), critical=True)
        self.engine.log(
            f"[PIPE BOUNDARY] client={self.client_id} type={boundary_type} "
            f"epoch={stream_epoch} generation={payload['generation']} "
            f"turn={payload['turn_id']} unit={unit_start} "
            f"safe_media_time_s={safe_media_time_s:.3f}"
        )

    def _write_to_ffmpeg(
        self,
        generation: int,
        frames: np.ndarray,
        audio: np.ndarray,
        audio_frame_meta: Optional[list[Dict[str, Any]]] = None,
    ):
        if self.proc is None or self.proc.stdin is None or self.audio_w_fd is None:
            h, w = frames.shape[1], frames.shape[2]
            self._start_ffmpeg(w, h)
        if self.proc is None or self.proc.stdin is None or self.audio_w_fd is None:
            return
        if self.proc.poll() is not None:
            raise RuntimeError(f"ffmpeg exited code={self.proc.returncode}")

        frames = np.ascontiguousarray(frames.astype(np.uint8))
        audio = np.asarray(audio, dtype=np.float32)
        n_frames = int(frames.shape[0])

        # DyStream produces 25fps. Each source frame corresponds to 640 samples.
        src_expected = n_frames * 640
        if len(audio) < src_expected:
            audio = np.pad(audio, (0, src_expected - len(audio)))
        elif len(audio) > src_expected:
            audio = audio[:src_expected]
        frame_meta = list(audio_frame_meta or ())
        if len(frame_meta) < n_frames:
            frame_meta.extend({} for _ in range(n_frames - len(frame_meta)))
        elif len(frame_meta) > n_frames:
            frame_meta = frame_meta[:n_frames]

        enq = 0
        for i in range(0, n_frames, self.frame_stride):
            frame_bytes = np.ascontiguousarray(frames[i]).tobytes()
            a = audio[i * 640:i * 640 + self.samples_per_out_frame]
            if len(a) < self.samples_per_out_frame:
                a = np.pad(a, (0, self.samples_per_out_frame - len(a)))
            audio_bytes = self._float_to_s16le(a)
            with self._generation_lock:
                active_media_key = self._active_assistant_media_key
            next_active_media_key = active_media_key
            assistant_boundaries = []
            for source_offset, meta in enumerate(
                frame_meta[i:i + self.frame_stride]
            ):
                is_assistant = str(meta.get("mode", "")).startswith(
                    "ASSISTANT"
                )
                media_key = None
                if is_assistant:
                    media_key = (
                        int(meta.get("generation", generation)),
                        int(meta.get("turn_id", -1)),
                    )
                if media_key == next_active_media_key:
                    continue
                if next_active_media_key is not None:
                    assistant_boundaries.append({
                        "type": "assistant_media_ended",
                        "generation": next_active_media_key[0],
                        "turn_id": next_active_media_key[1],
                        "source_frame_offset": source_offset,
                    })
                if media_key is not None:
                    assistant_boundaries.append({
                        "type": "assistant_media_boundary",
                        "generation": media_key[0],
                        "turn_id": media_key[1],
                        "source_frame_offset": source_offset,
                    })
                next_active_media_key = media_key
            accepted = self._enqueue_av_unit(
                generation,
                frame_bytes,
                audio_bytes,
                assistant_boundaries,
            )
            if accepted:
                with self._generation_lock:
                    if generation == self.accepted_generation:
                        self._active_assistant_media_key = next_active_media_key
                enq += 1

        self.enqueued_frames += enq
        if self.enqueued_frames <= 3 or self.enqueued_frames % 25 == 0:
            self.engine.log(
                f"[PIPE-ENQ] client={self.client_id} enqueued_frames={self.enqueued_frames} "
                f"fps={self.output_fps:g} stride={self.frame_stride} "
                f"pairq={self.av_pair_q.qsize()} rawq={self.segment_q.qsize()}"
            )

    def _run(self):
        while self.running.is_set():
            reset_claimed = False
            with self._generation_lock:
                if (
                    self._reset_requested.is_set()
                    and not self._reset_in_progress.is_set()
                ):
                    self._reset_requested.clear()
                    self._reset_in_progress.set()
                    self._reset_done.clear()
                    self._stream_epoch += 1
                    self._epoch_media_units = 0
                    self._generation_start_units.clear()
                    self._assistant_boundaries_scheduled.clear()
                    self._active_assistant_media_key = None
                    self._next_av_unit_id = 0
                    reset_claimed = True
            if reset_claimed:
                try:
                    # Tell the browser to build its hidden replacement MSE
                    # before waiting for the old encoder to terminate. Bytes
                    # from the old process are already fenced by stream epoch.
                    self._notify_stream_reset()
                    current_turn = getattr(
                        self.engine,
                        "assistant_turn_control_snapshot",
                        lambda: None,
                    )()
                    if current_turn is not None:
                        # stream_reset clears the browser's old epoch gate.
                        # Re-arm the in-flight turn on the replacement epoch.
                        self.put_text(
                            json.dumps(current_turn, ensure_ascii=False),
                            critical=True,
                        )
                    self._stop_ffmpeg()
                    self.clear_backlog()
                    self.engine.log(
                        f"[PIPE LIVE] client={self.client_id} "
                        f"encoder reset epoch={self._stream_epoch}"
                    )
                except Exception as e:
                    self.engine.log(
                        f"[PIPE ERROR] client={self.client_id} "
                        f"encoder recovery failed: {repr(e)}"
                    )
                    self._stop_ffmpeg()
                finally:
                    with self._generation_lock:
                        self._reset_in_progress.clear()
                        self._reset_done.set()
            try:
                item = self.segment_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            generation, frames, audio, audio_frame_meta = item
            with self._generation_lock:
                if generation != self.accepted_generation:
                    continue
            try:
                self._write_to_ffmpeg(
                    generation,
                    frames,
                    audio,
                    audio_frame_meta,
                )
            except Exception as e:
                self.engine.log(f"[PIPE ERROR] client={self.client_id} write failed: {repr(e)}")
                self._request_encoder_recovery(
                    "encoder enqueue failure",
                    self._io_stop,
                )
                continue

    def _thread_main(self):
        try:
            self._run()
        except Exception as e:
            self.engine.log(
                f"[PIPE ERROR] client={self.client_id} encoder thread failed: {repr(e)}"
            )
        finally:
            self.running.clear()
            try:
                self._stop_ffmpeg()
            finally:
                self.engine.log(f"[PIPE] client={self.client_id} encoder thread ended")
                self._signal_stopped("encoder thread ended")

    def _stop_ffmpeg(self):
        self._io_stop.set()
        for q in (
            self.av_pair_q,
            self.video_job_q,
            self.audio_job_q,
        ):
            try:
                q.put_nowait(None)
            except Exception:
                pass
        if self.audio_w_fd is not None:
            try:
                os.close(self.audio_w_fd)
            except Exception:
                pass
            self.audio_w_fd = None
        proc = self.proc
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:
                pass
        for thread in self._io_threads:
            if thread is threading.current_thread():
                continue
            try:
                thread.join(timeout=0.5)
            except Exception:
                pass
        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
        self.proc = None
        self._io_threads = []
        self._writer_threads_started = False
        if self.running.is_set():
            self.av_pair_q = queue.Queue(maxsize=int(os.getenv("PIPE_AV_Q", "8")))
            self.video_job_q = queue.Queue(maxsize=1)
            self.audio_job_q = queue.Queue(maxsize=1)
            self._io_stop = threading.Event()


class RealtimeMSEEngine:
    WARMUP_IDLE = "WARMUP_IDLE"
    ASSISTANT_ACTIVE = "ASSISTANT_ACTIVE"
    ASSISTANT_TAIL = "ASSISTANT_TAIL"

    def __init__(self, args, loop: asyncio.AbstractEventLoop):
        self.args = args
        self.loop = loop
        self.log_clients: Set[asyncio.Queue] = set()
        self.media_clients: Set[MediaPipeClient] = set()
        self._client_seq = 0
        self.running = threading.Event()
        self.running.set()

        self.listener_audio_enabled = os.getenv(
            "ENGINE_LISTENER_AUDIO",
            "0",
        ).lower() not in ("0", "false", "no", "off")
        self.listener_virtual_only = os.getenv(
            "ENGINE_LISTENER_VIRTUAL_ONLY",
            "0",
        ).lower() not in ("", "0", "false", "no", "off")
        self._listener_virtual_audio = np.zeros(0, dtype=np.float32)
        self._listener_virtual_cursor = 0
        self._listener_virtual_path = ""
        self._listener_other_source = "zero"
        self._listener_transition_samples = max(
            1,
            min(int(args.hop_ms * 16), int(0.08 * 16000)),
        )
        if self.listener_audio_enabled:
            virtual_path = os.getenv(
                "ENGINE_LISTENER_VIRTUAL_AUDIO",
                "",
            ).strip()
            if not virtual_path:
                raise RuntimeError(
                    "ENGINE_LISTENER_AUDIO=1 requires "
                    "ENGINE_LISTENER_VIRTUAL_AUDIO"
                )
            loop_crossfade_samples = max(1, int(args.hop_ms * 16))
            self._listener_virtual_audio = self._load_listener_virtual_audio(
                virtual_path,
                loop_crossfade_samples,
            )
            self._listener_virtual_path = str(
                Path(virtual_path).expanduser().resolve()
            )

        self.worker_args = SimpleNamespace(
            sample=args.sample,
            hop_ms=args.hop_ms,
            denoising_steps=args.denoising_steps,
            motion_gpu=args.motion_gpu,
            render_gpu=args.render_gpu,
            feature_lag_frames=args.feature_lag_frames,
            flush_silence_sec=0.5,
            save_timeout_sec=10.0,
            port=args.port,
        )
        self.manager = omni.DyStreamWorkerManager(self.worker_args)
        self.manager.start()
        self.audio_q, self.frame_q, _ = self.manager.queues()

        self._assistant_lock = threading.Lock()
        self._speaker_chunks = deque()
        self._speaker_head_offset = 0
        self._speaker_samples = 0
        self._user_chunks = deque()
        self._user_head_offset = 0
        self._user_samples = 0
        self._last_user_audio_ts = 0.0
        self._last_user_voice_ts = 0.0
        self._user_last_rms = 0.0
        self._state = self.WARMUP_IDLE
        self._turn_id = 0
        self._stream_generation = 0
        self._last_tts_audio_ts = 0.0
        self._tail_samples_remaining = 0
        self._interrupt_bridge_audio = np.zeros(0, dtype=np.float32)
        self._interrupt_bridge_generation = -1
        self._interrupt_grace_active = False
        self._tts_reset_requested = threading.Event()
        self._assistant_turn_started_at = 0.0
        self._first_motion_submit_turn = -1
        self._first_rendered_frame_turn = -1
        self.assistant_media_drain_sec = max(
            0.0,
            float(os.getenv("ENGINE_ASSISTANT_MEDIA_DRAIN_SEC", "1.2")),
        )
        self._assistant_media_pending_until = 0.0
        self._media_control_seq = 0
        self.max_audio_buf_samples = int(float(os.getenv("ENGINE_MAX_AUDIO_BUF_SEC", "2.0")) * 16000)
        hop_samples_hint = max(1, int(args.hop_ms * 16))
        requested_high_water = int(
            float(os.getenv("ENGINE_AUDIO_INFLIGHT_HIGH_WATER_SEC", "1.0")) * 16000
        )
        max_high_water = max(
            hop_samples_hint,
            self.max_audio_buf_samples - hop_samples_hint,
        )
        self.audio_inflight_high_water_samples = min(
            max(requested_high_water, hop_samples_hint),
            max_high_water,
        )
        seg_frames_hint, _ = self._segment_frames_aligned_to_pipe_stride(
            int(args.segment_frames)
        )
        seg_samples_hint = seg_frames_hint * 640
        idle_high_water_floor = hop_samples_hint + (
            seg_samples_hint - math.gcd(hop_samples_hint, seg_samples_hint)
        )
        requested_idle_high_water = int(
            float(os.getenv("ENGINE_IDLE_INFLIGHT_HIGH_WATER_SEC", "0.32")) * 16000
        )
        self.idle_audio_inflight_high_water_samples = min(
            max(requested_idle_high_water, idle_high_water_floor),
            self.audio_inflight_high_water_samples,
        )
        self.idle_audio_inflight_floor_samples = idle_high_water_floor
        self.idle_audio_q_max = max(
            0,
            int(os.getenv("ENGINE_IDLE_AUDIO_Q_MAX", "1")),
        )
        self.idle_motion_q_max = max(
            0,
            int(os.getenv("ENGINE_IDLE_MOTION_Q_MAX", "1")),
        )
        self.idle_frame_q_max = max(
            0,
            int(os.getenv("ENGINE_IDLE_FRAME_Q_MAX", "8")),
        )
        self.max_frame_buf_frames = int(float(os.getenv("ENGINE_MAX_FRAME_BUF_SEC", "2.0")) * 25)
        self.max_user_audio_buf_samples = int(float(os.getenv("ENGINE_MAX_USER_AUDIO_BUF_SEC", "1.0")) * 16000)
        self.user_speaking_rms = float(os.getenv("ENGINE_USER_SPEAKING_RMS", "0.006"))
        self.user_speaking_hold_sec = float(os.getenv("ENGINE_USER_SPEAKING_HOLD_SEC", "0.6"))
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_seq = 0
        self.latest_frame_at = 0.0
        self.latest_visible_frame_seq = 0
        self.latest_visible_frame_at = 0.0
        capture_enabled = os.getenv("TURN_FRAME_CAPTURE", "1").strip() != "0"
        self._turn_capture_lock = threading.Lock()
        self._turn_capture_active_key: Optional[tuple[int, int]] = None
        self._turn_capture_last_frame: Optional[np.ndarray] = None
        self._turn_capture_write_q: queue.Queue = queue.Queue(maxsize=32)
        self.turn_frame_capture_dir: Optional[Path] = None
        if capture_enabled:
            capture_root = os.getenv("TURN_FRAME_CAPTURE_DIR", "").strip()
            root = (
                Path(capture_root).resolve()
                if capture_root
                else Path(__file__).resolve().parent / "logs" / "turn_frames"
            )
            run_name = time.strftime("run-%Y%m%d-%H%M%S") + f"-p{os.getpid()}"
            self.turn_frame_capture_dir = root / run_name
            self.turn_frame_capture_dir.mkdir(parents=True, exist_ok=True)
            threading.Thread(
                target=self._turn_capture_writer_loop,
                name="turn-frame-writer",
                daemon=True,
            ).start()
        compare_dir = os.getenv("AUDIO_COMPARE_DIR", "").strip()
        self.audio_compare_dir = Path(compare_dir).resolve() if compare_dir else None
        self._debug_tts_turn: Optional[int] = None
        self._debug_tts_chunks = []
        if self.audio_compare_dir is not None:
            self.audio_compare_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"[AUDIO_COMPARE] enabled dir={self.audio_compare_dir}")

        self.log(f"[ENGINE VERSION] {ENGINE_VERSION}")
        if self.listener_audio_enabled:
            virtual_rms = float(
                np.sqrt(
                    np.mean(self._listener_virtual_audio.astype(np.float32) ** 2)
                    + 1e-12
                )
            )
            self.log(
                "[ENGINE VIRTUAL OTHER] loaded "
                f"path={self._listener_virtual_path} "
                f"samples={len(self._listener_virtual_audio)} "
                f"seconds={len(self._listener_virtual_audio) / 16000.0:.3f} "
                f"rms={virtual_rms:.6f} "
                f"transition_ms={self._listener_transition_samples / 16.0:.1f}"
            )
        if self.turn_frame_capture_dir is not None:
            self.log(f"[TURN CAPTURE] dir={self.turn_frame_capture_dir}")
        self.feed_thread = threading.Thread(target=self._feed_and_collect_loop, daemon=True)
        self.feed_thread.start()

    @staticmethod
    def _put_log_nowait(q: asyncio.Queue, payload: Dict[str, str]):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def log(self, msg: str):
        now = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(now))
        line = f"[{ts}.{int(now % 1 * 1000):03d}] {msg}"
        print(line, flush=True)
        for q in list(self.log_clients):
            try:
                self.loop.call_soon_threadsafe(
                    self._put_log_nowait,
                    q,
                    {"type": "log", "message": line},
                )
            except RuntimeError:
                pass
    def _broadcast_media_control(self, payload: Dict[str, Any]):
        with self._assistant_lock:
            self._media_control_seq += 1
            event_seq = self._media_control_seq
        message = dict(payload)
        message.setdefault("event_seq", event_seq)
        encoded = json.dumps(message, ensure_ascii=False)
        for client in list(self.media_clients):
            try:
                client.put_text(encoded, critical=True)
            except Exception:
                pass

    def _turn_capture_path(
        self,
        key: tuple[int, int],
        position: str,
    ) -> Path:
        generation, turn_id = key
        return self.turn_frame_capture_dir / (
            f"turn-{turn_id:04d}-g{generation:04d}-{position}.png"
        )

    def _queue_turn_capture_write(
        self,
        key: tuple[int, int],
        position: str,
        frame: np.ndarray,
    ):
        if getattr(self, "turn_frame_capture_dir", None) is None:
            return
        job = (
            self._turn_capture_path(key, position),
            np.ascontiguousarray(frame, dtype=np.uint8),
        )
        try:
            self._turn_capture_write_q.put_nowait(job)
        except queue.Full:
            self.log(
                f"[TURN CAPTURE] writer queue full; dropped "
                f"turn={key[1]} position={position}"
            )

    def _turn_capture_writer_loop(self):
        while self.running.is_set():
            try:
                path, frame = self._turn_capture_write_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                Image.fromarray(frame).save(
                    path,
                    format="PNG",
                    compress_level=1,
                )
                self.log(f"[TURN CAPTURE] wrote {path}")
            except Exception as exc:
                self.log(f"[TURN CAPTURE] write failed: {exc!r}")

    def _capture_assistant_frame(
        self,
        generation: int,
        turn_id: int,
        frame: np.ndarray,
    ):
        if getattr(self, "turn_frame_capture_dir", None) is None or turn_id < 0:
            return
        key = (int(generation), int(turn_id))
        current = np.ascontiguousarray(frame, dtype=np.uint8).copy()
        writes = []
        with self._turn_capture_lock:
            if self._turn_capture_active_key != key:
                if (
                    self._turn_capture_active_key is not None
                    and self._turn_capture_last_frame is not None
                ):
                    writes.append((
                        self._turn_capture_active_key,
                        "last",
                        self._turn_capture_last_frame,
                    ))
                self._turn_capture_active_key = key
                writes.append((key, "first", current))
            self._turn_capture_last_frame = current
        for write_key, position, write_frame in writes:
            self._queue_turn_capture_write(write_key, position, write_frame)

    def _finish_assistant_frame_capture(self, reason: str):
        if getattr(self, "turn_frame_capture_dir", None) is None:
            return
        with self._turn_capture_lock:
            key = self._turn_capture_active_key
            frame = self._turn_capture_last_frame
            self._turn_capture_active_key = None
            self._turn_capture_last_frame = None
        if key is None or frame is None:
            return
        self._queue_turn_capture_write(key, "last", frame)
        self.log(f"[TURN CAPTURE] finalized turn={key[1]} reason={reason}")

    def _debug_begin_tts_dump(self, turn_id: int):
        if getattr(self, "audio_compare_dir", None) is None:
            return
        self._debug_tts_turn = turn_id
        self._debug_tts_chunks = []
        self.log(f"[AUDIO_COMPARE] begin turn={turn_id}")

    def _debug_append_tts_dump(self, turn_id: int, x16: np.ndarray):
        if getattr(self, "audio_compare_dir", None) is None:
            return
        if self._debug_tts_turn != turn_id:
            self._debug_begin_tts_dump(turn_id)
        self._debug_tts_chunks.append(np.asarray(x16, dtype=np.float32).copy())

    def _debug_write_tts_dump(self, turn_id: int):
        if getattr(self, "audio_compare_dir", None) is None or self._debug_tts_turn != turn_id:
            return
        if not self._debug_tts_chunks:
            self.log(f"[AUDIO_COMPARE] no tts chunks turn={turn_id}")
            return
        audio = np.concatenate(self._debug_tts_chunks).astype(np.float32)
        path = self.audio_compare_dir / f"turn_{turn_id:04d}_doubao_tts_16k.wav"
        sf.write(path, audio, 16000)
        self.log(f"[AUDIO_COMPARE] wrote {path} samples={len(audio)} sec={len(audio)/16000:.3f}")

    def begin_assistant_turn(self):
        with self._assistant_lock:
            self._turn_id += 1
            self._state = self.ASSISTANT_ACTIVE
            self._interrupt_grace_active = False
            self._speaker_chunks.clear()
            self._speaker_head_offset = 0
            self._speaker_samples = 0
            self._clear_user_audio_locked()
            self._tail_samples_remaining = 0
            turn_id = self._turn_id
            stream_generation = self._stream_generation
            self._assistant_turn_started_at = time.monotonic()
        self._broadcast_media_control(
            {
                "type": "assistant_turn_started",
                "generation": stream_generation,
                "turn_id": turn_id,
            }
        )
        self.log(f"[ENGINE STATE] {self.ASSISTANT_ACTIVE} turn={turn_id}")
        self._debug_begin_tts_dump(turn_id)
        return turn_id

    def enqueue_speaker_audio(self, x16: np.ndarray):
        x16 = np.asarray(x16, dtype=np.float32)
        if len(x16) == 0:
            return
        fallback_turn = None
        fallback_generation = None
        debug_turn = None
        with self._assistant_lock:
            if self._state == self.WARMUP_IDLE:
                self._turn_id += 1
                self._state = self.ASSISTANT_ACTIVE
                self._clear_user_audio_locked()
                fallback_turn = self._turn_id
                fallback_generation = self._stream_generation
                self._assistant_turn_started_at = time.monotonic()
            debug_turn = self._turn_id
            self._speaker_chunks.append(x16.copy())
            self._speaker_samples += len(x16)
            self._last_tts_audio_ts = time.time()
        if fallback_turn is not None:
            self._broadcast_media_control(
                {
                    "type": "assistant_turn_started",
                    "generation": fallback_generation,
                    "turn_id": fallback_turn,
                }
            )
            self.log(f"[ENGINE STATE] fallback {self.ASSISTANT_ACTIVE} turn={fallback_turn}")
            self._debug_begin_tts_dump(fallback_turn)
        if debug_turn is not None:
            self._debug_append_tts_dump(debug_turn, x16)

    def end_assistant_turn(self):
        with self._assistant_lock:
            if self._state == self.WARMUP_IDLE:
                return
            self._state = self.ASSISTANT_TAIL
            self._interrupt_grace_active = False
            self._tail_samples_remaining = int(
                float(os.getenv("ENGINE_TTS_TAIL_SEC", "0.4")) * 16000
            )
            turn_id = self._turn_id
        self.log(f"[ENGINE STATE] {self.ASSISTANT_TAIL} turn={turn_id}")
        self._debug_write_tts_dump(turn_id)

    def clear_assistant_audio(self) -> int:
        with self._assistant_lock:
            dropped = len(self._speaker_chunks)
            self._speaker_chunks.clear()
            self._speaker_head_offset = 0
            self._speaker_samples = 0
            self._tail_samples_remaining = 0
            self._interrupt_bridge_audio = np.zeros(0, dtype=np.float32)
            self._interrupt_bridge_generation = -1
            self._interrupt_grace_active = False
        return dropped

    def _speaker_prefix_locked(self, n: int) -> np.ndarray:
        pieces = []
        remaining = max(0, int(n))
        for index, chunk in enumerate(self._speaker_chunks):
            if remaining <= 0:
                break
            start = self._speaker_head_offset if index == 0 else 0
            available = max(0, len(chunk) - start)
            take = min(remaining, available)
            if take:
                pieces.append(chunk[start:start + take])
                remaining -= take
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    @staticmethod
    def _half_cosine_fade_to_zero(
        source: np.ndarray,
        output_samples: int,
    ) -> np.ndarray:
        output_samples = max(1, int(output_samples))
        source = np.asarray(source, dtype=np.float32).reshape(-1)
        take = min(len(source), output_samples)
        out = np.zeros(output_samples, dtype=np.float32)
        if take <= 0:
            return out
        if take == 1:
            envelope = np.ones(1, dtype=np.float32)
        else:
            phase = np.linspace(0.0, np.pi, take, dtype=np.float32)
            envelope = 0.5 * (1.0 + np.cos(phase))
        out[:take] = source[:take] * envelope
        return out

    def interrupt_assistant(self) -> int:
        with self._assistant_lock:
            dropped = len(self._speaker_chunks)
            hop_samples = max(1, int(self.args.hop_ms * 16))
            grace_sec = max(
                0.0,
                float(
                    os.getenv(
                        "ENGINE_INTERRUPT_GRACE_SEC",
                        os.getenv("ENGINE_INTERRUPT_BRIDGE_SEC", "0.40"),
                    )
                ),
            )
            grace_samples = int(math.ceil(grace_sec * 16000 / hop_samples)) * hop_samples
            grace_source = self._speaker_prefix_locked(grace_samples)
            grace_audio = (
                self._half_cosine_fade_to_zero(
                    grace_source,
                    grace_samples,
                )
                if grace_samples > 0
                else np.zeros(0, dtype=np.float32)
            )
            self._speaker_chunks.clear()
            self._speaker_head_offset = 0
            if len(grace_audio) > 0:
                self._speaker_chunks.append(grace_audio)
            self._speaker_samples = len(grace_audio)
            self._tail_samples_remaining = 0
            self._state = self.ASSISTANT_TAIL
            self._interrupt_grace_active = True
            turn_id = self._turn_id
            stream_generation = self._stream_generation
            self._interrupt_bridge_audio = np.zeros(0, dtype=np.float32)
            self._interrupt_bridge_generation = -1
            self._assistant_turn_started_at = 0.0
        grace_rms = float(
            np.sqrt(
                np.mean(
                    grace_audio.astype(np.float32) ** 2
                )
                + 1e-12
            )
        ) if len(grace_audio) else 0.0
        self._broadcast_media_control(
            {
                "type": "assistant_interrupted",
                "generation": stream_generation,
                "turn_id": turn_id,
                "graceful": True,
                "grace_samples": len(grace_audio),
                "grace_sec": len(grace_audio) / 16000.0,
            }
        )
        self.log(
            f"[ENGINE STATE] interrupt -> {self.ASSISTANT_TAIL} "
            f"turn={turn_id} generation={stream_generation} "
            f"grace_samples={len(grace_audio)} "
            f"grace_source_samples={len(grace_source)} "
            f"grace_rms={grace_rms:.6f} "
            "generation_preserved=1 reset=0"
        )
        return dropped

    def _pop_interrupt_bridge(
        self,
        n: int,
        generation: int,
    ) -> Optional[np.ndarray]:
        with self._assistant_lock:
            if self._tts_reset_requested.is_set():
                return None
            if self._interrupt_bridge_generation != generation:
                if self._interrupt_bridge_generation < generation:
                    self._interrupt_bridge_audio = np.zeros(
                        0,
                        dtype=np.float32,
                    )
                    self._interrupt_bridge_generation = -1
                return None
            bridge = self._interrupt_bridge_audio
            self._interrupt_bridge_audio = np.zeros(0, dtype=np.float32)
            self._interrupt_bridge_generation = -1
        if len(bridge) < n:
            bridge = np.pad(bridge, (0, n - len(bridge)))
        elif len(bridge) > n:
            bridge = bridge[:n]
        return np.asarray(bridge, dtype=np.float32)

    def _mark_pipeline_milestone(self, turn_id: int, milestone: str):
        marker_attr = {
            "motion_submit": "_first_motion_submit_turn",
            "rendered_frame": "_first_rendered_frame_turn",
        }.get(milestone)
        if marker_attr is None:
            raise ValueError(f"unknown pipeline milestone: {milestone}")
        with self._assistant_lock:
            if (
                turn_id != self._turn_id
                or self._assistant_turn_started_at <= 0.0
                or getattr(self, marker_attr) == turn_id
            ):
                return
            setattr(self, marker_attr, turn_id)
            elapsed_ms = (
                time.monotonic() - self._assistant_turn_started_at
            ) * 1000.0
        self.log(
            f"[PIPELINE LATENCY] turn={turn_id} milestone={milestone} "
            f"after_engine_pcm_ms={elapsed_ms:.1f}"
        )

    def register_media_client(self) -> MediaPipeClient:
        self._client_seq += 1
        client = MediaPipeClient(self, self._client_seq)
        _, _, stream_generation, _, _ = self._state_snapshot()
        client.advance_generation(stream_generation)
        self.media_clients.add(client)
        client.start()
        self.log(f"[MEDIA] pipe client registered id={client.client_id} clients={len(self.media_clients)}")
        return client

    def unregister_media_client(self, client: MediaPipeClient):
        self.media_clients.discard(client)
        client.stop()
        self.log(f"[MEDIA] pipe client unregistered id={client.client_id} clients={len(self.media_clients)}")

    def register_log_client(self, q: asyncio.Queue):
        self.log_clients.add(q)

    def unregister_log_client(self, q: asyncio.Queue):
        self.log_clients.discard(q)

    def _state_snapshot(self):
        with self._assistant_lock:
            return (
                self._state,
                self._turn_id,
                self._stream_generation,
                self._speaker_samples,
                self._tail_samples_remaining,
            )

    def assistant_turn_control_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the in-flight turn needed to re-arm a replacement MSE epoch."""
        with self._assistant_lock:
            if self._state == self.WARMUP_IDLE:
                return None
            return {
                "type": "assistant_turn_started",
                "generation": self._stream_generation,
                "turn_id": self._turn_id,
            }

    def _mark_assistant_media_pending(self):
        deadline = time.monotonic() + self.assistant_media_drain_sec
        with self._assistant_lock:
            self._assistant_media_pending_until = max(
                self._assistant_media_pending_until,
                deadline,
            )

    def assistant_output_pending(self) -> bool:
        """Return whether an interruption still needs to invalidate local media."""
        now = time.monotonic()
        with self._assistant_lock:
            return (
                self._state != self.WARMUP_IDLE
                or self._speaker_samples > 0
                or self._tail_samples_remaining > 0
                or now < self._assistant_media_pending_until
            )

    def _pop_speaker_samples(self, n: int, pad: bool) -> np.ndarray:
        pieces = []
        need = n
        with self._assistant_lock:
            while need > 0 and self._speaker_chunks:
                head = self._speaker_chunks[0]
                available = len(head) - self._speaker_head_offset
                take = min(need, available)
                start = self._speaker_head_offset
                pieces.append(head[start:start + take])
                self._speaker_head_offset += take
                self._speaker_samples -= take
                need -= take
                if self._speaker_head_offset >= len(head):
                    self._speaker_chunks.popleft()
                    self._speaker_head_offset = 0
        if pieces:
            out = np.concatenate(pieces).astype(np.float32, copy=False)
        else:
            out = np.zeros(0, dtype=np.float32)
        if pad and len(out) < n:
            out = np.pad(out, (0, n - len(out)))
        return out

    @staticmethod
    def _pcm16le_to_float16k(pcm16: bytes) -> np.ndarray:
        if not pcm16:
            return np.zeros(0, dtype=np.float32)
        usable = len(pcm16) - (len(pcm16) % 2)
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(pcm16[:usable], dtype="<i2").astype(np.float32) / 32768.0

    @staticmethod
    def _load_listener_virtual_audio(
        path_value: str,
        loop_crossfade_samples: int,
    ) -> np.ndarray:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"missing Listener virtual audio: {path}")

        try:
            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to read Listener virtual audio {path}: {exc}"
            ) from exc

        if sample_rate <= 0 or audio.shape[0] == 0:
            raise RuntimeError(f"empty Listener virtual audio: {path}")
        audio = np.mean(audio, axis=1, dtype=np.float32)
        if sample_rate != 16000:
            divisor = math.gcd(int(sample_rate), 16000)
            audio = resample_poly(
                audio,
                16000 // divisor,
                int(sample_rate) // divisor,
            ).astype(np.float32, copy=False)

        audio = np.nan_to_num(
            audio,
            copy=False,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

        # The verified source WAV contains a short speech region followed by
        # several seconds of zero padding. Trim only leading/trailing silence
        # so the virtual other channel cannot spend most of each loop at 0/0.
        analysis_frame = 320  # 20 ms at 16 kHz
        frame_count = len(audio) // analysis_frame
        if frame_count <= 0:
            raise RuntimeError(f"Listener virtual audio is too short: {path}")
        framed = audio[:frame_count * analysis_frame].reshape(
            frame_count,
            analysis_frame,
        )
        frame_rms = np.sqrt(
            np.mean(framed.astype(np.float32) ** 2, axis=1) + 1e-12
        )
        active_frames = np.flatnonzero(frame_rms >= 0.003)
        if len(active_frames) == 0:
            raise RuntimeError(
                f"Listener virtual audio contains no usable speech: {path}"
            )
        edge_padding_frames = 4  # retain 80 ms of natural speech boundary
        active_start = max(
            0,
            (int(active_frames[0]) - edge_padding_frames) * analysis_frame,
        )
        active_end = min(
            len(audio),
            (int(active_frames[-1]) + 1 + edge_padding_frames) * analysis_frame,
        )
        audio = audio[active_start:active_end]

        crossfade = min(
            max(1, int(loop_crossfade_samples)),
            max(1, len(audio) // 4),
        )
        if len(audio) <= crossfade * 2:
            raise RuntimeError(
                "Listener virtual audio must be longer than two model hops: "
                f"path={path} samples={len(audio)}"
            )
        fade_in = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
        loop_bridge = (
            audio[-crossfade:] * (1.0 - fade_in)
            + audio[:crossfade] * fade_in
        )
        audio = np.concatenate(
            [audio[crossfade:-crossfade], loop_bridge]
        ).astype(np.float32, copy=False)
        if len(audio) < loop_crossfade_samples:
            raise RuntimeError(
                "Listener virtual audio is shorter than one model hop after "
                f"loop preparation: path={path} samples={len(audio)}"
            )
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _next_virtual_listener_hop(self, n: int) -> np.ndarray:
        audio = self._listener_virtual_audio
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if len(audio) < n:
            raise RuntimeError(
                "Listener virtual audio is unavailable or shorter than one hop"
            )
        start = self._listener_virtual_cursor
        end = start + n
        if end <= len(audio):
            out = audio[start:end].copy()
        else:
            split = len(audio) - start
            out = np.concatenate([audio[start:], audio[:n - split]])
        self._listener_virtual_cursor = end % len(audio)
        return np.ascontiguousarray(out, dtype=np.float32)

    def _clear_user_audio_locked(self):
        self._user_chunks.clear()
        self._user_head_offset = 0
        self._user_samples = 0
        self._last_user_voice_ts = 0.0
        self._user_last_rms = 0.0

    def _pop_user_samples_locked(self, n: int, pad: bool) -> np.ndarray:
        pieces = []
        need = n
        while need > 0 and self._user_chunks:
            head = self._user_chunks[0]
            available = len(head) - self._user_head_offset
            take = min(need, available)
            start = self._user_head_offset
            pieces.append(head[start:start + take])
            self._user_head_offset += take
            self._user_samples -= take
            need -= take
            if self._user_head_offset >= len(head):
                self._user_chunks.popleft()
                self._user_head_offset = 0
        if pieces:
            out = np.concatenate(pieces).astype(np.float32, copy=False)
        else:
            out = np.zeros(0, dtype=np.float32)
        if pad and len(out) < n:
            out = np.pad(out, (0, n - len(out)))
        return out

    def enqueue_user_audio(self, pcm16: bytes):
        x16 = self._pcm16le_to_float16k(pcm16)
        if len(x16) == 0:
            return
        rms = float(np.sqrt(np.mean(x16.astype(np.float32) ** 2) + 1e-12))
        now = time.time()
        with self._assistant_lock:
            self._user_chunks.append(x16.copy())
            self._user_samples += len(x16)
            self._last_user_audio_ts = now
            self._user_last_rms = rms
            if rms >= self.user_speaking_rms:
                self._last_user_voice_ts = now
            excess = self._user_samples - self.max_user_audio_buf_samples
            if excess > 0:
                self._pop_user_samples_locked(excess, pad=False)

    def _pop_user_hop(
        self,
        n: int,
    ) -> tuple[np.ndarray, bool, float, int, int]:
        with self._assistant_lock:
            # This channel controls the current avatar motion, so stale mic
            # samples are worse than dropped samples. Keep only the freshest
            # model hop after render/backpressure pauses.
            stale = max(0, self._user_samples - n)
            if stale:
                self._pop_user_samples_locked(stale, pad=False)
            available = min(n, self._user_samples)
            out = self._pop_user_samples_locked(n, pad=True)
            remaining = self._user_samples
        hop_rms = float(np.sqrt(np.mean(out.astype(np.float32) ** 2) + 1e-12)) if len(out) else 0.0
        # Voice hold is useful for dialogue turn detection, but it must not
        # suppress virtual Listener conditioning with a padded/quiet mic hop.
        speaking = available > 0 and hop_rms >= self.user_speaking_rms
        return out, speaking, hop_rms, remaining, available

    def _pop_listener_other_hop(
        self,
        n: int,
    ) -> tuple[np.ndarray, str, float]:
        (
            user_hop,
            user_speaking,
            user_rms,
            _,
            user_available,
        ) = self._pop_user_hop(n)
        virtual_hop = self._next_virtual_listener_hop(n)

        if (
            not self.listener_virtual_only
            and user_speaking
            and user_available > 0
        ):
            desired = user_hop
            source = "mic"
            mode = "USER_SPEAKING"
        else:
            desired = virtual_hop
            source = "virtual"
            mode = "LISTENER_VIRTUAL"

        previous_source = self._listener_other_source
        if previous_source != source:
            fade_count = min(self._listener_transition_samples, n)
            fade_in = np.linspace(0.0, 1.0, fade_count, dtype=np.float32)
            if previous_source == "virtual":
                previous = virtual_hop
            elif previous_source == "mic" and user_available > 0:
                previous = user_hop
            else:
                previous = np.zeros(n, dtype=np.float32)
            desired = desired.copy()
            desired[:fade_count] = (
                previous[:fade_count] * (1.0 - fade_in)
                + desired[:fade_count] * fade_in
            )
            self.log(
                "[ENGINE VIRTUAL OTHER] "
                f"transition={previous_source}->{source} "
                f"fade_ms={fade_count / 16.0:.1f}"
            )
        self._listener_other_source = source
        other_rms = float(
            np.sqrt(np.mean(desired.astype(np.float32) ** 2) + 1e-12)
        )
        return np.ascontiguousarray(desired, dtype=np.float32), mode, other_rms

    def _route_partial_speaker_hop_other(
        self,
        hop_samples: int,
        real_speaker_samples: int,
    ) -> np.ndarray:
        """Switch from self to other at the last real speaker sample."""
        real = max(0, min(int(real_speaker_samples), int(hop_samples)))
        out = np.zeros(hop_samples, dtype=np.float32)
        padding = hop_samples - real
        if padding <= 0:
            return out
        # This is the normal reply-completion path, not barge-in.  Do not use
        # an arbitrarily short padding slice (which can be one sample) to
        # classify or consume the microphone.  Barge-in already has its own
        # full-hop self+mic route.  Here the trained Listener/other branch
        # resumes from the existing virtual-audio cursor exactly at the first
        # padded self sample.
        listener_tail = self._next_virtual_listener_hop(padding)
        if self._listener_other_source != "virtual":
            self.log(
                "[ENGINE VIRTUAL OTHER] "
                f"transition={self._listener_other_source}->virtual "
                "strict_branch_handoff=1"
            )
        self._listener_other_source = "virtual"
        out[real:] = listener_tail
        return np.ascontiguousarray(out, dtype=np.float32)

    def _consume_tail_silence(self, n: int) -> np.ndarray:
        with self._assistant_lock:
            take = min(n, self._tail_samples_remaining)
            self._tail_samples_remaining -= take
        return np.zeros(n, dtype=np.float32)

    def _finish_tail_if_ready(self):
        with self._assistant_lock:
            if (
                self._state == self.ASSISTANT_TAIL
                and self._speaker_samples == 0
                and self._tail_samples_remaining == 0
            ):
                self._state = self.WARMUP_IDLE
                self._interrupt_grace_active = False
                turn_id = self._turn_id
            else:
                return
        self.log(f"[ENGINE STATE] {self.WARMUP_IDLE} turn={turn_id}")

    def _drain_any_queue(self, q, max_items: int = 4096) -> int:
        if q is None:
            return 0
        n = 0
        while n < max_items:
            try:
                q.get_nowait()
                n += 1
            except Exception:
                break
        return n

    @staticmethod
    def _queue_size(q) -> int:
        if q is None:
            return 0
        try:
            return max(0, int(q.qsize()))
        except (AttributeError, NotImplementedError, OSError):
            return 0

    @staticmethod
    def _segment_frames_aligned_to_pipe_stride(requested_frames: int) -> tuple[int, int]:
        seg_frames = max(1, int(requested_frames))
        try:
            pipe_stride = max(1, int(os.getenv("PIPE_FRAME_STRIDE", "2")))
        except ValueError:
            pipe_stride = 2
        if pipe_stride > 1 and seg_frames % pipe_stride != 0:
            seg_frames = max(pipe_stride, (seg_frames // pipe_stride) * pipe_stride)
        return seg_frames, pipe_stride

    def _apply_live_reset(
        self,
        audio_buf: np.ndarray,
        audio_frame_meta_buf: list,
        frames_buf: list,
    ):
        dropped_audio = int(len(audio_buf))
        dropped_audio_meta = int(len(audio_frame_meta_buf))
        dropped_frames = int(len(frames_buf))
        audio_buf = np.zeros(0, dtype=np.float32)
        audio_frame_meta_buf.clear()
        frames_buf.clear()

        drained_audio_q = self._drain_any_queue(self.audio_q)
        drained_frame_q = self._drain_any_queue(self.frame_q)
        drained_motion_q = self._drain_any_queue(getattr(self.manager, "motion_q", None))
        _, _, stream_generation, _, _ = self._state_snapshot()
        try:
            self.audio_q.put_nowait(
                {"type": "reset", "generation": stream_generation}
            )
        except Exception as e:
            self.log(f"[ENGINE WARN] worker reset enqueue failed: {repr(e)}")

        reset_client_stream = os.getenv(
            "ENGINE_TTS_STREAM_RESET",
            "0",
        ).lower() in ("1", "true", "yes", "on")
        dropped_clients = 0
        for client in list(self.media_clients):
            try:
                dropped_clients += int(
                    client.advance_generation(stream_generation)
                )
                if reset_client_stream:
                    dropped_clients += int(client.request_stream_reset())
            except Exception:
                pass

        self.log(
            f"[ENGINE LIVE] reset applied dropped_audio_buf={dropped_audio} "
            f"dropped_audio_meta={dropped_audio_meta} "
            f"dropped_frames_buf={dropped_frames} drained_audio_q={drained_audio_q} "
            f"drained_motion_q={drained_motion_q} drained_frame_q={drained_frame_q} "
            f"dropped_client_q={dropped_clients} "
            f"client_stream_reset={int(reset_client_stream)} "
            f"generation={stream_generation}"
        )
        return audio_buf, audio_frame_meta_buf, frames_buf

    def _feed_and_collect_loop(self):
        hop_samples = int(16000 * self.args.hop_ms / 1000)
        if hop_samples <= 0:
            hop_samples = 3200

        requested_seg_frames = max(1, int(self.args.segment_frames))
        seg_frames, pipe_stride = self._segment_frames_aligned_to_pipe_stride(requested_seg_frames)
        seg_samples = seg_frames * 640  # source DyStream grid is 25fps

        feed_idle = os.getenv("ENGINE_FEED_IDLE", "1").lower() not in ("0", "false", "no", "off")
        idle_warmup_segments = max(0, int(os.getenv("ENGINE_IDLE_WARMUP_SEGMENTS", "16")))
        idle_continuous = os.getenv("ENGINE_IDLE_CONTINUOUS", "1").lower() not in ("0", "false", "no", "off")
        idle_visible = os.getenv("ENGINE_IDLE_VISIBLE", "1").lower() not in ("0", "false", "no", "off")
        idle_only_with_client = os.getenv("ENGINE_IDLE_ONLY_WITH_CLIENT", "1").lower() not in ("0", "false", "no", "off")
        idle_compact_segments = max(
            0,
            int(os.getenv("ENGINE_IDLE_COMPACT_SEGMENTS", "40")),
        )
        listener_audio = self.listener_audio_enabled

        audio_buf = np.zeros(0, dtype=np.float32)
        audio_frame_meta_buf = []
        frames_buf = []
        next_feed_time = time.time()
        seg_idx = 0
        idle_segments_sent = 0
        idle_segments_since_compact = 0
        last_drop_log = 0.0
        last_backpressure_log = 0.0
        last_idle_backpressure_log = 0.0
        last_listener_log = 0.0
        last_listener_mode = None

        self.log(
            f"[ENGINE] feed loop started hop_samples={hop_samples} segment_frames={seg_frames} "
            f"requested_segment_frames={requested_seg_frames} pipe_frame_stride={pipe_stride} "
            f"feed_idle={int(feed_idle)} idle_warmup_segments={idle_warmup_segments} "
            f"idle_continuous={int(idle_continuous)} idle_visible={int(idle_visible)} "
            f"idle_compact_segments={idle_compact_segments} "
            f"listener_audio={int(listener_audio)} "
            f"listener_virtual_only={int(self.listener_virtual_only)} "
            f"user_rms={self.user_speaking_rms:.4f} "
            f"max_audio_buf={self.max_audio_buf_samples} "
            f"audio_inflight_high_water={self.audio_inflight_high_water_samples} "
            f"idle_inflight_high_water={self.idle_audio_inflight_high_water_samples} "
            f"idle_inflight_floor={self.idle_audio_inflight_floor_samples} "
            f"idle_queue_max={self.idle_audio_q_max}/"
            f"{self.idle_motion_q_max}/{self.idle_frame_q_max} "
            f"max_frame_buf={self.max_frame_buf_frames}"
        )
        if seg_frames != requested_seg_frames:
            self.log(
                f"[ENGINE AUDIO] adjusted segment_frames {requested_seg_frames}->{seg_frames} "
                f"to avoid padding silence with PIPE_FRAME_STRIDE={pipe_stride}"
            )

        while self.running.is_set():
            now = time.time()

            if self._tts_reset_requested.is_set():
                self._tts_reset_requested.clear()
                (
                    audio_buf,
                    audio_frame_meta_buf,
                    frames_buf,
                ) = self._apply_live_reset(
                    audio_buf,
                    audio_frame_meta_buf,
                    frames_buf,
                )
                next_feed_time = now

            if now >= next_feed_time:
                (
                    state,
                    turn_id,
                    stream_generation,
                    speaker_samples,
                    tail_samples,
                ) = self._state_snapshot()
                chunk = None
                media_chunk = None
                chunk_other = None
                idle_candidate = False
                visible = False
                mode = "IDLE_NO_USER"
                assistant_feed_allowed = (
                    len(audio_buf) + hop_samples
                    <= self.audio_inflight_high_water_samples
                )
                idle_audio_q_depth = self._queue_size(self.audio_q)
                idle_motion_q_depth = self._queue_size(
                    getattr(self.manager, "motion_q", None)
                )
                idle_frame_q_depth = self._queue_size(self.frame_q)
                idle_backlog_allowed = (
                    len(audio_buf) + hop_samples
                    <= self.idle_audio_inflight_high_water_samples
                    and idle_audio_q_depth < max(1, self.idle_audio_q_max)
                    and idle_motion_q_depth < max(1, self.idle_motion_q_max)
                    and idle_frame_q_depth < max(1, self.idle_frame_q_max)
                )

                interrupt_bridge = self._pop_interrupt_bridge(
                    hop_samples,
                    stream_generation,
                )
                if interrupt_bridge is not None:
                    chunk = interrupt_bridge
                    media_chunk = np.zeros_like(
                        interrupt_bridge,
                        dtype=np.float32,
                    )
                    if listener_audio:
                        chunk_other, _, _ = self._pop_listener_other_hop(
                            hop_samples,
                        )
                    visible = len(self.media_clients) > 0
                    mode = "INTERRUPT_BRIDGE"
                    idle_segments_since_compact = 0
                    self.log(
                        f"[ENGINE BRIDGE] generation={stream_generation} "
                        f"turn={turn_id} motion_rms="
                        f"{float(np.sqrt(np.mean(chunk ** 2) + 1e-12)):.6f} "
                        "media_rms=0.000000"
                    )
                elif state == self.WARMUP_IDLE:
                    has_client = len(self.media_clients) > 0
                    would_be_visible = bool(idle_visible and has_client)
                    # The first model hop may produce only one frame. Let one
                    # extra hop through before the first complete media segment
                    # so the 200 ms model grid cannot deadlock on the 160 ms
                    # output grid.
                    idle_bootstrap_allowed = (
                        would_be_visible
                        and seg_idx == 0
                        and len(frames_buf) < seg_frames
                    )
                    idle_feed_allowed = (
                        not would_be_visible
                        or idle_bootstrap_allowed
                        or idle_backlog_allowed
                    )
                    should_feed_idle = (
                        idle_segments_sent < idle_warmup_segments
                        or (
                            idle_continuous
                            and (has_client or not idle_only_with_client)
                        )
                    )
                    if feed_idle and should_feed_idle and idle_feed_allowed:
                        if (
                            idle_compact_segments
                            and idle_segments_since_compact >= idle_compact_segments
                        ):
                            try:
                                self.audio_q.put_nowait(
                                    {
                                        "type": "compact",
                                        "generation": stream_generation,
                                    }
                                )
                            except Exception as e:
                                self.log(
                                    f"[ENGINE WARN] idle compact enqueue failed: {repr(e)}"
                                )
                            idle_segments_since_compact = 0
                            self.log(
                                f"[ENGINE IDLE] audio history compact requested "
                                f"turn={turn_id}"
                            )
                        chunk = np.zeros(hop_samples, dtype=np.float32)
                        idle_candidate = True
                        if listener_audio:
                            chunk_other, mode, _ = self._pop_listener_other_hop(
                                hop_samples,
                            )
                        visible = would_be_visible
                        idle_segments_sent += 1
                        idle_segments_since_compact += 1
                    elif (
                        feed_idle
                        and should_feed_idle
                        and not idle_feed_allowed
                        and now - last_idle_backpressure_log > 1.0
                    ):
                        self.log(
                            "[ENGINE IDLE] backpressure "
                            f"audio_inflight={len(audio_buf)} "
                            f"high_water={self.idle_audio_inflight_high_water_samples} "
                            f"queues={idle_audio_q_depth}/"
                            f"{idle_motion_q_depth}/{idle_frame_q_depth}"
                        )
                        last_idle_backpressure_log = now
                elif state == self.ASSISTANT_ACTIVE:
                    idle_segments_since_compact = 0
                    # Do not turn websocket packet jitter into audible zero gaps.
                    # Wait until a complete model hop is available.
                    if speaker_samples >= hop_samples and assistant_feed_allowed:
                        chunk = self._pop_speaker_samples(hop_samples, pad=False)
                        if listener_audio:
                            self._listener_other_source = "zero"
                        visible = True
                        mode = "ASSISTANT_ACTIVE"
                elif state == self.ASSISTANT_TAIL:
                    idle_segments_since_compact = 0
                    with self._assistant_lock:
                        interrupt_grace_active = self._interrupt_grace_active
                    if speaker_samples >= hop_samples and assistant_feed_allowed:
                        chunk = self._pop_speaker_samples(hop_samples, pad=False)
                        if listener_audio:
                            if interrupt_grace_active:
                                chunk_other, _, _ = self._pop_listener_other_hop(
                                    hop_samples,
                                )
                            else:
                                self._listener_other_source = "zero"
                        visible = True
                        mode = (
                            "ASSISTANT_INTERRUPT_TAIL"
                            if interrupt_grace_active
                            else "ASSISTANT_TAIL"
                        )
                    elif speaker_samples > 0 and assistant_feed_allowed:
                        real_speaker_samples = min(speaker_samples, hop_samples)
                        chunk = self._pop_speaker_samples(hop_samples, pad=True)
                        if listener_audio:
                            if interrupt_grace_active:
                                chunk_other, _, _ = self._pop_listener_other_hop(
                                    hop_samples,
                                )
                            else:
                                chunk_other = self._route_partial_speaker_hop_other(
                                    hop_samples,
                                    real_speaker_samples,
                                )
                                self.log(
                                    "[ENGINE LISTENER] "
                                    "transition=speaker->listener "
                                    f"speaker_samples={real_speaker_samples} "
                                    "padding_samples="
                                    f"{hop_samples - real_speaker_samples} "
                                    "strict_branch_handoff=1"
                                )
                        visible = True
                        mode = (
                            "ASSISTANT_INTERRUPT_TAIL"
                            if interrupt_grace_active
                            else "ASSISTANT_TAIL"
                        )
                    elif tail_samples > 0 and assistant_feed_allowed:
                        chunk = self._consume_tail_silence(hop_samples)
                        if listener_audio:
                            chunk_other, _, _ = self._pop_listener_other_hop(
                                hop_samples,
                            )
                        visible = True
                        mode = "ASSISTANT_TAIL"
                    else:
                        self._finish_tail_if_ready()

                if (
                    state in (self.ASSISTANT_ACTIVE, self.ASSISTANT_TAIL)
                    and not assistant_feed_allowed
                    and now - last_backpressure_log > 1.0
                ):
                    self.log(
                        "[ENGINE LIVE] render backpressure "
                        f"audio_inflight={len(audio_buf)} "
                        f"high_water={self.audio_inflight_high_water_samples} "
                        f"speaker_pending={speaker_samples}"
                    )
                    last_backpressure_log = now

                if chunk is not None:
                    if media_chunk is None:
                        media_chunk = chunk
                    if chunk_other is None:
                        chunk_other = np.zeros_like(chunk, dtype=np.float32)
                    elif len(chunk_other) != len(chunk):
                        if len(chunk_other) < len(chunk):
                            chunk_other = np.pad(chunk_other, (0, len(chunk) - len(chunk_other)))
                        else:
                            chunk_other = chunk_other[:len(chunk)]
                    chunk_other = np.ascontiguousarray(
                        chunk_other,
                        dtype=np.float32,
                    )
                    if listener_audio and (mode != last_listener_mode or now - last_listener_log > 2.0):
                        self_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-12))
                        other_rms = float(np.sqrt(np.mean(chunk_other.astype(np.float32) ** 2) + 1e-12))
                        self.log(
                            f"[ENGINE LISTENER] mode={mode} "
                            f"self_rms={self_rms:.5f} other_rms={other_rms:.5f} "
                            f"other_source={self._listener_other_source}"
                        )
                        last_listener_mode = mode
                        last_listener_log = now
                    item = {
                        "type": "audio",
                        "samples": chunk,
                        "samples_other": chunk_other,
                        "mode": mode,
                        "turn_id": turn_id,
                        "generation": stream_generation,
                        "visible": visible,
                    }
                    queued = False
                    stale_idle = False
                    put_error = None
                    if idle_candidate:
                        # Serialize the final idle revalidation and enqueue
                        # with begin_assistant_turn(). This prevents a stale
                        # 200 ms Listener hop from landing ahead of first TTS.
                        with self._assistant_lock:
                            stale_idle = not (
                                self._state == self.WARMUP_IDLE
                                and self._turn_id == turn_id
                                and self._stream_generation == stream_generation
                            )
                            if not stale_idle:
                                try:
                                    self.audio_q.put_nowait(item)
                                    queued = True
                                except Exception as e:
                                    put_error = e
                    else:
                        try:
                            self.audio_q.put_nowait(item)
                            queued = True
                        except Exception as e:
                            put_error = e
                    if put_error is not None and visible:
                        self.log(
                            f"[ENGINE WARN] audio_q put failed: {repr(put_error)}"
                        )
                    if stale_idle:
                        idle_segments_sent = max(0, idle_segments_sent - 1)
                        idle_segments_since_compact = max(
                            0,
                            idle_segments_since_compact - 1,
                        )
                        self.log(
                            "[ENGINE IDLE] stale hop skipped before assistant "
                            f"snapshot_turn={turn_id} "
                            f"snapshot_generation={stream_generation}"
                        )
                    if visible and queued:
                        audio_buf = np.concatenate(
                            [audio_buf, media_chunk]
                        ).astype(np.float32)
                        source_audio_frames = int(
                            np.ceil(len(media_chunk) / 640.0)
                        )
                        audio_frame_meta_buf.extend(
                            {
                                "generation": stream_generation,
                                "turn_id": turn_id,
                                "mode": mode,
                            }
                            for _ in range(source_audio_frames)
                        )
                    if queued and mode.startswith("ASSISTANT"):
                        self._mark_assistant_media_pending()
                        self._mark_pipeline_milestone(
                            turn_id,
                            "motion_submit",
                        )

                    next_feed_time += self.args.hop_ms / 1000.0
                    if next_feed_time < now - 0.2:
                        next_feed_time = now
                else:
                    next_feed_time = now + 0.005

            got_frames = []
            while True:
                try:
                    item = self.frame_q.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break
                frame_generation = item.get("generation", -1) if isinstance(item, dict) else -1
                frame_visible = bool(item.get("visible", False)) if isinstance(item, dict) else False
                frame_mode = str(item.get("mode", "")) if isinstance(item, dict) else ""
                frame_turn_id = int(item.get("turn_id", -1)) if isinstance(item, dict) else -1
                frame = omni.extract_frame(item)
                _, _, current_generation, _, _ = self._state_snapshot()
                if frame is not None:
                    self.latest_frame = frame
                    self.latest_frame_seq += 1
                    self.latest_frame_at = time.monotonic()
                    if (
                        frame_visible
                        and frame_generation == current_generation
                    ):
                        self.latest_visible_frame_seq += 1
                        self.latest_visible_frame_at = time.monotonic()
                        got_frames.append(frame)
                        if frame_mode.startswith("ASSISTANT"):
                            self._capture_assistant_frame(
                                frame_generation,
                                frame_turn_id,
                                frame,
                            )
                            self._mark_assistant_media_pending()
                            self._mark_pipeline_milestone(
                                frame_turn_id,
                                "rendered_frame",
                            )
                        else:
                            self._finish_assistant_frame_capture(
                                "assistant-media-ended"
                            )
            if got_frames:
                frames_buf.extend(got_frames)

            if (
                len(frames_buf) > self.max_frame_buf_frames
                or len(audio_buf) > self.max_audio_buf_samples
            ):
                dropped_frames = len(frames_buf)
                dropped_audio = len(audio_buf)
                frames_buf.clear()
                audio_buf = np.zeros(0, dtype=np.float32)
                audio_frame_meta_buf.clear()
                for client in list(self.media_clients):
                    client.request_stream_reset()
                if now - last_drop_log > 1.0:
                    self.log(
                        f"[ENGINE LIVE] visible backlog reset frames={dropped_frames} "
                        f"audio_samples={dropped_audio}"
                    )
                    last_drop_log = now

            while len(frames_buf) >= seg_frames and len(audio_buf) >= seg_samples:
                seg_f = np.stack(frames_buf[:seg_frames], axis=0).astype(np.uint8)
                seg_a = audio_buf[:seg_samples].astype(np.float32)
                seg_meta = list(audio_frame_meta_buf[:seg_frames])
                del frames_buf[:seg_frames]
                audio_buf = audio_buf[seg_samples:]
                del audio_frame_meta_buf[:seg_frames]

                clients = list(self.media_clients)
                segment_generation = (
                    int(seg_meta[0].get("generation", stream_generation))
                    if seg_meta
                    else stream_generation
                )
                for client in clients:
                    client.push_segment(
                        seg_f,
                        seg_a,
                        generation=segment_generation,
                        audio_frame_meta=seg_meta,
                    )

                seg_idx += 1
                if seg_idx <= 3 or seg_idx % 25 == 0:
                    self.log(
                        f"[ENGINE] produced raw segment={seg_idx} clients={len(clients)} "
                        f"frames_buf={len(frames_buf)} audio_buf={len(audio_buf)} "
                        f"audio_meta_buf={len(audio_frame_meta_buf)} "
                        f"generation={segment_generation}"
                    )

            time.sleep(0.005)

    def shutdown(self):
        self.running.clear()
        for client in list(self.media_clients):
            self.unregister_media_client(client)
        self.manager.stop()

    def health_snapshot(self):
        (
            state,
            turn_id,
            stream_generation,
            speaker_samples,
            tail_samples,
        ) = self._state_snapshot()
        workers = self.manager.health()
        frame_ready = self.latest_frame is not None
        now = time.monotonic()
        latest_frame_age_sec = (
            max(0.0, now - self.latest_frame_at)
            if self.latest_frame_at > 0.0
            else None
        )
        latest_visible_frame_age_sec = (
            max(0.0, now - self.latest_visible_frame_at)
            if self.latest_visible_frame_at > 0.0
            else None
        )
        launch_ready = False
        launch_token = ""
        launch_ready_path = Path(__file__).resolve().parent / "logs" / "demo_ready.pid"
        try:
            marker_fields = launch_ready_path.read_text(encoding="utf-8").split()
            if len(marker_fields) < 3:
                raise ValueError("invalid demo-ready marker")
            marker_pid, marker_version, launch_token = marker_fields[:3]
            launch_ready = (
                int(marker_pid) == os.getpid()
                and marker_version.strip() == ENGINE_VERSION
            )
        except (OSError, ValueError):
            pass
        healthy = (
            self.feed_thread.is_alive()
            and workers["motion_alive"]
            and workers["render_alive"]
            and frame_ready
        )
        return {
            "status": "ok" if healthy else "degraded",
            "engine_version": ENGINE_VERSION,
            "launch_ready": launch_ready,
            "launch_token": launch_token if launch_ready else "",
            "feed_thread_alive": self.feed_thread.is_alive(),
            "frame_ready": frame_ready,
            "latest_frame_seq": self.latest_frame_seq,
            "latest_frame_age_sec": latest_frame_age_sec,
            "latest_visible_frame_seq": self.latest_visible_frame_seq,
            "latest_visible_frame_age_sec": latest_visible_frame_age_sec,
            "state": state,
            "turn_id": turn_id,
            "stream_generation": stream_generation,
            "speaker_samples": speaker_samples,
            "tail_samples": tail_samples,
            "user_samples": self._user_samples,
            "listener_audio_enabled": self.listener_audio_enabled,
            "listener_virtual_only": self.listener_virtual_only,
            "listener_virtual_samples": len(self._listener_virtual_audio),
            "listener_other_source": self._listener_other_source,
            "media_clients": len(self.media_clients),
            "workers": workers,
            "time": time.time(),
        }


# -------------------------
# aiohttp server
# -------------------------


class ConversationLease:
    """One anonymous microphone owner with bounded stale-connection recovery."""

    def __init__(self, timeout_sec: float = 20.0):
        self.timeout_sec = max(1.0, float(timeout_sec))
        self._lock = asyncio.Lock()
        self._owner = None
        self._owner_client_id = ""
        self._epoch = 0
        self._last_seen = 0.0
        self._releasing = False
        self._aux_sockets: Set[Any] = set()

    @property
    def active(self) -> bool:
        return self._owner is not None

    def snapshot(self, client_id: str = "") -> Dict[str, Any]:
        active = self.active
        expires_in = (
            max(0.0, self.timeout_sec - (time.monotonic() - self._last_seen))
            if active
            else 0.0
        )
        return {
            "active": active,
            "owner_self": bool(
                active
                and client_id
                and client_id == self._owner_client_id
            ),
            "releasing": bool(active and self._releasing),
            "expires_in_sec": expires_in,
        }

    async def try_acquire(self, owner, client_id: str) -> Optional[int]:
        async with self._lock:
            if self._owner is not None:
                return None
            self._epoch += 1
            self._owner = owner
            self._owner_client_id = client_id
            self._last_seen = time.monotonic()
            self._releasing = False
            return self._epoch

    async def touch(self, owner, epoch: int) -> bool:
        async with self._lock:
            if self._owner is not owner or self._epoch != epoch or self._releasing:
                return False
            self._last_seen = time.monotonic()
            return True

    async def is_owner(self, owner, epoch: int) -> bool:
        async with self._lock:
            return self._owner is owner and self._epoch == epoch and not self._releasing

    async def is_client_owner(self, client_id: str) -> bool:
        async with self._lock:
            return bool(
                client_id
                and self._owner is not None
                and not self._releasing
                and client_id == self._owner_client_id
            )

    async def bind_aux(self, client_id: str, ws) -> bool:
        async with self._lock:
            if (
                self._owner is None
                or self._releasing
                or client_id != self._owner_client_id
            ):
                return False
            self._aux_sockets.add(ws)
            return True

    async def unbind_aux(self, ws) -> None:
        async with self._lock:
            self._aux_sockets.discard(ws)

    async def begin_release(self, owner, epoch: int) -> Optional[list]:
        async with self._lock:
            if self._owner is not owner or self._epoch != epoch or self._releasing:
                return None
            self._releasing = True
            sockets = list(self._aux_sockets)
            self._aux_sockets.clear()
            return sockets

    async def finish_release(self, owner, epoch: int) -> bool:
        async with self._lock:
            if self._owner is not owner or self._epoch != epoch:
                return False
            self._owner = None
            self._owner_client_id = ""
            self._last_seen = 0.0
            self._releasing = False
            self._aux_sockets.clear()
            return True


async def _watch_mic_lease(
    request: web.Request,
    ws: web.WebSocketResponse,
    epoch: int,
) -> None:
    lease: ConversationLease = request.app["conversation_lease"]
    interval = min(5.0, max(0.25, lease.timeout_sec / 3.0))
    while await lease.is_owner(ws, epoch):
        await asyncio.sleep(interval)
        snapshot = lease.snapshot()
        if not snapshot["active"] or snapshot["releasing"]:
            return
        if snapshot["expires_in_sec"] > 0:
            continue
        try:
            await asyncio.wait_for(
                ws.close(code=4008, message=b"microphone lease expired"),
                timeout=1.0,
            )
        except Exception:
            transport = request.transport
            if transport is not None:
                transport.abort()
        return


def _dialog_public_load(dialog_health: Dict[str, Any]) -> Dict[str, Any]:
    tts_active = bool(dialog_health.get("tts_active", False))
    active_requests = 0
    custom = dialog_health.get("custom_cascade")
    if isinstance(custom, dict):
        tts = custom.get("tts")
        if isinstance(tts, dict):
            try:
                active_requests = max(0, int(tts.get("active_requests", 0)))
            except (TypeError, ValueError):
                active_requests = 0
    return {
        "tts_active": tts_active,
        "active_requests": active_requests,
    }


_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def _request_client_id(request: web.Request) -> str:
    client_id = request.query.get("client_id", "").strip()
    if not _CLIENT_ID_RE.fullmatch(client_id):
        raise web.HTTPBadRequest(
            text="A valid client_id is required.",
            headers={"Cache-Control": "no-store"},
        )
    return client_id


def _is_loopback_request(request: web.Request) -> bool:
    if any(request.headers.get(name) for name in CUSTOMIZATION_FORWARDED_HEADERS):
        return False
    return bool(
        is_loopback_host(request.host)
        and is_loopback_host(request.remote or "")
    )


def _require_public_websocket_origin(request: web.Request) -> None:
    """Reject cross-origin browser sockets that could reuse the access cookie."""

    if _is_loopback_request(request):
        return
    origin = request.headers.get("Origin", "").strip().rstrip("/").lower()
    # Bind chat sockets to the page's current public host. This deliberately
    # does not reuse the customization-admin origin: Quick Tunnel hostnames can
    # rotate, and a stale admin setting must not lock ordinary chat out.
    expected = f"https://{request.host}".lower()
    if not origin or not hmac.compare_digest(origin, expected):
        raise web.HTTPForbidden(
            text="WebSocket origin is not allowed.",
            headers={"Cache-Control": "no-store"},
        )


def _customization_workload_active(app: web.Application) -> bool:
    prepare_lock = app.get("customization_prepare_lock")
    if prepare_lock is not None and prepare_lock.locked():
        return True
    activation_lock = app.get("customization_activation_start_lock")
    if activation_lock is not None and activation_lock.locked():
        return True
    activation_job = app.get("customization_activation_job")
    if activation_job:
        return True
    activation_state = app.get("customization_activation_state")
    if bool(
        isinstance(activation_state, dict)
        and activation_state.get("job_id")
    ):
        return True
    # The controller persists activation state across MSE restarts. Checking
    # it here closes the restart window where the in-memory locks are empty.
    return runtime_customization_workload_active(app)


async def _close_lease_sockets(sockets: list) -> None:
    if not sockets:
        return
    await asyncio.gather(
        *(
            ws.close(code=4403, message=b"microphone lease released")
            for ws in sockets
            if not ws.closed
        ),
        return_exceptions=True,
    )

async def index(request: web.Request):
    path = Path(__file__).resolve().parent / "static" / "index.html"
    return web.FileResponse(path)


async def close_open_websockets(app: web.Application) -> None:
    sockets = list(app.get("open_websockets", ()))
    if sockets:
        await asyncio.gather(
            *(
                ws.close(
                    code=aiohttp.WSCloseCode.GOING_AWAY,
                    message=b"server shutdown",
                )
                for ws in sockets
            ),
            return_exceptions=True,
        )


async def _send_media_ws_item(
    ws: web.WebSocketResponse,
    item,
    timeout: float,
) -> None:
    send = ws.send_bytes(item) if isinstance(item, bytes) else ws.send_str(str(item))
    await asyncio.wait_for(send, timeout=timeout)


async def _close_failed_media_ws(
    request: web.Request,
    ws: web.WebSocketResponse,
    timeout: float = 0.5,
) -> None:
    try:
        await asyncio.wait_for(
            ws.close(
                code=aiohttp.WSCloseCode.INTERNAL_ERROR,
                message=b"media encoder stopped",
                drain=False,
            ),
            timeout=timeout,
        )
    except Exception:
        transport = request.transport
        if transport is not None:
            transport.abort()


async def media_ws(request: web.Request):
    _require_public_websocket_origin(request)
    engine: RealtimeMSEEngine = request.app["engine"]
    diagnostic = _is_loopback_request(request) and not request.query.get("client_id")
    client_id = "" if diagnostic else _request_client_id(request)
    lease: Optional[ConversationLease] = request.app.get("conversation_lease")
    if not diagnostic and (
        lease is None or not await lease.is_client_owner(client_id)
    ):
        raise web.HTTPConflict(
            text="An active microphone lease is required.",
            headers={"Cache-Control": "no-store"},
        )
    try:
        send_timeout = max(
            0.1,
            float(os.getenv("MEDIA_WS_SEND_TIMEOUT_SEC", "2.0")),
        )
    except (TypeError, ValueError):
        send_timeout = 2.0
    ws = web.WebSocketResponse(
        max_msg_size=0,
        timeout=2.0,
        heartbeat=15.0,
        compress=False,
    )
    await ws.prepare(request)
    if not diagnostic and not await lease.bind_aux(client_id, ws):
        await ws.close(code=4403, message=b"microphone lease required")
        return ws
    open_websockets: set[web.WebSocketResponse] = request.app["open_websockets"]
    open_websockets.add(ws)
    client = None
    out_task = None
    recv_task = None
    stopped_task = None
    send_task = None
    pipeline_stopped = False
    try:
        client = engine.register_media_client()
        await _send_media_ws_item(
            ws,
            json.dumps({"type": "mime", "mime": 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"'}),
            send_timeout,
        )
        await _send_media_ws_item(
            ws,
            json.dumps({"type": "log", "message": "[MEDIA] connected pipe encoder"}),
            send_timeout,
        )
        current_turn = getattr(
            engine,
            "assistant_turn_control_snapshot",
            lambda: None,
        )()
        if current_turn is not None:
            await _send_media_ws_item(
                ws,
                json.dumps(current_turn, ensure_ascii=False),
                send_timeout,
            )
        out_task = asyncio.create_task(client.out_q.get())
        recv_task = asyncio.create_task(ws.receive())
        stopped_task = asyncio.create_task(client.stopped.wait())
        while True:
            wait_tasks = {recv_task, stopped_task}
            if send_task is None:
                wait_tasks.add(out_task)
            else:
                wait_tasks.add(send_task)
            done, _ = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopped_task in done:
                pipeline_stopped = True
                break
            if recv_task in done:
                msg = recv_task.result()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
                recv_task = asyncio.create_task(ws.receive())
            if send_task is not None and send_task in done:
                send_task.result()
                send_task = None
                out_task = asyncio.create_task(client.out_q.get())
            if send_task is None and out_task in done:
                item = client.filter_output_item(out_task.result())
                out_task = None
                if item is not None:
                    send_task = asyncio.create_task(
                        _send_media_ws_item(ws, item, send_timeout)
                    )
                else:
                    out_task = asyncio.create_task(client.out_q.get())
    except Exception as e:
        pipeline_stopped = True
        if client is not None:
            client._signal_stopped(
                f"media websocket failed: {type(e).__name__}"
            )
    finally:
        tasks = [
            task
            for task in (out_task, recv_task, stopped_task, send_task)
            if task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if client is not None:
            await asyncio.to_thread(engine.unregister_media_client, client)
            client = None
        if not diagnostic:
            await lease.unbind_aux(ws)
        open_websockets.discard(ws)
        if pipeline_stopped and not ws.closed:
            await _close_failed_media_ws(request, ws)
    return ws


async def logs_ws(request: web.Request):
    if not _is_loopback_request(request):
        raise web.HTTPForbidden(
            text="Runtime logs are available through the local SSH tunnel only.",
            headers={"Cache-Control": "no-store"},
        )
    engine: RealtimeMSEEngine = request.app["engine"]
    ws = web.WebSocketResponse(max_msg_size=0, timeout=2.0)
    await ws.prepare(request)
    open_websockets: set[web.WebSocketResponse] = request.app["open_websockets"]
    open_websockets.add(ws)
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    registered = False
    send_task = None
    recv_task = None
    try:
        engine.register_log_client(q)
        registered = True
        await ws.send_str(json.dumps({"type": "log", "message": "[LOG] connected"}))
        send_task = asyncio.create_task(q.get())
        recv_task = asyncio.create_task(ws.receive())
        while True:
            done, _ = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if recv_task in done:
                message = recv_task.result()
                if message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
                recv_task = asyncio.create_task(ws.receive())
            if send_task in done:
                await ws.send_str(json.dumps(send_task.result(), ensure_ascii=False))
                send_task = asyncio.create_task(q.get())
    except Exception:
        pass
    finally:
        tasks = [task for task in (send_task, recv_task) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if registered:
            engine.unregister_log_client(q)
        open_websockets.discard(ws)
    return ws


async def mic_ws(request: web.Request):
    _require_public_websocket_origin(request)
    engine: RealtimeMSEEngine = request.app["engine"]
    client_id = (
        f"localdiagnostic{uuid.uuid4().hex}"
        if _is_loopback_request(request) and not request.query.get("client_id")
        else _request_client_id(request)
    )
    backend_name: str = request.app["dialog_backend_name"]
    proto: Optional[DoubaoRealtimeProtocol] = request.app.get("doubao")
    shared_live = request.app.get("dialog_session")
    lease: ConversationLease = request.app["conversation_lease"]
    admission_lock: asyncio.Lock = request.app["workload_admission_lock"]
    owner_lock: asyncio.Lock = request.app["mic_owner_lock"]
    input_lock: asyncio.Lock = request.app["mic_input_lock"]
    ws = web.WebSocketResponse(
        max_msg_size=0,
        timeout=2.0,
        heartbeat=min(10.0, max(1.0, lease.timeout_sec / 2.0)),
    )
    await ws.prepare(request)
    open_websockets: set[web.WebSocketResponse] = request.app["open_websockets"]
    open_websockets.add(ws)

    instruction = "请用自然口语直接回答，默认一到三句；只有我明确要求时才详细展开。"
    live: Any = shared_live
    lease_epoch: Optional[int] = None
    watchdog_task: Optional[asyncio.Task] = None

    async def ensure_live_session():
        nonlocal live
        if backend_name == "pipecat":
            if live is None or live.closed.is_set() or live.task is None or live.task.done():
                raise RuntimeError("Pipecat speech provider is not running")
            return live
        if live is not None and not live.closed.is_set() and (live.task is None or not live.task.done()):
            return live
        if live is not None:
            engine.log("[MIC] live Doubao session is closed/done; starting a new one")
        if proto is None:
            raise RuntimeError("Doubao protocol is not initialized")
        live = DoubaoLiveSession(
            proto,
            instruction,
            engine.begin_assistant_turn,
            engine.enqueue_speaker_audio,
            engine.end_assistant_turn,
            engine.log,
        )
        live.start()
        engine.log("[MIC] live Doubao session started")
        return live

    try:
        async with admission_lock:
            customization_busy = _customization_workload_active(request.app)
            lease_epoch = (
                None
                if customization_busy
                else await lease.try_acquire(ws, client_id)
            )
        if lease_epoch is None:
            await ws.send_str(json.dumps({
                "type": "service_busy" if customization_busy else "lease_busy",
                "reason": "customizing" if customization_busy else "conversation_busy",
                "retry_after_ms": 1000,
            }))
            await ws.close(code=4409, message=b"conversation busy")
            return ws

        async with owner_lock:
            request.app["active_mic_ws"] = ws

        if backend_name == "pipecat" and live is not None:
            async with input_lock:
                await live.reset_dialog()

        await ws.send_str(json.dumps({
            "type": "lease_granted",
            "heartbeat_ms": 5000,
            "timeout_ms": round(lease.timeout_sec * 1000),
        }))
        watchdog_task = asyncio.create_task(
            _watch_mic_lease(request, ws, lease_epoch),
            name=f"mic-lease-watchdog-{lease_epoch}",
        )

        async for msg in ws:
            if not await lease.touch(ws, lease_epoch):
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    obj = json.loads(msg.data)
                except Exception:
                    obj = {}
                typ = obj.get("type")
                if typ == "config":
                    instruction = obj.get("instruction") or instruction
                    await ensure_live_session()
                    await ws.send_str(json.dumps({
                        "type": "ok",
                        "message": f"{backend_name} session running",
                    }, ensure_ascii=False))
                elif typ == "interrupt":
                    async with input_lock:
                        if not await lease.is_owner(ws, lease_epoch):
                            break
                        if live is not None:
                            dropped = await live.interrupt()
                        else:
                            dropped = engine.interrupt_assistant()
                        engine.log(
                            f"[MIC] interrupt; dropped assistant audio chunks={dropped}"
                        )
                elif typ in {"ping", "lease_heartbeat"}:
                    await ws.send_str(json.dumps({"type": "pong"}))
            elif msg.type == aiohttp.WSMsgType.BINARY:
                async with input_lock:
                    if not await lease.is_owner(ws, lease_epoch):
                        break
                    engine.enqueue_user_audio(msg.data)
                    live = await ensure_live_session()
                    sent = await live.send_audio(msg.data)
                    if not sent:
                        if backend_name == "pipecat":
                            raise RuntimeError("Pipecat speech provider stopped")
                        live = None
                        live = await ensure_live_session()
                        await live.send_audio(msg.data)
    finally:
        async def finish_microphone_cleanup() -> None:
            # Begin release before any other cancellable cleanup. The request
            # task itself can be cancelled as part of the WebSocket close.
            release_sockets = (
                await lease.begin_release(ws, lease_epoch)
                if lease_epoch is not None
                else None
            )
            if watchdog_task is not None:
                watchdog_task.cancel()
                await asyncio.gather(watchdog_task, return_exceptions=True)
            try:
                if release_sockets is not None:
                    try:
                        if backend_name == "pipecat" and live is not None:
                            async with input_lock:
                                dropped = await live.interrupt()
                                await live.reset_dialog()
                            engine.log(
                                "[MIC] active socket disconnected; "
                                f"dropped assistant audio chunks={dropped}"
                            )
                    finally:
                        await _close_lease_sockets(release_sockets)
                        await lease.finish_release(ws, lease_epoch)
                        async with owner_lock:
                            if request.app.get("active_mic_ws") is ws:
                                request.app["active_mic_ws"] = None
                if live is not None and backend_name == "doubao":
                    await live.close()
            finally:
                open_websockets.discard(ws)

        cleanup_task = asyncio.create_task(
            finish_microphone_cleanup(),
            name=f"mic-lease-cleanup-{lease_epoch or 'unclaimed'}",
        )
        cleanup_tasks: Set[asyncio.Task] = request.app["mic_cleanup_tasks"]
        cleanup_tasks.add(cleanup_task)

        def cleanup_finished(task: asyncio.Task) -> None:
            cleanup_tasks.discard(task)
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                engine.log(f"[MIC WARN] lease cleanup failed: {exc!r}")

        cleanup_task.add_done_callback(cleanup_finished)
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            # The shield keeps cleanup alive. Waiting once more makes normal
            # socket closes deterministic; the app also tracks it for shutdown.
            await asyncio.shield(cleanup_task)
            raise
    return ws


async def health(request: web.Request):
    if not _is_loopback_request(request):
        payload = _realtime_status_payload(request, require_client_id=False)
        return web.json_response(
            payload,
            status=200 if payload["service_ready"] else 503,
            headers={"Cache-Control": "no-store"},
        )

    engine: RealtimeMSEEngine = request.app["engine"]
    snapshot = engine.health_snapshot()
    snapshot["dialog_backend"] = request.app["dialog_backend_name"]
    dialog_session = request.app.get("dialog_session")
    if dialog_session is not None:
        snapshot["dialog_session"] = dialog_session.health_snapshot()
        if not snapshot["dialog_session"]["ready"]:
            snapshot["status"] = "degraded"
    status = 200 if snapshot["status"] == "ok" else 503
    return web.json_response(snapshot, status=status)


def _realtime_status_payload(
    request: web.Request,
    *,
    require_client_id: bool = True,
) -> Dict[str, Any]:
    engine: RealtimeMSEEngine = request.app["engine"]
    lease: ConversationLease = request.app["conversation_lease"]
    client_id = (
        _request_client_id(request)
        if require_client_id
        else request.query.get("client_id", "").strip()
    )
    if client_id and not _CLIENT_ID_RE.fullmatch(client_id):
        client_id = ""
    dialog_session = request.app.get("dialog_session")
    try:
        engine_health = engine.health_snapshot()
        service_ready = engine_health.get("status") == "ok"
    except Exception:
        service_ready = False

    dialog_health: Dict[str, Any] = {}
    if dialog_session is not None:
        try:
            dialog_health = dialog_session.health_snapshot()
            service_ready = service_ready and bool(dialog_health.get("ready"))
        except Exception:
            service_ready = False

    lease_snapshot = lease.snapshot(client_id)
    conversation_active = bool(lease_snapshot["active"])
    customization_busy = _customization_workload_active(request.app)
    conversation_state = (
        "unavailable"
        if not service_ready
        else "unavailable" if customization_busy
        else "yours" if lease_snapshot["owner_self"]
        else "busy" if conversation_active
        else "available"
    )
    return {
        "service_ready": service_ready,
        "phase": (
            "unavailable"
            if not service_ready
            else "customizing" if customization_busy
            else "ready"
        ),
        "conversation": {
            "state": conversation_state,
            "active": conversation_active,
            "capacity": 1,
            "available": service_ready and not customization_busy and (
                not conversation_active or lease_snapshot["owner_self"]
            ),
        },
        "media_clients": len(engine.media_clients),
        "speech": _dialog_public_load(dialog_health),
        "updated_at": time.time(),
        "poll_after_ms": 2000,
    }


async def realtime_status(request: web.Request):
    """Return only coarse public capacity; never expose provider internals."""

    payload = _realtime_status_payload(
        request,
        require_client_id=not _is_loopback_request(request),
    )
    return web.json_response(
        payload,
        headers={"Cache-Control": "no-store"},
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=6008)
    p.add_argument("--sample", type=int, default=1)
    p.add_argument("--hop_ms", type=int, default=200)
    p.add_argument("--denoising_steps", type=int, default=1)
    p.add_argument("--motion_gpu", type=int, default=0)
    p.add_argument("--render_gpu", type=int, default=1)
    p.add_argument("--feature_lag_frames", type=int, default=3)
    p.add_argument("--segment_frames", type=int, default=4, help="visible source frames per push; 4 frames = 0.16s and aligns with PIPE_FRAME_STRIDE=2")
    return p.parse_args()


@web.middleware
async def public_access_gate(request: web.Request, handler: Any):
    if is_customization_path(request.path):
        return await handler(request)
    expected_token = os.getenv("PUBLIC_ACCESS_TOKEN", "").strip()
    if not expected_token or _is_loopback_request(request):
        return await handler(request)

    query_token = request.query.get("access_token")
    if token_matches(query_token, expected_token):
        clean_query = [
            (key, value)
            for key, value in request.rel_url.query.items()
            if key != "access_token"
        ]
        location = str(request.rel_url.with_query(clean_query)) or "/"
        response = web.HTTPFound(location=location)
        response.set_cookie(
            ACCESS_COOKIE_NAME,
            expected_token,
            secure=True,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    if token_matches(request.cookies.get(ACCESS_COOKIE_NAME), expected_token):
        return await handler(request)

    return web.Response(
        status=401,
        text="Public access token is required.",
        headers={"Cache-Control": "no-store"},
    )


def build_app(args):
    app = web.Application(
        client_max_size=MAX_REQUEST_BYTES,
        middlewares=[public_access_gate],
    )
    app["open_websockets"] = set()
    app["workload_admission_lock"] = asyncio.Lock()

    async def on_startup(app_obj: web.Application):
        # Important: use the actual aiohttp running loop, not a stale loop.
        loop = asyncio.get_running_loop()
        engine = RealtimeMSEEngine(args, loop)
        backend_name = os.getenv("DIALOG_BACKEND", "doubao").strip().lower()
        if backend_name not in {"doubao", "pipecat"}:
            raise RuntimeError(f"unsupported DIALOG_BACKEND={backend_name!r}")
        app_obj["engine"] = engine
        app_obj["dialog_backend_name"] = backend_name
        app_obj["mic_owner_lock"] = asyncio.Lock()
        app_obj["mic_input_lock"] = asyncio.Lock()
        app_obj["mic_cleanup_tasks"] = set()
        app_obj["conversation_lease"] = ConversationLease(
            float(os.getenv("MIC_LEASE_TIMEOUT_SEC", "20"))
        )
        app_obj["active_mic_ws"] = None
        if backend_name == "pipecat":
            from pipecat_dystream.mse_session import PipecatMSESession

            dialog_session = PipecatMSESession(engine, engine.log)
            app_obj["dialog_session"] = dialog_session
            dialog_session.start()
            await dialog_session.wait_ready()
        else:
            app_obj["doubao"] = DoubaoRealtimeProtocol(engine.log)
        print(
            f"[SERVER] engine initialized backend={backend_name} loop id={id(loop)}",
            flush=True,
        )

    async def on_cleanup(app_obj: web.Application):
        cleanup_tasks = list(app_obj.get("mic_cleanup_tasks", ()))
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        dialog_session = app_obj.get("dialog_session")
        if dialog_session is not None:
            await dialog_session.close()
        engine = app_obj.get("engine")
        if engine is not None:
            engine.shutdown()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(close_open_websockets)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/api/realtime/status", realtime_status)
    app.router.add_get("/ws/media", media_ws)
    app.router.add_get("/ws/logs", logs_ws)
    app.router.add_get("/ws/mic", mic_ws)
    app.router.add_static("/static", str(Path(__file__).resolve().parent / "static"), show_index=False)
    register_customization_routes(app)
    return app


def main():
    args = parse_args()
    app = build_app(args)
    bind_host = os.getenv("SERVER_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    print(f"[SERVER] http://{bind_host}:{args.port}", flush=True)
    web.run_app(
        app,
        host=bind_host,
        port=args.port,
        access_log_class=RedactedAccessLogger,
    )


if __name__ == "__main__":
    main()
