#!/usr/bin/env python3
"""Credential-safe Qwen-Audio Realtime handshake and optional speech probe."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import threading
import time
import wave
from pathlib import Path
from urllib.parse import quote

import websockets


MODEL = "qwen-audio-3.0-realtime-flash"
LEGACY_PUBLIC_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


def _api_key() -> str:
    for name in (
        "PIPECAT_S2S_API_KEY",
        "DASHSCOPE_API_KEY",
        "PIPECAT_LLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError("missing Qwen-Audio API key")


def _base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url
    if args.legacy_public_endpoint:
        return LEGACY_PUBLIC_ENDPOINT
    configured = os.getenv("PIPECAT_S2S_BASE_URL", "").strip()
    if configured:
        return configured
    workspace_id = os.getenv("PIPECAT_S2S_WORKSPACE_ID", "").strip()
    if workspace_id:
        return (
            f"wss://{workspace_id}.cn-beijing.maas.aliyuncs.com"
            "/api-ws/v1/realtime"
        )
    raise RuntimeError(
        "missing PIPECAT_S2S_WORKSPACE_ID or PIPECAT_S2S_BASE_URL"
    )


def _url(base_url: str, model: str) -> str:
    if "model=" in base_url:
        return base_url
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}model={quote(model, safe='')}"


def _read_pcm16_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        actual = (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
        )
        expected = (1, 2, 16_000)
        if actual != expected:
            raise RuntimeError(
                "probe WAV must be PCM16 mono 16 kHz; "
                f"got channels={actual[0]} width={actual[1]} rate={actual[2]}"
            )
        return reader.readframes(reader.getnframes())


def _write_pcm16_wav(path: Path, pcm: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(pcm)


def _synthesize_probe_input(text: str, api_key: str, timeout_sec: float) -> bytes:
    import dashscope
    from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime

    class Callback:
        def __init__(self):
            self.audio = bytearray()
            self.done = threading.Event()
            self.error: str | None = None

        def on_open(self):
            return None

        def on_close(self, close_status_code, close_msg):
            if not self.done.is_set():
                self.error = (
                    "TTS connection closed before response.done: "
                    f"code={close_status_code} message={close_msg}"
                )
                self.done.set()

        def on_event(self, event):
            event_type = event.get("type")
            if event_type == "response.audio.delta":
                self.audio.extend(base64.b64decode(event.get("delta", "")))
            elif event_type == "response.done":
                response = event.get("response", {})
                if response.get("status") != "completed":
                    self.error = f"TTS response status={response.get('status')}"
                self.done.set()
            elif event_type == "error":
                error = event.get("error", {})
                self.error = (
                    "TTS provider error: "
                    f"type={error.get('type')} code={error.get('code')} "
                    f"message={error.get('message')}"
                )
                self.done.set()

    dashscope.api_key = api_key
    callback = Callback()
    provider = QwenTtsRealtime(
        model="qwen3-tts-flash-realtime",
        callback=callback,
    )
    try:
        provider.connect()
        provider.update_session(
            voice="Cherry",
            mode="commit",
            sample_rate=16_000,
            audio_format="pcm",
            language_type="Chinese",
        )
        provider.append_text(text)
        provider.commit()
        if not callback.done.wait(timeout_sec):
            raise TimeoutError("fixed-text TTS probe input timed out")
        if callback.error:
            raise RuntimeError(callback.error)
        if not callback.audio:
            raise RuntimeError("fixed-text TTS returned no audio")
        return bytes(callback.audio)
    finally:
        try:
            provider.close()
        except Exception:
            pass


async def _wait_for_session(websocket, timeout_sec: float):
    async with asyncio.timeout(timeout_sec):
        async for message in websocket:
            event = json.loads(message)
            event_type = event.get("type")
            if event_type == "session.updated":
                return
            if event_type == "error":
                error = event.get("error", {})
                raise RuntimeError(
                    "provider error during handshake: "
                    f"type={error.get('type')} code={error.get('code')} "
                    f"param={error.get('param')} message={error.get('message')}"
                )
    raise RuntimeError("connection closed before session.updated")


async def _receive_response(websocket, timeout_sec: float) -> tuple[bytes, dict]:
    audio = bytearray()
    metrics: dict[str, float | str | None] = {
        "speech_stopped_at": None,
        "response_created_at": None,
        "first_audio_at": None,
        "status": None,
    }
    started_at = time.monotonic()
    async with asyncio.timeout(timeout_sec):
        async for message in websocket:
            now = time.monotonic()
            event = json.loads(message)
            event_type = event.get("type")
            if event_type == "input_audio_buffer.speech_stopped":
                metrics["speech_stopped_at"] = now
            elif event_type == "response.created":
                metrics["response_created_at"] = now
            elif event_type == "response.audio.delta":
                if metrics["first_audio_at"] is None:
                    metrics["first_audio_at"] = now
                audio.extend(base64.b64decode(event.get("delta", "")))
            elif event_type == "response.done":
                response = event.get("response", {})
                metrics["status"] = response.get("status")
                metrics["elapsed_ms"] = (now - started_at) * 1000.0
                speech_stopped_at = metrics["speech_stopped_at"]
                first_audio_at = metrics["first_audio_at"]
                metrics["speech_to_first_audio_ms"] = (
                    (first_audio_at - speech_stopped_at) * 1000.0
                    if isinstance(speech_stopped_at, float)
                    and isinstance(first_audio_at, float)
                    else None
                )
                return bytes(audio), metrics
            elif event_type == "error":
                error = event.get("error", {})
                raise RuntimeError(
                    "provider error during response: "
                    f"type={error.get('type')} code={error.get('code')} "
                    f"param={error.get('param')} message={error.get('message')}"
                )
    raise RuntimeError("connection closed before response.done")


async def _run(args: argparse.Namespace):
    model = args.model or os.getenv("PIPECAT_S2S_MODEL", MODEL).strip() or MODEL
    voice = (
        args.voice
        or os.getenv("PIPECAT_S2S_VOICE", "longanqian").strip()
        or "longanqian"
    )
    url = _url(_base_url(args), model)
    api_key = _api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    async with websockets.connect(
        url,
        additional_headers=headers,
        open_timeout=args.timeout,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "voice": voice,
                        "input_audio_format": "pcm",
                        "output_audio_format": "pcm",
                        "turn_detection": {"type": "smart_turn"},
                    },
                }
            )
        )
        await _wait_for_session(websocket, args.timeout)
        print(f"S2S_HANDSHAKE_OK model={model} voice={voice}", flush=True)
        if args.wav is None and not args.synthesize_input_text:
            return

        if args.wav is not None:
            pcm = _read_pcm16_wav(args.wav)
        else:
            pcm = await asyncio.to_thread(
                _synthesize_probe_input,
                args.synthesize_input_text,
                api_key,
                args.response_timeout,
            )
            print(
                "S2S_PROBE_INPUT_OK "
                f"source=fixed_text bytes={len(pcm)}",
                flush=True,
            )
        chunk_bytes = 3_200
        response_task = asyncio.create_task(
            _receive_response(websocket, args.response_timeout)
        )
        try:
            for offset in range(0, len(pcm), chunk_bytes):
                chunk = pcm[offset : offset + chunk_bytes]
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
                await asyncio.sleep(len(chunk) / 32_000.0)
            silence = bytes(chunk_bytes)
            for _ in range(15):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(silence).decode("ascii"),
                        }
                    )
                )
                await asyncio.sleep(0.1)
            response_pcm, metrics = await response_task
        except Exception:
            response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)
            raise
        if metrics["status"] != "completed":
            raise RuntimeError(f"response ended with status={metrics['status']}")
        if not response_pcm:
            raise RuntimeError("response completed without audio")
        if args.output is not None:
            _write_pcm16_wav(args.output, response_pcm)
        print(
            "S2S_RESPONSE_OK "
            f"bytes={len(response_pcm)} "
            f"elapsed_ms={metrics['elapsed_ms']:.1f} "
            f"speech_to_first_audio_ms={metrics['speech_to_first_audio_ms']}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--voice", default="")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--wav", type=Path)
    input_group.add_argument("--synthesize-input-text", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--response-timeout", type=float, default=45.0)
    parser.add_argument(
        "--legacy-public-endpoint",
        action="store_true",
        help="diagnostic only; production Qwen-Audio 3.0 uses a MAAS workspace URL",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except Exception as exc:
        print(f"S2S_PROBE_FAILED {type(exc).__name__}: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
