#!/usr/bin/env python3
"""Probe the local VoxCPM2 bridge and save its PCM stream as a WAV file."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect


def _parse_event(message: Any) -> dict[str, Any]:
    if not isinstance(message, str):
        raise RuntimeError("expected a JSON control message")
    event = json.loads(message)
    if not isinstance(event, dict):
        raise RuntimeError("control message must be a JSON object")
    return event


async def _probe(url: str, text: str, output: Path, timeout_s: float) -> None:
    request_id = uuid.uuid4().hex
    probe_started = time.perf_counter()
    request_sent: float | None = None
    start_received: float | None = None
    first_pcm_received: float | None = None
    done_received: float | None = None
    sample_rate = 0
    pcm_chunks: list[bytes] = []

    async with asyncio.timeout(timeout_s):
        async with connect(
            url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            connected = time.perf_counter()
            await websocket.send(
                json.dumps(
                    {"type": "synthesize", "request_id": request_id, "text": text}
                )
            )
            request_sent = time.perf_counter()

            async for message in websocket:
                received = time.perf_counter()
                if isinstance(message, bytes):
                    if start_received is None:
                        raise RuntimeError("PCM arrived before start metadata")
                    if not message or len(message) % 2:
                        raise RuntimeError("bridge returned invalid PCM16 data")
                    if first_pcm_received is None:
                        first_pcm_received = received
                    pcm_chunks.append(message)
                    continue

                event = _parse_event(message)
                if str(event.get("request_id", "")) not in {"", request_id}:
                    continue
                event_type = event.get("type")
                if event_type == "start":
                    if start_received is not None:
                        raise RuntimeError("bridge sent duplicate start metadata")
                    sample_rate = int(event.get("sample_rate", 0))
                    if (
                        sample_rate <= 0
                        or int(event.get("channels", 0)) != 1
                        or int(event.get("sample_width", 0)) != 2
                        or event.get("audio_format") != "pcm_s16le"
                    ):
                        raise RuntimeError(f"invalid stream metadata: {event}")
                    start_received = received
                elif event_type == "done":
                    if event.get("status") != "completed":
                        raise RuntimeError(f"synthesis did not complete: {event}")
                    done_received = received
                    break
                elif event_type == "error":
                    raise RuntimeError(str(event.get("error", "unknown bridge error")))

    if request_sent is None or start_received is None:
        raise RuntimeError("bridge closed before start metadata")
    if first_pcm_received is None or done_received is None or not pcm_chunks:
        raise RuntimeError("bridge closed before PCM completion")

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(pcm_chunks))

    pcm_bytes = sum(len(chunk) for chunk in pcm_chunks)
    audio_seconds = pcm_bytes / (sample_rate * 2)
    print(
        "VOXCPM2_PROBE_OK "
        f"connect_ms={(connected - probe_started) * 1000:.1f} "
        f"request_to_start_ms={(start_received - request_sent) * 1000:.1f} "
        f"request_to_first_pcm_ms={(first_pcm_received - request_sent) * 1000:.1f} "
        f"start_to_first_pcm_ms={(first_pcm_received - start_received) * 1000:.1f} "
        f"request_to_done_ms={(done_received - request_sent) * 1000:.1f} "
        f"sample_rate={sample_rate} audio_seconds={audio_seconds:.3f} "
        f"pcm_bytes={pcm_bytes} output={output.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the local VoxCPM2 PCM bridge")
    parser.add_argument("--url", default="ws://127.0.0.1:8770")
    parser.add_argument("--text", default="你好，这是一段实时语音合成测试。")
    parser.add_argument("--output", type=Path, default=Path("voxcpm2_probe.wav"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.set_defaults(
        text="\u4f60\u597d\uff0c\u8fd9\u662f\u4e00\u6bb5\u5b9e\u65f6\u8bed\u97f3\u5408\u6210\u6d4b\u8bd5\u3002"
    )
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    asyncio.run(_probe(args.url, args.text.strip(), args.output, args.timeout))


if __name__ == "__main__":
    main()
