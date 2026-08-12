#!/usr/bin/env python3
"""Exercise the browser WebSocket path with a real WAV for multiple turns."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.client import connect


TARGET_RATE = 16_000
CHUNK_SAMPLES = 640


def _load_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    if channels != 1 or sample_width != 2 or source_rate <= 0:
        raise ValueError("input must be mono PCM16 WAV with a valid sample rate")
    source = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if source.size == 0:
        raise ValueError("input WAV is empty")
    if source_rate == TARGET_RATE:
        return source.astype("<i2")
    output_size = max(1, int(round(source.size * TARGET_RATE / source_rate)))
    positions = np.arange(output_size, dtype=np.float64) * source_rate / TARGET_RATE
    resampled = np.interp(positions, np.arange(source.size), source)
    return np.clip(resampled, -32768, 32767).astype("<i2")


def _milliseconds(event_at: float | None, voice_stop_at: float) -> float | None:
    if event_at is None:
        return None
    return round((event_at - voice_stop_at) * 1000.0, 1)


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    pcm16 = _load_pcm16(args.input_wav)
    base = args.base.rstrip("/")
    events: asyncio.Queue[tuple[float, str, Any]] = asyncio.Queue()
    total_media_bytes = 0

    async with (
        connect(base + "/ws/media", max_size=None) as media,
        connect(base + "/ws/logs", max_size=None) as logs,
        connect(base + "/ws/mic", max_size=None) as mic,
    ):
        async def read_media() -> None:
            nonlocal total_media_bytes
            async for message in media:
                received_at = time.perf_counter()
                if isinstance(message, bytes):
                    total_media_bytes += len(message)
                    continue
                try:
                    event = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    continue
                kind = event.get("type")
                if kind in {
                    "assistant_turn_started",
                    "assistant_media_boundary",
                    "assistant_media_ended",
                }:
                    await events.put((received_at, str(kind), event))

        async def read_logs() -> None:
            async for message in logs:
                received_at = time.perf_counter()
                try:
                    event = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    continue
                value = str(event.get("message", ""))
                if "[PIPECAT PROVIDER] first audio" in value:
                    await events.put((received_at, "provider_first_audio", value))
                elif "[PIPECAT PROVIDER] TTS turn ended" in value:
                    await events.put((received_at, "provider_tts_ended", value))
                elif "[PIPECAT PROVIDER ERROR]" in value:
                    await events.put((received_at, "provider_error", value))

        media_task = asyncio.create_task(read_media())
        log_task = asyncio.create_task(read_logs())
        try:
            await mic.send(
                json.dumps(
                    {"type": "config", "instruction": args.instruction},
                    ensure_ascii=True,
                )
            )
            acknowledgement = await asyncio.wait_for(mic.recv(), timeout=5.0)
            if "session running" not in str(acknowledgement):
                raise RuntimeError(f"unexpected microphone acknowledgement: {acknowledgement}")

            results: list[dict[str, Any]] = []
            silence = np.zeros(CHUNK_SAMPLES, dtype="<i2").tobytes()
            for turn_index in range(1, args.turns + 1):
                while not events.empty():
                    events.get_nowait()
                media_before = total_media_bytes
                voice_stop_at: float | None = None
                for offset in range(0, pcm16.size, CHUNK_SAMPLES):
                    chunk = pcm16[offset : offset + CHUNK_SAMPLES]
                    if chunk.size < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.size))
                    await mic.send(chunk.astype("<i2").tobytes())
                    if int(np.max(np.abs(chunk.astype(np.int32)))) >= args.voice_threshold:
                        voice_stop_at = time.perf_counter()
                    await asyncio.sleep(CHUNK_SAMPLES / TARGET_RATE)
                if voice_stop_at is None:
                    raise RuntimeError("input WAV has no voiced chunks at the threshold")
                for _ in range(args.silence_chunks):
                    await mic.send(silence)
                    await asyncio.sleep(CHUNK_SAMPLES / TARGET_RATE)

                marks: dict[str, float] = {}
                provider_logs: list[str] = []
                deadline = time.monotonic() + args.timeout
                while "assistant_media_ended" not in marks:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"turn {turn_index} did not reach media end")
                    received_at, kind, payload = await asyncio.wait_for(
                        events.get(), timeout=remaining
                    )
                    if kind == "provider_error":
                        raise RuntimeError(str(payload))
                    marks.setdefault(kind, received_at)
                    if kind.startswith("provider_"):
                        provider_logs.append(str(payload))

                results.append(
                    {
                        "turn": turn_index,
                        "input_seconds": round(pcm16.size / TARGET_RATE, 3),
                        "voice_stop_to_provider_first_audio_ms": _milliseconds(
                            marks.get("provider_first_audio"), voice_stop_at
                        ),
                        "voice_stop_to_assistant_turn_started_ms": _milliseconds(
                            marks.get("assistant_turn_started"), voice_stop_at
                        ),
                        "voice_stop_to_media_boundary_ms": _milliseconds(
                            marks.get("assistant_media_boundary"), voice_stop_at
                        ),
                        "voice_stop_to_media_ended_ms": _milliseconds(
                            marks.get("assistant_media_ended"), voice_stop_at
                        ),
                        "media_bytes": total_media_bytes - media_before,
                        "provider_logs": provider_logs,
                    }
                )
                await asyncio.sleep(args.between_turns)
            return results
        finally:
            for task in (media_task, log_task):
                task.cancel()
            await asyncio.gather(media_task, log_task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="ws://127.0.0.1:7860")
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--instruction", default="Answer briefly in Chinese.")
    parser.add_argument("--voice-threshold", type=int, default=500)
    parser.add_argument("--silence-chunks", type=int, default=30)
    parser.add_argument("--between-turns", type=float, default=0.5)
    args = parser.parse_args()
    if args.turns <= 0 or args.timeout <= 0 or args.silence_chunks <= 0:
        parser.error("turns, timeout and silence-chunks must be positive")
    results = asyncio.run(_run(args))
    print("CUSTOM_E2E_PROBE_OK " + json.dumps(results, ensure_ascii=True))


if __name__ == "__main__":
    main()
