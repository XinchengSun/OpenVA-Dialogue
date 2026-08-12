#!/usr/bin/env python3
"""Barge into a live custom-cascade response and verify a second turn completes."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.client import connect

from probe_custom_e2e import CHUNK_SAMPLES, TARGET_RATE, _load_pcm16, _milliseconds


async def _send_audio(
    websocket: Any,
    pcm16: np.ndarray,
    *,
    voice_threshold: int,
    silence_chunks: int,
) -> float:
    voice_stop_at: float | None = None
    for offset in range(0, pcm16.size, CHUNK_SAMPLES):
        chunk = pcm16[offset : offset + CHUNK_SAMPLES]
        if chunk.size < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.size))
        await websocket.send(chunk.astype("<i2").tobytes())
        if int(np.max(np.abs(chunk.astype(np.int32)))) >= voice_threshold:
            voice_stop_at = time.perf_counter()
        await asyncio.sleep(CHUNK_SAMPLES / TARGET_RATE)
    if voice_stop_at is None:
        raise RuntimeError("input WAV has no voiced chunks at the threshold")
    silence = np.zeros(CHUNK_SAMPLES, dtype="<i2").tobytes()
    for _ in range(silence_chunks):
        await websocket.send(silence)
        await asyncio.sleep(CHUNK_SAMPLES / TARGET_RATE)
    return voice_stop_at


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    pcm16 = _load_pcm16(args.input_wav)
    base = args.base.rstrip("/")
    events: asyncio.Queue[tuple[float, str, Any]] = asyncio.Queue()
    media_bytes = 0

    async with (
        connect(base + "/ws/media", max_size=None) as media,
        connect(base + "/ws/logs", max_size=None) as logs,
        connect(base + "/ws/mic", max_size=None) as mic,
    ):
        async def read_media() -> None:
            nonlocal media_bytes
            async for message in media:
                received_at = time.perf_counter()
                if isinstance(message, bytes):
                    media_bytes += len(message)
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
                if "pipeline interruption reset assistant" in value:
                    await events.put((received_at, "pipeline_interruption", value))
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

            first_voice_stop = await _send_audio(
                mic,
                pcm16,
                voice_threshold=args.voice_threshold,
                silence_chunks=args.silence_chunks,
            )
            first_started_at: float | None = None
            first_turn_id: int | None = None
            deadline = time.monotonic() + args.timeout
            while first_turn_id is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("first response never started")
                received_at, kind, payload = await asyncio.wait_for(
                    events.get(), timeout=remaining
                )
                if kind == "provider_error":
                    raise RuntimeError(str(payload))
                if kind == "assistant_turn_started":
                    first_started_at = received_at
                    first_turn_id = int(payload["turn_id"])

            await asyncio.sleep(args.barge_in_delay)
            barge_send_started = time.perf_counter()
            second_voice_stop = await _send_audio(
                mic,
                pcm16,
                voice_threshold=args.voice_threshold,
                silence_chunks=args.silence_chunks,
            )

            interruption_at: float | None = None
            second_started_at: float | None = None
            second_boundary_at: float | None = None
            second_ended_at: float | None = None
            second_turn_id: int | None = None
            deadline = time.monotonic() + args.timeout
            while second_ended_at is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("barge-in turn never reached media end")
                received_at, kind, payload = await asyncio.wait_for(
                    events.get(), timeout=remaining
                )
                if kind == "provider_error":
                    raise RuntimeError(str(payload))
                if kind == "pipeline_interruption":
                    interruption_at = interruption_at or received_at
                    continue
                if not isinstance(payload, dict):
                    continue
                turn_id = int(payload.get("turn_id", -1))
                if kind == "assistant_turn_started" and turn_id != first_turn_id:
                    second_turn_id = turn_id
                    second_started_at = received_at
                elif (
                    kind == "assistant_media_boundary"
                    and second_turn_id is not None
                    and turn_id == second_turn_id
                ):
                    second_boundary_at = received_at
                elif (
                    kind == "assistant_media_ended"
                    and second_turn_id is not None
                    and turn_id == second_turn_id
                ):
                    second_ended_at = received_at

            if interruption_at is None or second_started_at is None or second_boundary_at is None:
                raise RuntimeError("barge-in did not produce all required lifecycle events")
            return {
                "ok": True,
                "first_turn_id": first_turn_id,
                "second_turn_id": second_turn_id,
                "first_voice_stop_to_first_response_ms": _milliseconds(
                    first_started_at, first_voice_stop
                ),
                "first_response_to_barge_send_ms": round(
                    (barge_send_started - first_started_at) * 1000.0, 1
                ),
                "barge_send_to_pipeline_interruption_ms": round(
                    (interruption_at - barge_send_started) * 1000.0, 1
                ),
                "second_voice_stop_to_response_ms": _milliseconds(
                    second_started_at, second_voice_stop
                ),
                "second_voice_stop_to_media_boundary_ms": _milliseconds(
                    second_boundary_at, second_voice_stop
                ),
                "second_voice_stop_to_media_end_ms": _milliseconds(
                    second_ended_at, second_voice_stop
                ),
                "media_bytes": media_bytes,
            }
        finally:
            for task in (media_task, log_task):
                task.cancel()
            await asyncio.gather(media_task, log_task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="ws://127.0.0.1:7860")
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--barge-in-delay", type=float, default=0.5)
    parser.add_argument("--instruction", default="Answer briefly in Chinese.")
    parser.add_argument("--voice-threshold", type=int, default=500)
    parser.add_argument("--silence-chunks", type=int, default=30)
    args = parser.parse_args()
    if args.timeout <= 0 or args.barge_in_delay < 0 or args.silence_chunks <= 0:
        parser.error("timeout/silence-chunks must be positive and delay non-negative")
    result = asyncio.run(_run(args))
    print("CUSTOM_BARGEIN_PROBE_OK " + json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
