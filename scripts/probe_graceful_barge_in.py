#!/usr/bin/env python3
"""Exercise one real S2S graceful barge-in on the continuous MSE timeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import aiohttp


async def wait_for_log(records, needle, *, after, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for timestamp, message in records:
            if timestamp >= after and needle in message:
                return timestamp, message
        await asyncio.sleep(0.01)
    raise TimeoutError(f"missing log {needle!r}; recent={records[-60:]}")


async def wait_for_control(records, event_type, *, after, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for timestamp, payload in records:
            if timestamp >= after and payload.get("type") == event_type:
                return timestamp, payload
        await asyncio.sleep(0.01)
    raise TimeoutError(f"missing control {event_type!r}; recent={records[-30:]}")


async def send_utterance(ws, pcm):
    chunk_bytes = 640 * 2  # 40 ms mono 16 kHz PCM16
    for offset in range(0, len(pcm), chunk_bytes):
        await ws.send_bytes(pcm[offset:offset + chunk_bytes])
        await asyncio.sleep(0.04)
    for _ in range(15):
        await ws.send_bytes(bytes(chunk_bytes))
        await asyncio.sleep(0.04)


async def run(args):
    with wave.open(str(args.audio), "rb") as source:
        if (
            source.getframerate() != 16_000
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
        ):
            raise ValueError("input must be mono 16 kHz PCM16 WAV")
        pcm = source.readframes(source.getnframes())[
            : int(args.input_seconds * 16_000 * 2)
        ]

    logs = []
    controls = []
    media = bytearray()
    async with aiohttp.ClientSession() as session:
        logs_ws = await session.ws_connect(f"{args.base_url}/ws/logs", max_msg_size=0)
        media_ws = await session.ws_connect(f"{args.base_url}/ws/media", max_msg_size=0)

        async def collect_logs():
            async for message in logs_ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    continue
                logs.append((time.monotonic(), payload.get("message", "")))

        async def collect_media():
            async for message in media_ws:
                if message.type == aiohttp.WSMsgType.BINARY:
                    media.extend(message.data)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        controls.append((time.monotonic(), json.loads(message.data)))
                    except json.JSONDecodeError:
                        pass

        log_task = asyncio.create_task(collect_logs())
        media_task = asyncio.create_task(collect_media())
        mic_ws = await session.ws_connect(f"{args.base_url}/ws/mic", max_msg_size=0)
        await mic_ws.send_json({
            "type": "config",
            "instruction": "Reply naturally in one short Chinese sentence.",
        })
        reply = await asyncio.wait_for(mic_ws.receive(), timeout=5)
        if json.loads(reply.data).get("type") != "ok":
            raise RuntimeError(f"microphone config rejected: {reply.data}")

        turn_started = time.monotonic()
        await send_utterance(mic_ws, pcm)
        first_audio_at, first_audio_log = await wait_for_log(
            logs,
            "[PIPECAT PROVIDER] first audio",
            after=turn_started,
            timeout=20,
        )
        _, first_boundary = await wait_for_control(
            controls,
            "assistant_media_boundary",
            after=turn_started,
            timeout=10,
        )
        await asyncio.sleep(0.15)

        async with session.get(f"{args.base_url}/health") as response:
            before = await response.json()
        bytes_before_interrupt = len(media)
        interrupt_at = time.monotonic()
        await mic_ws.send_json({"type": "interrupt"})

        interrupt_state_at, interrupt_state_log = await wait_for_log(
            logs,
            "[ENGINE STATE] interrupt -> ASSISTANT_TAIL",
            after=interrupt_at,
            timeout=3,
        )
        if "generation_preserved=1 reset=0" not in interrupt_state_log:
            raise AssertionError(interrupt_state_log)
        _, interrupt_control = await wait_for_control(
            controls,
            "assistant_interrupted",
            after=interrupt_at,
            timeout=3,
        )
        end_boundary_at, end_boundary = await wait_for_control(
            controls,
            "assistant_media_ended",
            after=interrupt_at,
            timeout=10,
        )
        await asyncio.sleep(0.5)

        forbidden_logs = [
            message
            for timestamp, message in logs
            if timestamp >= interrupt_at
            and any(marker in message for marker in (
                "[PIPE FENCE]",
                "[ENGINE LIVE] reset applied",
                "[ENGINE BRIDGE]",
                "encoder reset epoch=",
            ))
        ]
        forbidden_controls = [
            payload
            for timestamp, payload in controls
            if timestamp >= interrupt_at and payload.get("type") == "stream_reset"
        ]
        if forbidden_logs or forbidden_controls:
            raise AssertionError(
                f"graceful interrupt reset media: logs={forbidden_logs} "
                f"controls={forbidden_controls}"
            )
        if interrupt_control.get("graceful") is not True:
            raise AssertionError(interrupt_control)
        if interrupt_control.get("generation") != before["stream_generation"]:
            raise AssertionError((interrupt_control, before))
        if end_boundary.get("generation") != before["stream_generation"]:
            raise AssertionError((end_boundary, before))
        if len(media) <= bytes_before_interrupt:
            raise AssertionError("continuous media stopped after graceful interruption")

        async with session.get(f"{args.base_url}/health") as response:
            after = await response.json()
        if after["stream_generation"] != before["stream_generation"]:
            raise AssertionError((before, after))

        await mic_ws.close()
        await media_ws.close()
        await logs_ws.close()
        for task in (log_task, media_task):
            try:
                await asyncio.wait_for(task, timeout=2)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(media)
    return {
        "engine_version": after["engine_version"],
        "generation_before": before["stream_generation"],
        "generation_after": after["stream_generation"],
        "grace_sec": interrupt_control.get("grace_sec"),
        "media_bytes": len(media),
        "interrupt_to_engine_ms": round((interrupt_state_at - interrupt_at) * 1000, 1),
        "interrupt_to_media_end_ms": round((end_boundary_at - interrupt_at) * 1000, 1),
        "first_audio_ms": round((first_audio_at - turn_started) * 1000, 1),
        "first_audio_log": first_audio_log,
        "first_boundary": first_boundary,
        "interrupt_state_log": interrupt_state_log,
        "interrupt_control": interrupt_control,
        "end_boundary": end_boundary,
        "output": str(args.output) if args.output else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("/tmp/codex_fixed_hello_16k.wav"),
    )
    parser.add_argument("--input-seconds", type=float, default=2.2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))
