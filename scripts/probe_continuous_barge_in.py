#!/usr/bin/env python3
"""Exercise one real Qwen S2S barge-in without rebuilding FFmpeg/MSE."""

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
    raise TimeoutError(
        f"did not see {needle!r}; recent={[text for _, text in records[-60:]]}"
    )


async def wait_for_control(records, event_type, *, after, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for timestamp, payload in records:
            if timestamp >= after and payload.get("type") == event_type:
                return timestamp, payload
        await asyncio.sleep(0.01)
    raise TimeoutError(
        f"did not see control {event_type!r}; recent={records[-30:]}"
    )


async def send_utterance(ws, pcm):
    # Match the V20 browser path: 40 ms of 16 kHz PCM16 per WebSocket frame.
    chunk_bytes = 640 * 2
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
                    continue
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    continue
                controls.append((time.monotonic(), payload))

        log_task = asyncio.create_task(collect_logs())
        media_task = asyncio.create_task(collect_media())
        mic_ws = await session.ws_connect(f"{args.base_url}/ws/mic", max_msg_size=0)
        await mic_ws.send_json(
            {
                "type": "config",
                "instruction": "Reply naturally in one short Chinese sentence.",
            }
        )
        reply = await asyncio.wait_for(mic_ws.receive(), timeout=5)
        if json.loads(reply.data).get("type") != "ok":
            raise RuntimeError(f"microphone config rejected: {reply.data}")

        first_turn_at = time.monotonic()
        await send_utterance(mic_ws, pcm)
        first_audio_at, first_audio_log = await wait_for_log(
            logs,
            "[PIPECAT PROVIDER] first audio",
            after=first_turn_at,
            timeout=20,
        )
        await wait_for_control(
            controls,
            "assistant_media_boundary",
            after=first_turn_at,
            timeout=10,
        )
        await asyncio.sleep(0.25)

        bytes_before_interrupt = len(media)
        interrupt_at = time.monotonic()
        await mic_ws.send_json({"type": "interrupt"})
        interrupt_state_at, interrupt_state_log = await wait_for_log(
            logs,
            "[ENGINE STATE] interrupt -> WARMUP_IDLE",
            after=interrupt_at,
            timeout=2,
        )
        bridge_at, bridge_log = await wait_for_log(
            logs,
            "[ENGINE BRIDGE]",
            after=interrupt_at,
            timeout=5,
        )
        fence_at, fence_log = await wait_for_log(
            logs,
            "[PIPE FENCE]",
            after=interrupt_at,
            timeout=5,
        )
        _, interrupt_control = await wait_for_control(
            controls,
            "assistant_interrupted",
            after=interrupt_at,
            timeout=2,
        )
        await asyncio.sleep(1.0)
        forbidden_logs = [
            message
            for timestamp, message in logs
            if timestamp >= interrupt_at
            and (
                "encoder reset epoch=" in message
                or "client_stream_reset=1" in message
            )
        ]
        forbidden_controls = [
            payload
            for timestamp, payload in controls
            if timestamp >= interrupt_at and payload.get("type") == "stream_reset"
        ]
        if forbidden_logs or forbidden_controls:
            raise AssertionError(
                f"ordinary interruption rebuilt media: logs={forbidden_logs} "
                f"controls={forbidden_controls}"
            )
        if len(media) <= bytes_before_interrupt:
            raise AssertionError("continuous media stopped producing bytes after interruption")

        second_turn_at = time.monotonic()
        await send_utterance(mic_ws, pcm)
        second_tts_at, second_start_log = await wait_for_log(
            logs,
            "[PIPECAT PROVIDER] TTS turn started",
            after=second_turn_at,
            timeout=20,
        )
        second_audio_at, second_audio_log = await wait_for_log(
            logs,
            "[PIPECAT PROVIDER] first audio",
            after=second_tts_at,
            timeout=10,
        )
        second_boundary_at, second_boundary = await wait_for_control(
            controls,
            "assistant_media_boundary",
            after=second_tts_at,
            timeout=10,
        )
        second_end_at, second_end_log = await wait_for_log(
            logs,
            "[PIPECAT PROVIDER] TTS turn ended",
            after=second_tts_at,
            timeout=20,
        )

        async with session.get(f"{args.base_url}/health") as response:
            health = await response.json()
        if response.status != 200:
            raise AssertionError(health)
        if second_boundary["generation"] != health["stream_generation"]:
            raise AssertionError(
                f"boundary generation mismatch: {second_boundary} vs {health}"
            )

        await mic_ws.close()
        await asyncio.sleep(0.3)
        await media_ws.close()
        await logs_ws.close()
        for task in (log_task, media_task):
            try:
                await asyncio.wait_for(task, timeout=2)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    args.output.write_bytes(media)
    return {
        "engine_version": health["engine_version"],
        "generation": health["stream_generation"],
        "ffmpeg_stream_epoch": second_boundary["stream_epoch"],
        "media_bytes": len(media),
        "output": str(args.output),
        "first_audio_log": first_audio_log,
        "interrupt_to_engine_ms": round((interrupt_state_at - interrupt_at) * 1000, 1),
        "interrupt_to_bridge_ms": round((bridge_at - interrupt_at) * 1000, 1),
        "interrupt_to_fence_ms": round((fence_at - interrupt_at) * 1000, 1),
        "second_turn_to_tts_start_ms": round((second_tts_at - second_turn_at) * 1000, 1),
        "second_tts_to_first_audio_ms": round((second_audio_at - second_tts_at) * 1000, 1),
        "second_tts_to_media_boundary_ms": round(
            (second_boundary_at - second_tts_at) * 1000,
            1,
        ),
        "second_tts_duration_ms": round((second_end_at - second_tts_at) * 1000, 1),
        "interrupt_state_log": interrupt_state_log,
        "bridge_log": bridge_log,
        "fence_log": fence_log,
        "interrupt_control": interrupt_control,
        "second_start_log": second_start_log,
        "second_audio_log": second_audio_log,
        "second_boundary": second_boundary,
        "second_end_log": second_end_log,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/v20_continuous_barge_in.mp4"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = asyncio.run(run(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
