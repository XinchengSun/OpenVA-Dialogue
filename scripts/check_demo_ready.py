#!/usr/bin/env python3
"""Exercise the lazy MSE encoder and verify one fresh decodable video frame."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
import time

import aiohttp


EXPECTED_ENGINE_VERSION = "FLASHAV2AV_0.1.0"
EXPECTED_PAGE_MARKER = "DyStream"
REQUIRED_MP4_BOXES = (b"ftyp", b"moov", b"moof", b"mdat")


async def fetch_json(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()


async def capture_media(
    base_url: str,
    timeout_sec: float,
    expected_engine_version: str,
) -> tuple[bytes, str, dict]:
    deadline = time.monotonic() + timeout_sec
    timeout = aiohttp.ClientTimeout(total=max(timeout_sec + 5.0, 10.0))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{base_url}/") as response:
            response.raise_for_status()
            page = await response.text()
        if EXPECTED_PAGE_MARKER not in page:
            raise RuntimeError("demo page marker is missing")

        initial_health = await fetch_json(session, f"{base_url}/health")
        if initial_health.get("engine_version") != expected_engine_version:
            raise RuntimeError(
                "unexpected engine version: "
                f"{initial_health.get('engine_version')!r}, "
                f"expected {expected_engine_version!r}"
            )
        initial_seq = int(initial_health.get("latest_visible_frame_seq", 0))

        mime = ""
        media = bytearray()
        candidate_ready_at: float | None = None
        async with session.ws_connect(
            f"{base_url}/ws/media",
            max_msg_size=0,
            heartbeat=10,
        ) as ws:
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    message = await asyncio.wait_for(
                        ws.receive(),
                        timeout=min(remaining, 3.0),
                    )
                except asyncio.TimeoutError:
                    continue
                if message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "mime":
                        mime = str(payload.get("mime", ""))
                elif message.type == aiohttp.WSMsgType.BINARY:
                    media.extend(message.data)
                elif message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    raise RuntimeError(f"media WebSocket ended early: {message.type}")

                if (
                    "video/mp4" in mime
                    and all(box in media for box in REQUIRED_MP4_BOXES)
                    and media.count(b"moof") >= 4
                    and media.count(b"mdat") >= 4
                    and len(media) >= 512_000
                ):
                    health = await fetch_json(session, f"{base_url}/health")
                    visible_seq = int(health.get("latest_visible_frame_seq", 0))
                    visible_age = health.get("latest_visible_frame_age_sec")
                    if (
                        visible_seq > initial_seq
                        and visible_age is not None
                        and float(visible_age) < 2.0
                    ):
                        if candidate_ready_at is None:
                            candidate_ready_at = time.monotonic()
                        elif time.monotonic() - candidate_ready_at >= 0.5:
                            return bytes(media), mime, health

        missing = [
            box.decode("ascii") for box in REQUIRED_MP4_BOXES if box not in media
        ]
        raise TimeoutError(
            "media readiness timeout: "
            f"mime={mime!r} bytes={len(media)} missing_boxes={missing}"
        )


def verify_decodable(media: bytes) -> dict:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(media)
            temp_path = Path(temp_file.name)

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                str(temp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        probe_data = json.loads(probe.stdout)
        streams = probe_data.get("streams", [])
        video = next(
            (stream for stream in streams if stream.get("codec_name") == "h264"),
            None,
        )
        audio = next(
            (stream for stream in streams if stream.get("codec_name") == "aac"),
            None,
        )
        if video is None or audio is None:
            raise RuntimeError(f"expected H.264 + AAC streams, got: {streams!r}")
        if int(video.get("width", 0)) < 128 or int(video.get("height", 0)) < 128:
            raise RuntimeError(f"invalid video dimensions: {video!r}")

        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(temp_path),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        return {
            "video": video["codec_name"],
            "audio": audio["codec_name"],
            "width": int(video["width"]),
            "height": int(video["height"]),
        }
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


async def async_main(
    port: int,
    timeout_sec: float,
    expected_engine_version: str,
) -> None:
    base_url = f"http://127.0.0.1:{port}"
    media, mime, health = await capture_media(
        base_url,
        timeout_sec,
        expected_engine_version,
    )
    stream = verify_decodable(media)
    print(
        "MEDIA_SMOKE_OK "
        f"bytes={len(media)} mime={mime!r} "
        f"video={stream['video']} audio={stream['audio']} "
        f"size={stream['width']}x{stream['height']} "
        f"visible_frame_seq={health['latest_visible_frame_seq']} "
        f"visible_frame_age_sec={health['latest_visible_frame_age_sec']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--expected-engine-version",
        default=EXPECTED_ENGINE_VERSION,
    )
    args = parser.parse_args()
    asyncio.run(
        async_main(
            args.port,
            args.timeout,
            args.expected_engine_version,
        )
    )


if __name__ == "__main__":
    main()
