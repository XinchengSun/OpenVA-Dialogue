#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from probe_public_access import COOKIE_NAME, read_token


REQUIRED_MP4_BOXES = (b"ftyp", b"moov", b"moof", b"mdat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the authenticated public media WebSocket.",
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=35.0)
    return parser.parse_args()


def websocket_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit(f"invalid public URL: {base_url}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, "/ws/media", "", ""))


async def probe(base_url: str, token: str, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    cookie = f"{COOKIE_NAME}={token}"
    headers = {"Cookie": cookie, "User-Agent": "dystream-public-media-probe/1"}
    timeout = aiohttp.ClientTimeout(total=timeout_sec + 10)
    mime = ""
    media = bytearray()

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(f"{base_url}/health") as response:
            response.raise_for_status()
            health = await response.json()
        if health.get("status") not in {"ok", "ready"}:
            raise RuntimeError(f"unexpected health status: {health.get('status')!r}")

        async with session.ws_connect(
            websocket_url(base_url),
            max_msg_size=0,
            heartbeat=10,
        ) as websocket:
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=min(remaining, 5.0),
                )
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
                    raise RuntimeError(f"public media WebSocket ended early: {message.type}")

                if (
                    "video/mp4" in mime
                    and all(box in media for box in REQUIRED_MP4_BOXES)
                    and media.count(b"moof") >= 2
                    and media.count(b"mdat") >= 2
                    and len(media) >= 256_000
                ):
                    print(
                        "PUBLIC_MEDIA_WS_OK "
                        f"bytes={len(media)} mime={mime!r} "
                        "boxes=ftyp,moov,moof,mdat; token was not printed"
                    )
                    return

    missing = [box.decode("ascii") for box in REQUIRED_MP4_BOXES if box not in media]
    raise TimeoutError(
        "public media readiness timeout: "
        f"mime={mime!r} bytes={len(media)} missing_boxes={missing}"
    )


def main() -> None:
    args = parse_args()
    token = read_token(args.env_file.expanduser().resolve())
    base_url = args.url_file.expanduser().resolve().read_text(encoding="utf-8").strip()
    asyncio.run(probe(base_url.rstrip("/"), token, args.timeout))


if __name__ == "__main__":
    main()
