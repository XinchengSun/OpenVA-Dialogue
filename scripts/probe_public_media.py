#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

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


def endpoint_url(
    base_url: str,
    path: str,
    client_id: str,
    *,
    websocket: bool = False,
) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit(f"invalid public URL: {base_url}")
    scheme = (
        "wss" if parsed.scheme == "https" else "ws"
    ) if websocket else parsed.scheme
    return urlunsplit(
        (scheme, parsed.netloc, path, urlencode({"client_id": client_id}), "")
    )


async def keep_lease_alive(websocket: aiohttp.ClientWebSocketResponse) -> None:
    try:
        while not websocket.closed:
            await asyncio.sleep(4.0)
            await websocket.send_json({"type": "lease_heartbeat"})
    except (asyncio.CancelledError, ConnectionError, RuntimeError):
        return


async def probe(base_url: str, token: str, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    client_id = uuid.uuid4().hex
    cookie = f"{COOKIE_NAME}={token}"
    headers = {"Cookie": cookie, "User-Agent": "dystream-public-media-probe/1"}
    timeout = aiohttp.ClientTimeout(total=timeout_sec + 10)
    mime = ""
    media = bytearray()

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(
            endpoint_url(base_url, "/api/realtime/status", client_id)
        ) as response:
            response.raise_for_status()
            status = await response.json()
        conversation_state = status.get("conversation", {}).get("state")
        if conversation_state == "busy":
            raise RuntimeError(
                "public conversation slot is busy; probe did not take over the active user"
            )
        if conversation_state != "available":
            raise RuntimeError(
                "public conversation slot is unavailable: "
                f"phase={status.get('phase')!r} state={conversation_state!r}"
            )

        async with session.ws_connect(
            endpoint_url(base_url, "/ws/mic", client_id, websocket=True),
            max_msg_size=0,
            heartbeat=10,
        ) as microphone:
            admission = await asyncio.wait_for(microphone.receive(), timeout=5.0)
            if admission.type != aiohttp.WSMsgType.TEXT:
                raise RuntimeError("microphone lease ended before admission")
            try:
                admission_payload = json.loads(admission.data)
            except json.JSONDecodeError as exc:
                raise RuntimeError("microphone lease returned invalid admission data") from exc
            if admission_payload.get("type") != "lease_granted":
                raise RuntimeError(
                    "microphone lease was not granted: "
                    f"{admission_payload.get('reason') or admission_payload.get('type')!r}"
                )

            heartbeat_task = asyncio.create_task(keep_lease_alive(microphone))
            try:
                async with session.ws_connect(
                    endpoint_url(base_url, "/ws/media", client_id, websocket=True),
                    max_msg_size=0,
                    heartbeat=10,
                ) as media_websocket:
                    while time.monotonic() < deadline:
                        remaining = max(0.1, deadline - time.monotonic())
                        message = await asyncio.wait_for(
                            media_websocket.receive(),
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
                            raise RuntimeError(
                                "public media WebSocket ended early: "
                                f"{message.type}"
                            )

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
            finally:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)

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
