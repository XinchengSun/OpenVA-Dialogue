"""SmallWebRTC ICE server configuration."""

from __future__ import annotations

import os

from aiortc import RTCIceServer

DEFAULT_BROWSER_STUN_URLS = [
    "stun:stun.miwifi.com:3478",
    "stun:stun.chat.bilibili.com:3478",
]



def ice_servers_from_env() -> list[RTCIceServer]:
    urls = [
        value.strip()
        for value in os.getenv("PIPECAT_ICE_SERVERS", "").split(",")
        if value.strip()
    ]
    return [RTCIceServer(urls=url) for url in urls]


def browser_ice_servers_from_env() -> list[dict[str, object]]:
    servers: list[dict[str, object]] = [{"urls": DEFAULT_BROWSER_STUN_URLS}]
    turn_url = os.getenv("PIPECAT_TURN_URL", "").strip()
    if turn_url:
        servers.append(
            {
                "urls": turn_url,
                "username": os.environ["PIPECAT_TURN_USERNAME"],
                "credential": os.environ["PIPECAT_TURN_CREDENTIAL"],
            }
        )
    return servers
