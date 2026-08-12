"""Pipecat integration for the existing DyStream realtime engine.

Heavy Pipecat imports stay lazy so lightweight helpers such as customization
and public-access validation can be imported by setup/test tooling.
"""

from __future__ import annotations

from typing import Any


_BRIDGE_EXPORTS = {
    "DyStreamAvatarProcessor",
    "DyStreamOutputClient",
    "DyStreamUserAudioTap",
}
__all__ = sorted(_BRIDGE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _BRIDGE_EXPORTS:
        raise AttributeError(name)
    from . import bridge

    return getattr(bridge, name)
