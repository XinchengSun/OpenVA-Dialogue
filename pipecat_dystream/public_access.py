from __future__ import annotations

import hmac
import ipaddress

from aiohttp.abc import AbstractAccessLogger


ACCESS_COOKIE_NAME = "dystream_public_access"


class RedactedAccessLogger(AbstractAccessLogger):
    """Log request paths without query strings or authentication tokens."""

    def log(self, request, response, elapsed: float) -> None:
        self.logger.info(
            '%s "%s %s" %s %s %.6f',
            request.remote or "-",
            request.method,
            request.path,
            response.status,
            response.body_length,
            elapsed,
        )


def host_without_port(value: str) -> str:
    host = (value or "").strip().lower()
    if host.startswith("["):
        closing = host.find("]")
        return host[1:closing] if closing > 0 else ""
    if host.count(":") == 1:
        host, _ = host.rsplit(":", 1)
    return host.rstrip(".")


def is_loopback_host(value: str) -> bool:
    host = host_without_port(value)
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def token_matches(supplied: str | None, expected: str) -> bool:
    if not supplied or not expected:
        return False
    return hmac.compare_digest(
        supplied.encode("utf-8"),
        expected.encode("utf-8"),
    )
