#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import re
import ssl
from pathlib import Path
from urllib.parse import quote, urlsplit


COOKIE_NAME = "dystream_public_access"
TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the public access gate without printing its token.",
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--origin",
        default="http://127.0.0.1:7860",
        help="Origin or public base URL.",
    )
    parser.add_argument(
        "--public-host",
        help="Override Host when probing a loopback origin as if through a proxy.",
    )
    return parser.parse_args()


def read_token(env_file: Path) -> str:
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("PUBLIC_ACCESS_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if TOKEN_RE.fullmatch(token):
                return token
            break
    raise SystemExit("PUBLIC_ACCESS_TOKEN is missing or invalid")


def request(
    base_url: str,
    path: str,
    *,
    host_header: str | None = None,
    cookie: str | None = None,
) -> tuple[int, dict[str, str]]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"invalid origin URL: {base_url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    kwargs = {"timeout": 15}
    if parsed.scheme == "https":
        kwargs["context"] = ssl.create_default_context()
    connection = connection_cls(parsed.hostname, port, **kwargs)
    try:
        connection.putrequest("GET", path, skip_host=True)
        connection.putheader("Host", host_header or parsed.netloc)
        connection.putheader("User-Agent", "dystream-public-probe/1")
        if cookie:
            connection.putheader("Cookie", cookie)
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        return response.status, {key.lower(): value for key, value in response.getheaders()}
    finally:
        connection.close()


def expect(label: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise SystemExit(f"{label}: expected HTTP {expected}, got {actual}")
    print(f"{label}: HTTP {actual}")


def main() -> None:
    args = parse_args()
    token = read_token(args.env_file.expanduser().resolve())
    public_host = args.public_host

    unauthorized, _ = request(args.origin, "/", host_header=public_host)
    expect("unauthenticated public root", unauthorized, 401)

    authorized_redirect, redirect_headers = request(
        args.origin,
        f"/?access_token={quote(token, safe='')}",
        host_header=public_host,
    )
    expect("token exchange", authorized_redirect, 302)
    set_cookie = redirect_headers.get("set-cookie", "")
    required_cookie_flags = (
        f"{COOKIE_NAME}=",
        "Secure",
        "HttpOnly",
        "SameSite=Strict",
    )
    if not all(flag.lower() in set_cookie.lower() for flag in required_cookie_flags):
        raise SystemExit("token exchange did not return the required secure cookie")
    print("secure cookie flags: present")

    cookie = f"{COOKIE_NAME}={token}"
    authorized, _ = request(args.origin, "/", host_header=public_host, cookie=cookie)
    expect("authenticated public root", authorized, 200)

    customization, _ = request(
        args.origin,
        "/customize",
        host_header=public_host,
        cookie=cookie,
    )
    expect("public customization boundary", customization, 403)
    print("public access probe: 4/4 passed; token was not printed")


if __name__ == "__main__":
    main()
