#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import secrets
from pathlib import Path


MANAGED_KEYS = {
    "SERVER_BIND_HOST",
    "PUBLIC_ACCESS_TOKEN",
    "CUSTOMIZATION_ALLOW_REMOTE",
}
TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically configure the loopback origin and public access token.",
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--rotate-token", action="store_true")
    return parser.parse_args()


def read_existing_token(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("PUBLIC_ACCESS_TOKEN="):
            value = line.split("=", 1)[1].strip()
            return value if TOKEN_RE.fullmatch(value) else ""
    return ""


def configure(env_file: Path, bind_host: str, rotate_token: bool) -> None:
    env_file = env_file.expanduser().resolve()
    if not env_file.is_file():
        raise SystemExit(f"missing env file: {env_file}")
    if bind_host not in {"127.0.0.1", "::1"}:
        raise SystemExit("public tunnel origin must bind to a loopback address")

    original_lines = env_file.read_text(encoding="utf-8").splitlines()
    existing_token = read_existing_token(original_lines)
    token = secrets.token_hex(32) if rotate_token or not existing_token else existing_token
    kept = [
        line
        for line in original_lines
        if line.split("=", 1)[0].strip() not in MANAGED_KEYS
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    kept.extend([
        "",
        "# Public HTTPS/WSS frontend; managed by configure_public_access.py.",
        f"SERVER_BIND_HOST={bind_host}",
        f"PUBLIC_ACCESS_TOKEN={token}",
        "CUSTOMIZATION_ALLOW_REMOTE=0",
    ])

    temporary = env_file.with_name(f".{env_file.name}.public.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(kept) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, env_file)
        os.chmod(env_file, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    print(f"configured public access in {env_file}; token was not printed")


if __name__ == "__main__":
    arguments = parse_args()
    configure(arguments.env_file, arguments.bind_host, arguments.rotate_token)
