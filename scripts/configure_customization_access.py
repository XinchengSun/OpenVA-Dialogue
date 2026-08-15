#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import secrets
from pathlib import Path
from urllib.parse import urlparse


MANAGED_KEYS = {
    "CUSTOMIZATION_ALLOW_REMOTE",
    "CUSTOMIZATION_REMOTE_ENABLED",
    "CUSTOMIZATION_ADMIN_TOKEN",
    "CUSTOMIZATION_PUBLIC_ORIGIN",
    "CUSTOMIZATION_ADMIN_SESSION_TTL_SEC",
}
TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_SESSION_TTL_SEC = 3600
MIN_SESSION_TTL_SEC = 300
MAX_SESSION_TTL_SEC = 24 * 60 * 60


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically configure the separately-authenticated customization UI."
    )
    parser.add_argument(
        "action",
        choices=("enable", "disable", "rotate-token", "status", "show-token"),
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--public-origin")
    parser.add_argument("--session-ttl", type=int)
    return parser


def _parse_values(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key in MANAGED_KEYS:
            values[key] = value.strip()
    return values


def _normalize_origin(raw: str) -> str:
    value = (raw or "").strip().rstrip("/")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit("--public-origin must be one HTTPS origin without a path")
    return f"https://{parsed.netloc.lower()}"


def _session_ttl(requested: int | None, existing: str) -> int:
    if requested is None:
        try:
            requested = int(existing or DEFAULT_SESSION_TTL_SEC)
        except ValueError:
            requested = DEFAULT_SESSION_TTL_SEC
    if not MIN_SESSION_TTL_SEC <= requested <= MAX_SESSION_TTL_SEC:
        raise SystemExit(
            f"--session-ttl must be between {MIN_SESSION_TTL_SEC} and {MAX_SESSION_TTL_SEC}"
        )
    return requested


def _write_private(path: Path, lines: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.customization.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _configure(args: argparse.Namespace) -> None:
    env_file = args.env_file.expanduser().resolve()
    if not env_file.is_file():
        raise SystemExit(f"missing env file: {env_file}")
    original = env_file.read_text(encoding="utf-8").splitlines()
    values = _parse_values(original)
    existing_token = values.get("CUSTOMIZATION_ADMIN_TOKEN", "")
    existing_token_valid = bool(TOKEN_RE.fullmatch(existing_token))
    token = existing_token if existing_token_valid else secrets.token_hex(32)
    origin_raw = args.public_origin or values.get("CUSTOMIZATION_PUBLIC_ORIGIN", "")
    enabled = values.get("CUSTOMIZATION_REMOTE_ENABLED", "0") == "1"

    if args.action == "enable":
        origin = _normalize_origin(origin_raw)
        enabled = True
    elif args.action == "disable":
        origin = _normalize_origin(origin_raw) if origin_raw else ""
        enabled = False
    elif args.action == "rotate-token":
        token = secrets.token_hex(32)
        origin = _normalize_origin(origin_raw) if origin_raw else ""
    else:
        origin = _normalize_origin(origin_raw) if origin_raw else ""

    ttl = _session_ttl(args.session_ttl, values.get("CUSTOMIZATION_ADMIN_SESSION_TTL_SEC", ""))
    if args.action == "status":
        print(f"remote_customization_enabled={int(enabled)}")
        print(f"public_origin={origin or 'unset'}")
        print(f"admin_token_configured={int(existing_token_valid)}")
        print(f"admin_session_ttl_sec={ttl}")
        return
    if args.action == "show-token":
        if not TOKEN_RE.fullmatch(existing_token):
            raise SystemExit("CUSTOMIZATION_ADMIN_TOKEN is not configured")
        print(existing_token)
        return

    kept = [
        line
        for line in original
        if line.split("=", 1)[0].strip() not in MANAGED_KEYS
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    kept.extend([
        "",
        "# Remote customization admin access; managed by configure_customization_access.py.",
        "CUSTOMIZATION_ALLOW_REMOTE=0",
        f"CUSTOMIZATION_REMOTE_ENABLED={int(enabled)}",
        f"CUSTOMIZATION_ADMIN_TOKEN={token}",
        f"CUSTOMIZATION_PUBLIC_ORIGIN={origin}",
        f"CUSTOMIZATION_ADMIN_SESSION_TTL_SEC={ttl}",
    ])
    _write_private(env_file, kept)
    print(
        f"configured remote customization in {env_file}; "
        "the administrator token was not printed"
    )


if __name__ == "__main__":
    _configure(_parser().parse_args())
