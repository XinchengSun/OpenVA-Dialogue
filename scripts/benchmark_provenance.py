"""Credential-free provenance for benchmarks, without importing model runtimes.

The checkout and supplied configuration describe the measurement environment.
They do not attest which code, configuration, or weights a running server loaded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import platform
from pathlib import Path
import subprocess


PACKAGES = ("torch", "numpy", "pipecat-ai", "aiohttp", "websockets", "transformers", "voxcpm")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True, timeout=15,
    )
    return result.stdout


def checkout_identity(root: Path) -> dict:
    try:
        commit = git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
        # No diff text, filenames, remote URLs, or environment values are saved.
        diff = git_bytes(root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--")
        untracked = git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
        return {
            "commit": commit,
            "dirty_tracked": bool(diff),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_file_count": len([name for name in untracked.split(b"\0") if name]),
            "scope": "benchmark checkout; running server revision is not attested",
        }
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        return {"commit": None, "error": type(exc).__name__}


def collect_provenance(root: Path, args) -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    manifest = root / "weights-manifest.json"
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout": checkout_identity(root),
        "declared_service_commit": args.service_commit.lower() if args.service_commit else None,
        "measurement_runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "packages": versions,
        },
        "profile": args.profile,
        "declared_weights_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
        "runtime_config_sha256": sorted(file_sha256(path) for path in args.runtime_config),
        "input_wav_sha256": file_sha256(args.input_wav) if args.input_wav else None,
        "instruction_sha256": hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()
        if args.input_wav else None,
        "measurement": {
            name: getattr(args, name) for name in (
                "duration_sec", "sample_sec", "input_delay_sec", "turns",
                "turn_timeout_sec", "voice_threshold",
            )
        },
        "limitations": [
            "Package versions belong to the benchmark interpreter, which may differ from the service environment.",
            "Service commit is an operator declaration; the collector does not read or attest running server code.",
            "Manifest hash records declared model revisions, not a hash or verification of loaded weights.",
            "Supplied configuration hashes do not prove the service loaded those files; private values are never saved.",
            "Untracked files are counted but not fingerprinted; commit source changes before matched performance runs.",
            "Warm-up state, provider load, and public network conditions require an operator record.",
        ],
    }
