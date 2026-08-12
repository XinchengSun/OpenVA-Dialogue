#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


KEY = "CUDA_VISIBLE_DEVICES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically change only the physical GPU used by VoxCPM2.",
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def update_gpu(env_file: Path, gpu: int) -> None:
    env_file = env_file.expanduser().resolve()
    if not env_file.is_file():
        raise SystemExit(f"missing env file: {env_file}")
    if gpu < 0 or gpu > 255:
        raise SystemExit("--gpu must be a non-negative physical GPU index")

    lines = env_file.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(f"{KEY}=")]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {KEY} entry, found {len(matches)}")
    lines[matches[0]] = f"{KEY}={gpu}"

    temporary = env_file.with_name(f".{env_file.name}.gpu.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines) + "\n")
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
    print(f"updated {KEY}={gpu} in {env_file}; no other values were printed")


if __name__ == "__main__":
    arguments = parse_args()
    update_gpu(arguments.env_file, arguments.gpu)
