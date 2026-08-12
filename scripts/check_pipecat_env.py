#!/usr/bin/env python3
"""Validate the exact imports used by the Pipecat + MSE demo entry point."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

EXPECTED_DISTRIBUTIONS = {
    "cryptography": "49.0.0",
    "dashscope": "1.26.4",
    "numpy": "1.26.4",
    "opencv-python-headless": "4.11.0.86",
    "pipecat-ai": "1.6.0",
    "protobuf": "4.25.9",
    # PyPI metadata omits the local CUDA suffix; runtime is checked below.
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "websockets": "14.1",
}


def main() -> None:
    errors: list[str] = []
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{distribution}: missing (expected {expected})")
            continue
        if actual != expected:
            errors.append(f"{distribution}: {actual} (expected {expected})")

    if errors:
        raise RuntimeError("Pipecat environment version mismatch:\n- " + "\n- ".join(errors))

    # MediaPipe must import with the protobuf runtime used by DyStream.
    from mediapipe.framework.formats import landmark_pb2  # noqa: F401

    import torch

    runtime_torch = str(torch.__version__)
    runtime_cuda = str(torch.version.cuda)
    if runtime_torch != "2.8.0+cu128" or runtime_cuda != "12.8":
        raise RuntimeError(
            "PyTorch runtime mismatch: "
            f"torch={runtime_torch}, cuda={runtime_cuda} "
            "(expected torch=2.8.0+cu128, cuda=12.8)"
        )

    # Import the current Pipecat/MSE surface. These imports intentionally fail
    # before GPU startup so a stale virtual environment cannot produce a
    # half-initialized demo.
    from pipecat.frames.frames import TTSAudioRawFrame  # noqa: F401
    from pipecat.pipeline.pipeline import Pipeline  # noqa: F401
    from pipecat_dystream.mse_session import PipecatMSESession  # noqa: F401
    from pipecat_dystream.qwen_audio_s2s import (  # noqa: F401
        QwenAudioRealtimeS2SProcessor,
    )

    versions = ", ".join(
        f"{name}={version}" for name, version in EXPECTED_DISTRIBUTIONS.items()
    )
    print(
        f"Pipecat MSE environment OK: {versions}, "
        f"torch-runtime={runtime_torch}, cuda-runtime={runtime_cuda}"
    )


if __name__ == "__main__":
    main()
