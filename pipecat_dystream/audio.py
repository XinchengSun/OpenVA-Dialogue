"""Small, deterministic PCM conversion helpers used by the bridge."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000


def pcm16le_to_float32_mono(
    audio: bytes,
    *,
    sample_rate: int,
    num_channels: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Convert interleaved PCM16LE to mono float32 at the target rate."""
    if not audio:
        return np.zeros(0, dtype=np.float32)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")

    frame_bytes = num_channels * 2
    usable = len(audio) - (len(audio) % frame_bytes)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)

    samples = np.frombuffer(audio[:usable], dtype="<i2")
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).astype(np.float32).mean(axis=1)
    else:
        samples = samples.astype(np.float32)
    mono = samples / 32768.0

    if sample_rate != target_sample_rate:
        divisor = gcd(sample_rate, target_sample_rate)
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            sample_rate // divisor,
        )

    return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)


def float32_to_pcm16le(audio: np.ndarray) -> bytes:
    """Convert normalized float audio to little-endian signed PCM16."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2").tobytes()
