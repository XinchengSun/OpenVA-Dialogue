"""Transcribe a normalized 16 kHz reference WAV with cached Paraformer."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 10 * 960  # 600 ms, matching the realtime Paraformer path.
SENSEVOICE_LANGUAGES = {
    "auto": "auto",
    "zh-CN": "zh",
    "en-US": "en",
    "ja-JP": "ja",
}


def _read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != SAMPLE_RATE
        ):
            raise ValueError("reference WAV must be mono PCM16 at 16 kHz")
        pcm = wav.readframes(wav.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def transcribe(
    path: Path,
    *,
    model_name: str,
    device: str,
    hub: str,
    language: str = "auto",
) -> str:
    from funasr import AutoModel

    model = AutoModel(
        model=model_name,
        device=device,
        hub=hub,
        disable_update=True,
    )
    if "sensevoice" in model_name.lower():
        try:
            sensevoice_language = SENSEVOICE_LANGUAGES[language]
        except KeyError as exc:
            raise ValueError(f"unsupported reference language: {language}") from exc
        result = model.generate(
            input=str(path),
            language=sensevoice_language,
            use_itn=True,
        )
        if not result:
            return ""
        # SenseVoice prefixes language/emotion/event tags separated by ``|>``.
        return str(result[0].get("text", "")).split("|>")[-1].strip()

    samples = _read_pcm16_mono(path)
    if samples.size == 0:
        return ""

    cache: dict = {}
    parts: list[str] = []
    for offset in range(0, samples.size, CHUNK_SAMPLES):
        chunk = samples[offset : offset + CHUNK_SAMPLES]
        is_final = offset + CHUNK_SAMPLES >= samples.size
        result = model.generate(
            input=chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=[0, 10, 5],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )
        if result:
            text = str(result[0].get("text", "")).strip()
            if text:
                parts.append(text)
    return "".join(parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hub", default="ms")
    parser.add_argument("--language", choices=tuple(SENSEVOICE_LANGUAGES), default="auto")
    args = parser.parse_args()

    text = transcribe(
        args.audio,
        model_name=args.model,
        device=args.device,
        hub=args.hub,
        language=args.language,
    )
    print(json.dumps({
        "text": text,
        "language": args.language,
        "model": args.model,
    }, ensure_ascii=False))
    return 0 if text else 2


if __name__ == "__main__":
    raise SystemExit(main())
