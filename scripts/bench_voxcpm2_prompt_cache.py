#!/usr/bin/env python3
"""Benchmark Nano bridge latency against official VoxCPM2 prompt-cache continuity.

The two modes intentionally bypass ASR, LLM, Pipecat, and avatar rendering.  They
write one WAV per sentence plus machine-readable metrics.  Official mode commits
generated acoustic features only after a sentence completes successfully.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_SENTENCES = (
    "当然可以，我先把最重要的结论告诉你。",
    "这个方案会保持实时生成，也会让前后声音更加稳定。",
    "如果首声出现变慢，我们就立即回退当前版本。",
)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _write_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    pcm16 = (np.clip(samples.reshape(-1), -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _audible_onset_ms(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = -50.0,
    confirm_ms: int = 20,
) -> float:
    frame_samples = max(1, round(sample_rate * 0.010))
    confirm_frames = max(1, math.ceil(confirm_ms / 10))
    usable = samples[: samples.size // frame_samples * frame_samples]
    if usable.size == 0:
        return 0.0
    frames = usable.reshape(-1, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
    active = rms >= 10.0 ** (threshold_dbfs / 20.0)
    for index in range(len(active) - confirm_frames + 1):
        if bool(np.all(active[index : index + confirm_frames])):
            return index * 10.0
    return 0.0


def _audio_stats(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    if samples.size == 0:
        raise ValueError("generated audio is empty")
    onset_ms = _audible_onset_ms(samples, sample_rate)
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    return {
        "audio_seconds": samples.size / sample_rate,
        "audible_onset_ms": onset_ms,
        "rms_dbfs": rms_dbfs,
    }


def _stream_delivery_stats(
    events: list[tuple[float, int]], sample_rate: int
) -> dict[str, float | int]:
    """Estimate playback starvation from timestamped mono PCM16 chunks."""
    if not events or sample_rate <= 0:
        return {
            "pcm_chunks": 0,
            "max_pcm_interarrival_ms": 0.0,
            "playback_starvation_total_ms": 0.0,
            "playback_starvation_max_ms": 0.0,
        }

    playout_deadline = events[0][0]
    previous_arrival = events[0][0]
    total_starvation_ms = 0.0
    max_starvation_ms = 0.0
    max_interarrival_ms = 0.0
    for arrival, byte_count in events:
        max_interarrival_ms = max(
            max_interarrival_ms,
            max(0.0, arrival - previous_arrival) * 1000.0,
        )
        starvation_ms = max(0.0, arrival - playout_deadline) * 1000.0
        total_starvation_ms += starvation_ms
        max_starvation_ms = max(max_starvation_ms, starvation_ms)
        playout_deadline = max(playout_deadline, arrival)
        playout_deadline += byte_count / (sample_rate * 2.0)
        previous_arrival = arrival

    return {
        "pcm_chunks": len(events),
        "max_pcm_interarrival_ms": max_interarrival_ms,
        "playback_starvation_total_ms": total_starvation_ms,
        "playback_starvation_max_ms": max_starvation_ms,
    }


def _write_results(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ttfa = [float(row["audible_ttfa_ms"]) for row in rows]
    first_pcm = [float(row["first_pcm_ms"]) for row in rows]
    summary = {
        **metadata,
        "samples": len(rows),
        "first_pcm_ms": {
            "median": statistics.median(first_pcm),
            "p95": _percentile(first_pcm, 0.95),
            "minimum": min(first_pcm),
            "maximum": max(first_pcm),
        },
        "audible_ttfa_ms": {
            "median": statistics.median(ttfa),
            "p95": _percentile(ttfa, 0.95),
            "minimum": min(ttfa),
            "maximum": max(ttfa),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def _bridge_sentence(
    url: str, text: str, timeout_s: float
) -> tuple[int, np.ndarray, dict[str, float | int]]:
    from websockets.asyncio.client import connect

    request_id = uuid.uuid4().hex
    request_sent = 0.0
    first_pcm_at: float | None = None
    done_at: float | None = None
    sample_rate = 0
    pcm_chunks: list[bytes] = []
    pcm_events: list[tuple[float, int]] = []

    async with asyncio.timeout(timeout_s):
        async with connect(url, max_size=None, ping_interval=20, ping_timeout=20) as websocket:
            await websocket.send(
                json.dumps({"type": "synthesize", "request_id": request_id, "text": text})
            )
            request_sent = time.perf_counter()
            async for message in websocket:
                received = time.perf_counter()
                if isinstance(message, bytes):
                    if first_pcm_at is None:
                        first_pcm_at = received
                    pcm_chunks.append(message)
                    pcm_events.append((received, len(message)))
                    continue
                event = json.loads(message)
                if str(event.get("request_id", "")) not in {"", request_id}:
                    continue
                if event.get("type") == "start":
                    sample_rate = int(event["sample_rate"])
                elif event.get("type") == "done":
                    if event.get("status") != "completed":
                        raise RuntimeError(f"bridge synthesis failed: {event}")
                    done_at = received
                    break
                elif event.get("type") == "error":
                    raise RuntimeError(str(event.get("error", "unknown bridge error")))

    if sample_rate <= 0 or first_pcm_at is None or done_at is None or not pcm_chunks:
        raise RuntimeError("bridge closed before a complete PCM response")
    pcm16 = np.frombuffer(b"".join(pcm_chunks), dtype="<i2")
    samples = pcm16.astype(np.float32) / 32767.0
    return sample_rate, samples, {
        "first_pcm_ms": (first_pcm_at - request_sent) * 1000.0,
        "synthesis_wall_ms": (done_at - request_sent) * 1000.0,
        "merge_ms": 0.0,
        "cache_audio_patches": 0,
        "cache_prompt_chars": 0,
        **_stream_delivery_stats(pcm_events, sample_rate),
    }


async def _run_bridge(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        for sentence_index, sentence in enumerate(args.sentences, 1):
            sample_rate, samples, timings = await _bridge_sentence(args.url, sentence, args.timeout)
            stats = _audio_stats(samples, sample_rate)
            output = args.output_dir / f"r{round_index}_s{sentence_index}.wav"
            _write_wav(output, sample_rate, samples)
            rows.append(
                {
                    "mode": "nano_bridge",
                    "round": round_index,
                    "sentence": sentence_index,
                    "text": sentence,
                    **timings,
                    **stats,
                    "audible_ttfa_ms": timings["first_pcm_ms"] + stats["audible_onset_ms"],
                    "sample_rate": sample_rate,
                    "output": str(output),
                }
            )
    _write_results(
        args.output_dir,
        rows,
        {"mode": "nano_bridge", "url": args.url, "sentences": list(args.sentences)},
    )


def _load_official_model(args: argparse.Namespace) -> Any:
    if args.source:
        sys.path.insert(0, str(args.source.resolve()))
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained(
        str(args.model.resolve()),
        load_denoiser=False,
        local_files_only=True,
        optimize=args.optimize,
        device=args.device,
    )


def _official_sentence(
    model: Any,
    prompt_cache: dict[str, Any],
    text: str,
    args: argparse.Namespace,
) -> tuple[int, np.ndarray, dict[str, float], dict[str, Any]]:
    torch = __import__("torch")
    request_started = time.perf_counter()
    first_pcm_at: float | None = None
    chunks: list[np.ndarray] = []
    generated_features: list[Any] = []
    generator = model.tts_model.generate_with_prompt_cache_streaming(
        target_text=text,
        prompt_cache=prompt_cache,
        inference_timesteps=args.inference_timesteps,
        cfg_value=args.cfg_value,
        streaming_prefix_len=args.streaming_prefix_len,
        seed=args.seed,
    )
    try:
        for waveform, _text_tokens, feature_sequence in generator:
            if first_pcm_at is None:
                first_pcm_at = time.perf_counter()
            chunk = waveform.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            if chunk.size:
                chunks.append(chunk)
            if not feature_sequence:
                raise RuntimeError("official VoxCPM2 returned no generated feature")
            generated_features.append(feature_sequence[-1].detach().cpu())
    finally:
        generator.close()
    synthesis_finished = time.perf_counter()

    if first_pcm_at is None or not chunks or not generated_features:
        raise RuntimeError("official VoxCPM2 returned an incomplete stream")
    new_audio_feature = torch.cat(generated_features, dim=1).squeeze(0)
    merge_started = time.perf_counter()
    merged_cache = model.tts_model.merge_prompt_cache(prompt_cache, text, new_audio_feature)
    merge_finished = time.perf_counter()
    sample_rate = int(model.tts_model.sample_rate)
    return sample_rate, np.concatenate(chunks), {
        "first_pcm_ms": (first_pcm_at - request_started) * 1000.0,
        "synthesis_wall_ms": (synthesis_finished - request_started) * 1000.0,
        "merge_ms": (merge_finished - merge_started) * 1000.0,
        "cache_audio_patches": int(merged_cache["audio_feat"].shape[0]),
        "cache_prompt_chars": len(str(merged_cache.get("prompt_text", ""))),
    }, merged_cache


def _run_official(args: argparse.Namespace) -> None:
    model = _load_official_model(args)
    base_cache = model.tts_model.build_prompt_cache(reference_wav_path=str(args.reference.resolve()))

    for _ in range(args.warmups):
        _official_sentence(model, dict(base_cache), args.sentences[0], args)

    rows: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        prompt_cache = dict(base_cache)
        for sentence_index, sentence in enumerate(args.sentences, 1):
            sample_rate, samples, timings, prompt_cache = _official_sentence(
                model, prompt_cache, sentence, args
            )
            stats = _audio_stats(samples, sample_rate)
            output = args.output_dir / f"r{round_index}_s{sentence_index}.wav"
            _write_wav(output, sample_rate, samples)
            rows.append(
                {
                    "mode": "official_prompt_cache",
                    "round": round_index,
                    "sentence": sentence_index,
                    "text": sentence,
                    **timings,
                    **stats,
                    "audible_ttfa_ms": timings["first_pcm_ms"] + stats["audible_onset_ms"],
                    "sample_rate": sample_rate,
                    "output": str(output),
                }
            )
    _write_results(
        args.output_dir,
        rows,
        {
            "mode": "official_prompt_cache",
            "model": str(args.model.resolve()),
            "reference": str(args.reference.resolve()),
            "source": str(args.source.resolve()) if args.source else None,
            "optimize": args.optimize,
            "device": args.device,
            "seed": args.seed,
            "inference_timesteps": args.inference_timesteps,
            "cfg_value": args.cfg_value,
            "streaming_prefix_len": args.streaming_prefix_len,
            "sentences": list(args.sentences),
        },
    )


def _sentences(value: Iterable[str] | None) -> tuple[str, ...]:
    result = tuple(item.strip() for item in (value or DEFAULT_SENTENCES) if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one non-empty sentence is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    bridge = subparsers.add_parser("bridge", help="measure the current Nano WebSocket bridge")
    bridge.add_argument("--url", default="ws://127.0.0.1:8770")
    bridge.add_argument("--rounds", type=int, default=3)
    bridge.add_argument("--timeout", type=float, default=120.0)
    bridge.add_argument("--sentence", action="append", dest="sentences")
    bridge.add_argument("--output-dir", type=Path, required=True)

    official = subparsers.add_parser("official", help="measure official prompt-cache continuity")
    official.add_argument("--source", type=Path)
    official.add_argument("--model", type=Path, required=True)
    official.add_argument("--reference", type=Path, required=True)
    official.add_argument("--rounds", type=int, default=3)
    official.add_argument("--warmups", type=int, default=2)
    official.add_argument("--sentence", action="append", dest="sentences")
    official.add_argument("--output-dir", type=Path, required=True)
    official.add_argument("--device", default="cuda:0")
    official.add_argument("--seed", type=int, default=42)
    official.add_argument("--inference-timesteps", type=int, default=10)
    official.add_argument("--cfg-value", type=float, default=2.0)
    official.add_argument("--streaming-prefix-len", type=int, default=4)
    official.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    args.sentences = _sentences(args.sentences)
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if getattr(args, "warmups", 0) < 0:
        parser.error("--warmups must be non-negative")
    if args.mode == "bridge":
        asyncio.run(_run_bridge(args))
    else:
        _run_official(args)


if __name__ == "__main__":
    main()
