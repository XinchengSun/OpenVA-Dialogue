#!/usr/bin/env python3
"""Exercise one real multi-sentence VoxCPM2 context and its barge-in path.

The probe intentionally overlaps sentence requests, records every PCM arrival,
simulates continuous playback from those arrival times, releases the completed
context, then measures cancel-to-done and a fresh post-cancel request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SENTENCES = (
    "当然可以，我先把最重要的结论告诉你。",
    "这个方案会保持实时生成，也会让前后声音更加稳定。",
    "如果首声出现变慢，我们就立即回退当前版本。",
)


@dataclass
class SynthesisResult:
    request_id: str
    context_id: str
    text: str
    request_sent: float = 0.0
    first_pcm: float | None = None
    done_at: float | None = None
    sample_rate: int = 0
    pcm_chunks: list[bytes] = field(default_factory=list)
    arrivals: list[tuple[float, int]] = field(default_factory=list)

    @property
    def pcm_bytes(self) -> bytes:
        return b"".join(self.pcm_chunks)

    @property
    def audio_ms(self) -> float:
        return len(self.pcm_bytes) / 2 / self.sample_rate * 1000.0

    def metrics(self, epoch: float) -> dict[str, Any]:
        if self.first_pcm is None or self.done_at is None:
            raise RuntimeError(f"request did not finish: {self.request_id}")
        return {
            "request_id": self.request_id,
            "context_id": self.context_id,
            "text": self.text,
            "request_sent_ms": (self.request_sent - epoch) * 1000.0,
            "first_pcm_ms": (self.first_pcm - self.request_sent) * 1000.0,
            "first_pcm_from_epoch_ms": (self.first_pcm - epoch) * 1000.0,
            "done_ms": (self.done_at - self.request_sent) * 1000.0,
            "audio_ms": self.audio_ms,
            "chunks": len(self.pcm_chunks),
            "pcm_bytes": len(self.pcm_bytes),
        }


async def synthesize(
    url: str,
    text: str,
    context_id: str,
    *,
    timeout_s: float,
    request_sent_event: asyncio.Event | None = None,
    first_pcm_event: asyncio.Event | None = None,
) -> SynthesisResult:
    from websockets.asyncio.client import connect

    result = SynthesisResult(uuid.uuid4().hex, context_id, text)
    async with asyncio.timeout(timeout_s):
        async with connect(url, max_size=None, ping_interval=20, ping_timeout=20) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "synthesize",
                        "request_id": result.request_id,
                        "context_id": context_id,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
            )
            result.request_sent = time.perf_counter()
            if request_sent_event is not None:
                request_sent_event.set()
            async for message in websocket:
                received = time.perf_counter()
                if isinstance(message, bytes):
                    if result.first_pcm is None:
                        result.first_pcm = received
                        if first_pcm_event is not None:
                            first_pcm_event.set()
                    result.pcm_chunks.append(message)
                    result.arrivals.append((received, len(message) // 2))
                    continue
                event = json.loads(message)
                if str(event.get("request_id", "")) not in {"", result.request_id}:
                    raise RuntimeError(f"foreign request event: {event}")
                event_type = event.get("type")
                if event_type == "start":
                    result.sample_rate = int(event["sample_rate"])
                elif event_type == "done":
                    if event.get("status") != "completed":
                        raise RuntimeError(f"synthesis did not complete: {event}")
                    result.done_at = received
                    break
                elif event_type == "error":
                    raise RuntimeError(str(event.get("error", "unknown bridge error")))
                else:
                    raise RuntimeError(f"unknown bridge event: {event}")

    if (
        result.sample_rate <= 0
        or result.first_pcm is None
        or result.done_at is None
        or not result.pcm_chunks
    ):
        raise RuntimeError(f"incomplete synthesis response: {result.request_id}")
    return result


async def release_context(url: str, context_id: str, timeout_s: float) -> float:
    from websockets.asyncio.client import connect

    started = time.perf_counter()
    async with asyncio.timeout(timeout_s):
        async with connect(url, max_size=None) as websocket:
            await websocket.send(
                json.dumps({"type": "release_context", "context_id": context_id})
            )
            event = json.loads(await websocket.recv())
    if event != {"type": "released", "context_id": context_id}:
        raise RuntimeError(f"invalid release response: {event}")
    return (time.perf_counter() - started) * 1000.0


def simulate_playback(results: list[SynthesisResult]) -> dict[str, Any]:
    cursor: float | None = None
    total_gap = 0.0
    boundary_gap = 0.0
    sentence_gaps: list[float] = []
    for sentence_index, result in enumerate(results):
        sentence_gap = 0.0
        for arrival, samples in result.arrivals:
            if cursor is None:
                cursor = arrival
            if arrival > cursor:
                gap = arrival - cursor
                total_gap += gap
                sentence_gap += gap
                if sentence_index > 0:
                    boundary_gap += gap
                cursor = arrival
            cursor += samples / result.sample_rate
        sentence_gaps.append(sentence_gap * 1000.0)
    return {
        "total_predicted_underflow_ms": total_gap * 1000.0,
        "boundary_predicted_gap_ms": boundary_gap * 1000.0,
        "sentence_predicted_gap_ms": sentence_gaps,
    }


async def cancel_probe(
    url: str,
    text: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    from websockets.asyncio.client import connect

    request_id = uuid.uuid4().hex
    context_id = f"cancel-{uuid.uuid4().hex}"
    sample_rate = 0
    first_pcm_at: float | None = None
    cancel_sent_at: float | None = None
    done_at: float | None = None
    late_pcm_chunks = 0
    async with asyncio.timeout(timeout_s):
        async with connect(url, max_size=None, ping_interval=20, ping_timeout=20) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "synthesize",
                        "request_id": request_id,
                        "context_id": context_id,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
            )
            request_sent = time.perf_counter()
            async for message in websocket:
                received = time.perf_counter()
                if isinstance(message, bytes):
                    if first_pcm_at is None:
                        first_pcm_at = received
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "cancel",
                                    "request_id": request_id,
                                    "context_id": context_id,
                                    "discard_context": True,
                                }
                            )
                        )
                        cancel_sent_at = time.perf_counter()
                    elif cancel_sent_at is not None:
                        late_pcm_chunks += 1
                    continue
                event = json.loads(message)
                if event.get("type") == "start":
                    sample_rate = int(event["sample_rate"])
                elif event.get("type") == "done":
                    done_at = received
                    if event.get("status") != "cancelled":
                        raise RuntimeError(f"cancel returned wrong status: {event}")
                    break
                elif event.get("type") == "error":
                    raise RuntimeError(str(event.get("error", "unknown bridge error")))
                else:
                    raise RuntimeError(f"unknown bridge event: {event}")

    if first_pcm_at is None or cancel_sent_at is None or done_at is None:
        raise RuntimeError("cancel probe did not reach all timing points")
    return {
        "context_id": context_id,
        "sample_rate": sample_rate,
        "first_pcm_ms": (first_pcm_at - request_sent) * 1000.0,
        "cancel_to_done_ms": (done_at - cancel_sent_at) * 1000.0,
        "late_pcm_chunks_on_raw_bridge": late_pcm_chunks,
    }


def write_wav(path: Path, sample_rate: int, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


async def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    epoch = time.perf_counter()
    context_id = f"turn-{uuid.uuid4().hex}"
    first_pcm = asyncio.Event()
    second_sent = asyncio.Event()
    first_task = asyncio.create_task(
        synthesize(
            args.url,
            args.sentences[0],
            context_id,
            timeout_s=args.timeout,
            first_pcm_event=first_pcm,
        )
    )
    await asyncio.wait_for(first_pcm.wait(), timeout=args.timeout)
    second_task = asyncio.create_task(
        synthesize(
            args.url,
            args.sentences[1],
            context_id,
            timeout_s=args.timeout,
            request_sent_event=second_sent,
        )
    )
    await asyncio.wait_for(second_sent.wait(), timeout=args.timeout)
    third_task = asyncio.create_task(
        synthesize(
            args.url,
            args.sentences[2],
            context_id,
            timeout_s=args.timeout,
        )
    )
    results = list(await asyncio.gather(first_task, second_task, third_task))
    release_ms = await release_context(args.url, context_id, args.timeout)

    for index, result in enumerate(results, 1):
        write_wav(
            args.output_dir / f"turn_s{index}.wav",
            result.sample_rate,
            result.pcm_bytes,
        )

    cancellation = await cancel_probe(
        args.url,
        args.cancel_text,
        timeout_s=args.timeout,
    )
    post_cancel = await synthesize(
        args.url,
        args.post_cancel_text,
        f"post-cancel-{uuid.uuid4().hex}",
        timeout_s=args.timeout,
    )
    await release_context(args.url, post_cancel.context_id, args.timeout)
    write_wav(
        args.output_dir / "post_cancel.wav",
        post_cancel.sample_rate,
        post_cancel.pcm_bytes,
    )

    report = {
        "url": args.url,
        "context_id": context_id,
        "sentences": [result.metrics(epoch) for result in results],
        "playback": simulate_playback(results),
        "release_ms": release_ms,
        "cancel": cancellation,
        "post_cancel": post_cancel.metrics(post_cancel.request_sent),
    }
    output = args.output_dir / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8770")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sentence", action="append", dest="sentences")
    parser.add_argument(
        "--cancel-text",
        default="这是一次打断响应测试，我会继续说足够长的内容来验证系统能否及时停止旧的语音生成。",
    )
    parser.add_argument("--post-cancel-text", default="打断后的新一轮应该立即恢复。")
    args = parser.parse_args()
    args.sentences = tuple(args.sentences or DEFAULT_SENTENCES)
    if len(args.sentences) != 3 or any(not text.strip() for text in args.sentences):
        parser.error("exactly three non-empty --sentence values are required")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
