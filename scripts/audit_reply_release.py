#!/usr/bin/env python3
"""Replay text-release and turn-stop policies; never call ASR/LLM/TTS services.

This is a controlled diagnostic, not a model or end-to-end latency benchmark.
AST loading executes the unchanged production parser and the supplied upstream
functions without installing their optional media dependencies.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_nodes(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
    if len(nodes) != len(names):
        raise ValueError(f"missing or duplicate diagnostic definitions: {path.name}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_parser():
    path = ROOT / "pipecat_dystream/resilient_llm.py"
    return load_nodes(path, {
        "_SENTENCE_TERMINATORS", "_SENTENCE_CLOSERS",
        "_terminator_width", "_split_complete_sentences",
    }, {})["_split_complete_sentences"]


def parser_replay(parser, chunks, eof_ms):
    pending = ""
    released = []
    for at_ms, text in chunks:
        pending += text
        sentences, pending = parser(pending, final=False)
        released.extend((at_ms, sentence) for sentence in sentences)
    sentences, pending = parser(pending, final=True)
    released.extend((eof_ms, sentence) for sentence in sentences)
    if pending.strip():
        released.append((eof_ms, pending))
    return released


def livetalking_replay(path, chunks, eof_ms):
    clock = SimpleNamespace(now=0)
    released = []

    def stream(**_kwargs):
        for at_ms, text in chunks:
            clock.now = at_ms
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        clock.now = eof_ms

    logger = SimpleNamespace(info=lambda *_a: None, exception=lambda *_a: None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=stream)))
    namespace = load_nodes(path, {"llm_response"}, {
        "time": SimpleNamespace(perf_counter=lambda: clock.now / 1000),
        "logger": logger,
        "_llm_client": lambda _opt: client,
        "_llm_model": lambda _opt: "synthetic-no-provider",
    })
    avatar = SimpleNamespace(
        opt=SimpleNamespace(),
        put_msg_txt=lambda text, _metadata: released.append((clock.now, text)),
    )
    namespace["llm_response"]("synthetic diagnostic", avatar)
    return released


def release_summary(released, expected):
    actual = "".join(text for _, text in released)
    if actual != expected:
        raise AssertionError("text changed during controlled replay")
    return {
        "first_release_ms": released[0][0] if released else None,
        "requests": len(released),
        "released_characters": len(actual),
        "text_sha256": hashlib.sha256(actual.encode()).hexdigest(),
    }


async def timer_replay(source: Path, final_delay: float):
    """Execute upstream's actual timer methods with minimal frame/task adapters."""
    class Frame:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    frame_names = ["InterimTranscriptionFrame", "STTMetadataFrame", "TranscriptionFrame",
                   "VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"]
    namespace = {name: type(name, (Frame,), {}) for name in frame_names}

    class TaskManager:
        def create_task(self, coroutine, name):
            return asyncio.create_task(coroutine, name=name)

        async def cancel_task(self, task):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    class BaseStrategy:
        def __init__(self, **_kwargs):
            self.task_manager = TaskManager()
            self.stopped = asyncio.Event()
            self.stopped_at = None

        async def trigger_user_turn_stopped(self):
            self.stopped_at = time.perf_counter()
            self.stopped.set()

    namespace.update({
        "asyncio": asyncio, "VAD_STOP_SECS": 0.2,
        "logger": SimpleNamespace(debug=lambda *_a: None, warning=lambda *_a: None),
        "BaseUserTurnStopStrategy": BaseStrategy,
        "ProcessFrameResult": SimpleNamespace(CONTINUE=None),
    })
    cls = load_nodes(source, {"SpeechTimeoutUserTurnStopStrategy"}, namespace)[
        "SpeechTimeoutUserTurnStopStrategy"
    ]
    strategy = cls(user_speech_timeout=0.45)
    try:
        await strategy.process_frame(namespace["STTMetadataFrame"](ttfs_p99_latency=2.0))
        await strategy.process_frame(namespace["VADUserStartedSpeakingFrame"]())
        started = time.perf_counter()
        await strategy.process_frame(namespace["VADUserStoppedSpeakingFrame"](
            stop_secs=0.2, timestamp=time.monotonic(),
        ))
        await asyncio.sleep(final_delay)
        final_at = time.perf_counter()
        await strategy.process_frame(namespace["TranscriptionFrame"](text="测试", finalized=True))
        await asyncio.wait_for(strategy.stopped.wait(), timeout=3)
        return {
            "injected_final_delay_ms": final_delay * 1000,
            "vad_stop_to_final_ms": (final_at - started) * 1000,
            "vad_stop_to_turn_stop_ms": (strategy.stopped_at - started) * 1000,
            "final_to_turn_stop_ms": (strategy.stopped_at - final_at) * 1000,
        }
    finally:
        await strategy._cancel_all_tasks()


async def run(args):
    parser = load_parser()
    fixtures = {
        "long_first_sentence": ([(0, "这个问题需要结合你的使用场景，"),
                                  (300, "先把用途和预算说清楚，"),
                                  (600, "再根据这些条件选择合适的方案。"),
                                  (850, "这样更容易找到适合你的产品。")], 900),
        "complete_sentence_then_delayed_eof": ([(0, "这是一句已经完整生成的回答。")], 400),
        "no_punctuation": ([(0, "这是一个没有标点的完整回答")], 700),
        "closing_quote": ([(0, "他说：“你好。"), (80, "”然后离开。")], 200),
    }
    results = {}
    for name, (chunks, eof_ms) in fixtures.items():
        expected = "".join(text for _, text in chunks)
        row = {"openva": release_summary(parser_replay(parser, chunks, eof_ms), expected),
               "injected_eof_ms": eof_ms}
        if args.livetalking_source:
            row["livetalking"] = release_summary(
                livetalking_replay(args.livetalking_source, chunks, eof_ms), expected,
            )
        results[name] = row

    # Real wall-clock confirmation: a supplied 300 ms provider pause, no model.
    started = time.perf_counter()
    released, pending = parser("这是一句已经完整生成的回答。", final=False)
    assert not released
    await asyncio.sleep(0.3)
    released, pending = parser(pending, final=True)
    assert released and not pending
    wall_ms = (time.perf_counter() - started) * 1000

    paths = [ROOT / "pipecat_dystream/resilient_llm.py"]
    paths.extend(path for path in (args.livetalking_source, args.pipecat_source) if path)
    report = {
        "scope": "Controlled policy replay only; injected provider times are not model/E2E measurements",
        "production_behavior_changed": False,
        "fixtures": results,
        "wall_clock_injected_300ms_eof_pause_release_ms": wall_ms,
        "sources": [{"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in paths],
    }
    if args.pipecat_source:
        report["upstream_actual_timer_replays"] = [
            await timer_replay(args.pipecat_source, delay) for delay in (0.05, 0.7)
        ]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--livetalking-source", type=Path)
    parser.add_argument("--pipecat-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
