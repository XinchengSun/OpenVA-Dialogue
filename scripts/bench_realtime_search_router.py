#!/usr/bin/env python3
"""Measure local real-time search routing overhead without network calls."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipecat_dystream" / "realtime_search.py"
SPEC = importlib.util.spec_from_file_location("realtime_search_bench", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
RealtimeSearchPolicy = MODULE.RealtimeSearchPolicy
last_user_text = MODULE.last_user_text


CASES = (
    ("杭州天气怎么样？", True),
    ("今天有什么重要新闻？", True),
    ("英伟达现在的股价是多少？", True),
    ("帮我联网查一下这个项目最近的更新", True),
    ("给我讲个笑话。", False),
    ("你现在在干嘛？", False),
    ("我们现在用的是什么模型？", False),
    ("解释天气和气候有什么区别。", False),
)


class _Context:
    def __init__(self, query: str):
        self._messages = [{"role": "user", "content": query}]

    def get_messages(self):
        return self._messages


def _percentile(values: list[float], percentile: float) -> float:
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20_000)
    args = parser.parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")

    policy = RealtimeSearchPolicy("smart")
    contexts = [(_Context(query), expected) for query, expected in CASES]
    samples_us: list[float] = []
    routed = 0

    for _ in range(100):
        for context, _ in contexts:
            policy.decide(last_user_text(context))

    for _ in range(args.rounds):
        for context, expected in contexts:
            started_at = time.perf_counter_ns()
            decision = policy.decide(last_user_text(context))
            samples_us.append((time.perf_counter_ns() - started_at) / 1000.0)
            if decision.enabled != expected:
                raise SystemExit(
                    f"routing mismatch query={last_user_text(context)!r} "
                    f"expected={expected} actual={decision.enabled}"
                )
            routed += int(decision.enabled)

    samples_us.sort()
    print(
        json.dumps(
            {
                "mode": policy.mode,
                "calls": len(samples_us),
                "routed": routed,
                "bypassed": len(samples_us) - routed,
                "mean_us": round(statistics.fmean(samples_us), 3),
                "p50_us": round(_percentile(samples_us, 50), 3),
                "p95_us": round(_percentile(samples_us, 95), 3),
                "p99_us": round(_percentile(samples_us, 99), 3),
                "max_us": round(max(samples_us), 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
