"""Small, dependency-free GPU picker for the realtime avatar service.

It selects at most two cards for motion/render (the service-wide hard limit is
four) and never hot-migrates a running process.  A restart is required to
apply a new choice.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

MAX_CARDS = 4


def snapshot() -> List[Dict[str, Any]]:
    query = "index,utilization.gpu,memory.used,memory.total,pstate"
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=3,
    )
    cards = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, util, used, total, pstate = parts
        cards.append({"index": int(idx), "utilization": float(util),
                      "memory_used_mib": float(used), "memory_total_mib": float(total),
                      "pstate": pstate})
    return cards


def choose(cards: List[Dict[str, Any]], required: int = 2) -> List[int]:
    if required < 1 or required > MAX_CARDS:
        raise ValueError("required GPU count exceeds hard limit")
    # Avoid active workloads; leave a little headroom for driver noise.
    candidates = [c for c in cards if c["utilization"] < 25.0 and
                  c["memory_used_mib"] / max(c["memory_total_mib"], 1.0) < 0.35]
    candidates.sort(key=lambda c: (c["utilization"], c["memory_used_mib"]))
    if len(candidates) < required:
        raise RuntimeError("fewer than two idle GPUs are available")
    return [int(c["index"]) for c in candidates[:required]]


def main() -> None:
    cards = snapshot()
    selected = choose(cards, 2)
    # The caller receives physical ids; CUDA logical ids remain 0 and 1.
    print(json.dumps({"cards_limit": MAX_CARDS, "selected": selected,
                      "cards": cards}, separators=(",", ":")))


if __name__ == "__main__":
    main()
