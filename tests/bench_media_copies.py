"""CPU packing microbenchmark; excludes ffmpeg, GPU and network latency."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_media_copy_performance import SOURCE, load_client, make_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="40153f85414c505238c6e948ce0aa45e5be076c6")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    old_source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:server_mse.py"], cwd=SOURCE.parent).decode("utf-8")
    clients = {"baseline": make_client(load_client(old_source), record=False),
               "candidate": make_client(record=False)}
    frames = np.random.default_rng(17).integers(0, 256, (4, 512, 512, 3), dtype=np.uint8)
    audio = np.linspace(-1.0, 1.0, 4 * 640, dtype=np.float32)
    timings = {name: [] for name in clients}
    for index in range(args.warmup + args.repeats):
        names = list(clients) if index % 2 else list(reversed(clients))
        for name in names:
            started = time.perf_counter_ns()
            clients[name]._write_to_ffmpeg(0, frames, audio)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if index >= args.warmup:
                timings[name].append(elapsed)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    print(json.dumps({
        "scope": "CPU production _write_to_ffmpeg packing; sink replaces encoder queue",
        "baseline_commit": args.baseline,
        "candidate_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "seed": 17,
        "frames_shape": list(frames.shape), "frame_stride": 2,
        "warmup": args.warmup, "repeats": args.repeats,
        "p50_ms": medians, "samples_ms": timings,
        "packing_speedup": medians["baseline"] / medians["candidate"],
        "removed_normal_segment_copy_bytes_per_client": frames.nbytes,
        "end_to_end_speedup": None,
    }, indent=2))


if __name__ == "__main__":
    main()
