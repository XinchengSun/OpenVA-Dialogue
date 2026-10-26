"""CPU microbenchmarks of production LIA constant caches, not rendering FPS."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
LIA_DIR = ROOT / "tools/visualization_0416/utils/model_0506/model/head_animation/LIA_3d"


def load_production_class(filename: str, class_name: str):
    """Load the unchanged class body without the demo's unrelated CUDA imports."""
    path = LIA_DIR / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[class_name]


class OriginalDirection(nn.Module):
    """Frozen pre-change arithmetic from commit 40153f8, with identical weights."""
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())

    def forward(self, value):
        q, _ = torch.qr(self.weight + 1e-8)
        if value is None:
            return q
        out = torch.matmul(torch.diag_embed(value), q.T)
        return torch.sum(out, dim=1)


class OriginalWSConv3d(nn.Conv3d):
    """Frozen pre-change weight standardization, including reduction order."""
    def forward(self, inp):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return torch.nn.functional.conv3d(inp, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def benchmark(*, component="direction", repeats=11, iterations=None, warmup=10, threads=1, seed=20261026):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if component == "direction":
        direction_class = load_production_class("flow_estimator.py", "Direction")
        optimized = direction_class(512, 64).eval()
        original = OriginalDirection(optimized.weight).eval()
        inputs = torch.randn(16, 1, 64)
        dimensions = {"latent_dim": 512, "motion_space": 64, "batch": 1}
        iterations = iterations or 100
    else:
        conv_class = load_production_class("../VASA1/building_blocks.py", "WSConv3d")
        optimized = conv_class(512, 256, 3, padding=1, bias=False).eval()
        original = OriginalWSConv3d(512, 256, 3, padding=1, bias=False).eval()
        original.load_state_dict(optimized.state_dict())
        # This is the real first warping residual block's input/weight shape.
        inputs = torch.randn(3, 1, 512, 8, 8, 8)
        dimensions = {"input_shape": [1, 512, 8, 8, 8], "weight_shape": [256, 512, 3, 3, 3]}
        iterations = iterations or 3
    samples = {"baseline": [], "optimized": []}
    with warnings.catch_warnings(), torch.inference_mode():
        warnings.simplefilter("ignore", UserWarning)
        for value in inputs:
            torch.testing.assert_close(optimized(value), original(value), rtol=0, atol=0)
        for index in range(warmup):
            original(inputs[index % len(inputs)])
            optimized(inputs[index % len(inputs)])
        # Alternate execution order to reduce a fixed warm-cache/order bias.
        for repeat in range(repeats):
            pairs = [("baseline", original), ("optimized", optimized)]
            if repeat % 2:
                pairs.reverse()
            for label, module in pairs:
                start = time.perf_counter_ns()
                for index in range(iterations):
                    module(inputs[index % len(inputs)])
                samples[label].append((time.perf_counter_ns() - start) / iterations / 1e6)
        standardization_samples = []
        if component == "wsconv3d":
            for _ in range(repeats):
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    optimized._standardized_weight()
                standardization_samples.append((time.perf_counter_ns() - start) / iterations / 1e6)
    medians = {key: statistics.median(values) for key, values in samples.items()}
    source_paths = [Path(__file__), LIA_DIR / "flow_estimator.py", LIA_DIR / "face_generator.py",
                    LIA_DIR / "../VASA1/building_blocks.py"]
    return {
        "scope": "CPU component steady-state microbenchmark using random weights; no complete face renderer, pretrained weights, GPU or end-to-end latency",
        "component": component,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "device": "cpu", "threads": threads, "seed": seed,
        **dimensions,
        "warmup": warmup, "repeats": repeats, "iterations_per_sample": iterations,
        "output_comparison": f"{len(inputs)} inputs exactly equal (rtol=0, atol=0)",
        "p50_ms": medians, "sample_ms": samples,
        "uncached_standardization_only_p50_ms": statistics.median(standardization_samples) if standardization_samples else None,
        "uncached_standardization_only_samples_ms": standardization_samples,
        "component_speedup": medians["baseline"] / medians["optimized"],
        "constant_recomputations_per_steady_state_call": {"baseline": 1, "optimized": 0},
        "cache_bytes": 512 * 64 * 4 if component == "direction" else 256 * 512 * 27 * 4,
        "renderer_four_wsconv3d_cache_bytes": (512*256 + 256*128 + 128*64 + 64*32) * 27 * 4,
        "grid_bytes_previously_copied_per_gpu_frame": 16 * 64 * 64 * 3 * 4,
        "grid_note": "Analytical byte count; device-copy improvement is not timed by this CPU benchmark",
        "source_sha256": {str(path.resolve().relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("direction", "wsconv3d"), default="direction")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.repeats, args.threads) <= 0 or args.warmup < 0 or (args.iterations is not None and args.iterations <= 0):
        parser.error("repeats, iterations and threads must be positive; warmup must be non-negative")
    print(json.dumps(benchmark(**vars(args)), indent=2))
