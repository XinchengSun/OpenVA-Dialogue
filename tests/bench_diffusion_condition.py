"""CPU component benchmark of the real diffusion head, not end-to-end latency.

Run from the checkout: python -m tests.bench_diffusion_condition
Requires torch and timm; does not download checkpoints or run CUDA graphs.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time

import torch
import timm

from tests.test_diffusion_condition_reuse import DiffusionHead, MODEL_PATH, original_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("steps, repeats and threads must be positive; warmup must be non-negative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261026)
    head = DiffusionHead(face_dim=512, hidden_size=768, num_layers=6,
                         num_heads=12, dropout=0.1).eval()
    gpt = torch.randn(4, 1, 512)
    noisy = torch.randn(4, 1, 512)
    times = torch.randn(args.steps, 1, 1, 768)

    def run(reuse):
        condition = head.prepare_gpt_condition(gpt) if reuse else None
        latent = noisy.clone()
        for step in range(args.steps):
            prediction = head(latent, gpt, times[step], gpt_condition=condition)
            sigma = 1.0 - step / args.steps
            latent = latent - (1.0 / args.steps) * ((latent - prediction) / (sigma + 1e-9))
        return latent

    timings = {"baseline": [], "reused_condition": []}
    with torch.inference_mode():
        reference = original_forward(head, noisy, gpt, times[0])
        torch.testing.assert_close(head(noisy, gpt, times[0]), reference, rtol=0, atol=0)
        expected, actual = run(False), run(True)
        max_abs_error = (expected - actual).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for _ in range(args.warmup):
            run(False)
            run(True)
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic first/second-run bias.
            for reuse in ((False, True) if repeat % 2 == 0 else (True, False)):
                started = time.perf_counter_ns()
                run(reuse)
                timings["reused_condition" if reuse else "baseline"].append(
                    (time.perf_counter_ns() - started) / 1e6)

    medians = {key: statistics.median(value) for key, value in timings.items()}
    report = {
        "scope": "CPU eager real production diffusion head; random weights; synthetic Euler schedule; no CUDA graphs or full avatar pipeline",
        "torch_version": torch.__version__, "timm_version": timm.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "threads": torch.get_num_threads(), "seed": 20261026,
        "shape": {"face_dim": 512, "hidden_size": 768, "layers": 6, "cfg_batch": 4},
        "shape_basis": "app.py passes cfg positionally via utils.instantiate_motion_gen; Audio2FaceGPT uses constructor hidden_size=768, not cfg.hidden_size=512",
        "steps": args.steps, "warmup_pairs": args.warmup, "repeat_pairs": args.repeats,
        "current_source_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "max_abs_output_error": max_abs_error,
        "condition_linear_calls_per_module": {"baseline": args.steps, "candidate": 1},
        "samples_ms": timings, "median_ms": medians,
        "component_latency_reduction_pct": 100 * (1 - medians["reused_condition"] / medians["baseline"]),
        "end_to_end_latency_reduction_pct": None,
        "cuda_latency_reduction_pct": None,
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
