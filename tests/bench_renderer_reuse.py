"""Fixed FP32 renderer component judge: profile, time, or check (not E2E)."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.bench_renderer_memory import make_renderer
from inference_memory import cuda_storage_inventory


def compare_outputs(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("output shape/dtype changed")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("non-finite output")
    if reference.tobytes() != candidate.tobytes():
        raise ValueError("output bits changed")


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_records(reference, candidate):
    for key in ("state_sha256", "input_shapes", "math_modes", "layer_outputs"):
        if reference[key] != candidate[key]:
            raise ValueError("reference mismatch: " + key)


def gpu_sample():
    command = ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("time", "check", "profile"))
    parser.add_argument("--variant", choices=("baseline", "spectral", "lifetimes", "both"), default="baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=35)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("positive warmup/iterations required")
    args.output_dir.mkdir(parents=True)
    before_gpu = gpu_sample()
    torch.manual_seed(20261028)
    model = make_renderer().cuda()
    original_state = state_hash(model)
    report = {"scope": "Full LIA FaceGenerator, random FP32 weights, synthetic features; not full pipeline",
              "mode": args.mode, "variant": args.variant, "state_sha256": original_state,
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "python": sys.version, "gpu": torch.cuda.get_device_name(0),
                              "cudnn_version": torch.backends.cudnn.version()},
              "math_modes": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                             "cudnn_benchmark": torch.backends.cudnn.benchmark,
                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
              "gpu_before": before_gpu, "input_shapes": [[1, 512], [1, 32, 16, 64, 64]],
              "warmup": args.warmup, "iterations_per_window": args.iterations,
              "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT / "inference_memory.py",
                                          ROOT / "tools/visualization_0416/utils/model_0506/model/head_animation/LIA_3d/face_generator.py")}}
    if args.variant in ("spectral", "both"):
        from inference_memory import cache_renderer_spectral_norm
        report["spectral_cache"] = cache_renderer_spectral_norm(torch, model)
        assert report["spectral_cache"]["enabled"]
    model._release_render_intermediates = args.variant in ("lifetimes", "both")
    assert state_hash(model) == original_state
    latent = [torch.randn(1, 512, device="cuda") * 0.1 for _ in range(4)]
    feature = torch.randn(1, 32, 16, 64, 64, device="cuda") * 0.1
    with torch.inference_mode():
        for i in range(args.warmup):
            model(latent[i % 4], feature)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if args.mode == "time":
            windows = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                started_ns = time.time_ns()
                started = time.perf_counter()
                start.record()
                for i in range(args.iterations):
                    model(latent[i % 4], feature)
                end.record()
                end.synchronize()
                windows.append({"cuda_ms_per_frame": start.elapsed_time(end) / args.iterations,
                                "wall_ms_per_frame": (time.perf_counter() - started) * 1000 / args.iterations,
                                "started_ns": started_ns, "finished_ns": time.time_ns(),
                                "gpu": gpu_sample()})
            report["windows"] = windows
            values = [w["cuda_ms_per_frame"] for w in windows]
            report["median_cuda_ms"] = statistics.median(values)
            report["relative_window_spread"] = (max(values) - min(values)) / statistics.median(values)
        elif args.mode == "check":
            layers = []
            def capture(name):
                def hook(module, inputs, output):
                    value = output.detach().cpu().contiguous().numpy()
                    if not np.isfinite(value).all():
                        raise ValueError("non-finite intermediate: " + name)
                    layers.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                                   "sha256": hashlib.sha256(value.tobytes()).hexdigest()})
                return hook
            hooks = [module.register_forward_hook(capture(name)) for name, module in model.named_children()]
            outputs = np.stack([model(value, feature).cpu().numpy() for value in latent])
            for hook in hooks:
                hook.remove()
            report["layer_outputs"] = layers
            if args.reference:
                compare_outputs(np.load(args.reference), outputs)
                compare_records(json.loads((args.reference.parent / "result.json").read_text(encoding="utf-8")), report)
            else:
                compare_outputs(outputs, np.stack([model(value, feature).cpu().numpy() for value in latent]))
            np.save(args.output_dir / "outputs.npy", outputs)
            report["comparison"] = {"frames": 4, "bits_equal": True,
                                    "reference": str(args.reference) if args.reference else "baseline self-repeat"}
        else:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as profiler:
                for value in latent[:2]:
                    model(value, feature)
                torch.cuda.synchronize()
            report["profile"] = [{"name": e.key, "device_type": str(e.device_type), "calls": e.count,
                                  "self_device_us": e.self_device_time_total, "device_us": e.device_time_total}
                                 for e in profiler.key_averages()]
        # Identical memory window independent of profiler/timing workspaces.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        output = model(latent[0], feature)
        torch.cuda.synchronize()
        report["memory"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                            "reserved_bytes": torch.cuda.memory_reserved(),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            **cuda_storage_inventory(torch, {"renderer": model})}
        del output
    assert state_hash(model) == original_state
    report["state_preserved"] = True
    report["gpu_after"] = gpu_sample()
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"variant": args.variant, "mode": args.mode,
                      "median_cuda_ms": report.get("median_cuda_ms"),
                      "peak_mib": report["memory"]["peak_allocated_bytes"] / 2**20,
                      "allocated_mib": report["memory"]["allocated_bytes"] / 2**20,
                      "reserved_mib": report["memory"]["reserved_bytes"] / 2**20}))


if __name__ == "__main__":
    main()
