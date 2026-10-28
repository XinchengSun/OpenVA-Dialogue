"""Strict renderer equivalence with a local pinned checkpoint and synthetic media."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.bench_renderer_reuse import compare_outputs, compare_records, state_hash
from tests.bench_renderer_memory import make_renderer
from tests.reference_encoder_fixture import HEAD, load_encoders
from inference_memory import cache_renderer_spectral_norm, cuda_storage_inventory


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def make_flow():
    path = HEAD / "LIA/modules.py"
    modules = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), modules)
    path = HEAD / "LIA_3d/flow_estimator.py"
    nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(n, ast.ClassDef) and n.name in ("Direction", "FlowEstimator")]
    namespace = dict(torch=torch, nn=nn, F=F, EqualLinear=modules["EqualLinear"])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FlowEstimator"](latent_dim=512, motion_space=64).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--variant", choices=("baseline", "spectral", "lifetimes", "both"), default="baseline")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new")
    digest = sha256_file(args.checkpoint)
    if digest != args.expected_sha256:
        parser.error("checkpoint SHA-256 mismatch")
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(20261028)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["state_dict"]
    encoders = load_encoders()
    model, flow = make_renderer(), make_flow()
    modules = {**encoders, "face_generator": model, "flow_estimator": flow}
    counts = {}
    for name, module in modules.items():
        prefix = name + "."
        values = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if name == "face_generator":
            # This pinned upstream checkpoint predates the unused warp_layer.
            # app.py also retains its initialized value via strict=False. Allow
            # only that known key, and hash its value along with the loaded state.
            missing, unexpected = module.load_state_dict(values, strict=False)
            assert missing == ["warp_layer.weight"] and not unexpected, (missing, unexpected)
        else:
            module.load_state_dict(values, strict=True)
        counts[name] = len(values)
        module.cuda()
    original_state = state_hash(model)
    with torch.inference_mode():
        # Fully synthetic reference: no private portrait or third-party identity.
        axis = torch.linspace(-1, 1, 512, device="cuda")
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        image = torch.stack((xx, yy, torch.sin(xx * 3) * torch.cos(yy * 3))).unsqueeze(0)
        features = encoders["face_encoder"](image)
        anchor = encoders["motion_encoder"](image)[0]
        latents = [anchor + torch.randn_like(anchor) * 0.01 for _ in range(4)]
        for encoder in encoders.values():
            encoder.cpu()
        del image
        torch.cuda.empty_cache()
        report = {"scope": "Trained LIA encoders/flow/generator; synthetic reference image and motion; not dialogue or perceptual-quality evaluation",
                  "variant": args.variant, "checkpoint_sha256": digest, "loaded_keys": counts,
                  "checkpoint_compatibility": {"initialized_unused_key": "face_generator.warp_layer.weight"},
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                  "state_sha256": original_state, "input_shapes": [[1, 512], [1, 32, 16, 64, 64]],
                  "math_modes": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                 "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                 "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                  "input_sha256": {"features": hashlib.sha256(features.cpu().numpy().tobytes()).hexdigest(),
                                   "latents": hashlib.sha256(torch.stack(latents).cpu().numpy().tobytes()).hexdigest()}}
        if args.variant in ("spectral", "both"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            report["spectral_cache"] = cache_renderer_spectral_norm(torch, model)
            torch.cuda.synchronize()
            report["cache_install_wall_ms"] = (time.perf_counter() - started) * 1000
            assert report["spectral_cache"]["enabled"]
        model._release_render_intermediates = args.variant in ("lifetimes", "both")
        for _ in range(3):
            model(flow(anchor, latents[0]), features)
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
        outputs = np.stack([model(flow(anchor, value), features).cpu().numpy() for value in latents])
        for hook in hooks:
            hook.remove()
        report["layer_outputs"] = layers
        if args.reference:
            reference = json.loads((args.reference.parent / "result.json").read_text(encoding="utf-8"))
            compare_records(reference, report)
            assert report["input_sha256"] == reference["input_sha256"]
            compare_outputs(np.load(args.reference), outputs)
        else:
            compare_outputs(outputs, np.stack([model(flow(anchor, value), features).cpu().numpy() for value in latents]))
        np.save(args.output_dir / "outputs.npy", outputs)
        assert state_hash(model) == original_state
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        output = model(flow(anchor, latents[0]), features)
        torch.cuda.synchronize()
        report["memory"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                            "reserved_bytes": torch.cuda.memory_reserved(),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            **cuda_storage_inventory(torch, {"generator": model, "flow": flow})}
        report["comparison"] = {"frames": 4, "intermediate_records": len(layers), "bits_equal": True,
                                "max_absolute_error": 0.0, "output_std": float(outputs.std())}
    report["benchmark_sha256"] = sha256_file(Path(__file__))
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"variant": args.variant, "comparison": report["comparison"],
                      "peak_mib": report["memory"]["peak_allocated_bytes"] / 2**20}))


if __name__ == "__main__":
    main()
