"""Full LIA architecture CUDA memory profile with synthetic features/weights."""

import argparse
import ast
import json
import math
import os
from pathlib import Path
import sys
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_memory import MemoryRecorder


def make_renderer(baseline_commit=None):
    base = ROOT / "tools/visualization_0416/utils/model_0506/model/head_animation"
    util, blocks = {}, {}
    for namespace, path in ((util, base / "LIA_3d/util.py"), (blocks, base / "VASA1/building_blocks.py")):
        if baseline_commit is not None and namespace is util:
            source = subprocess.check_output(["git", "show", f"{baseline_commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT).decode("utf-8")
        else:
            source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), namespace)
    namespace = dict(util)
    namespace.update({name: blocks[name] for name in ("USE_BIAS", "ResBlock3d", "ReshapeTo3DLayer", "WSConv3d", "ReshapeTo2DLayer")})
    namespace.update(torch=torch, nn=nn, F=F, math=math)
    path = base / "LIA_3d/face_generator.py"
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FaceGenerator"](size=512, reshape_channel=32, group_norm_channel=32,
                                       latent_dim=512, outputsize=512, flag_estimate_occlusion_map=True).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--baseline-commit", default="c6220ee5c618af2af95117d70d625e818c27dcf7")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENVA_MEMORY_PROFILE_DIR"] = str(args.output_dir)
    torch.manual_seed(20261028)
    model = make_renderer(args.baseline_commit if args.baseline else None).cuda()
    recorder = MemoryRecorder("renderer-component", torch, "cuda")
    loaded = recorder.record("loaded", {"renderer": model})
    layers = []
    with torch.inference_mode():
        latent = torch.randn(1, 512, device="cuda") * 0.1
        feature = torch.randn(1, 32, 16, 64, 64, device="cuda") * 0.1
        for _ in range(3):
            model(latent, feature)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        def capture(name):
            def hook(*_):
                torch.cuda.synchronize()
                layers.append({"layer": name,
                               "allocated_bytes": torch.cuda.memory_allocated(),
                               "cumulative_peak_bytes": torch.cuda.max_memory_allocated()})
            return hook
        hooks = [module.register_forward_hook(capture(name))
                 for name, module in model.named_children()]
        output = model(latent, feature)
        for hook in hooks:
            hook.remove()
        assert torch.isfinite(output).all()
        import numpy as np
        np.save(args.output_dir / "frame_float.npy", output.cpu().numpy())
        frame = (((output + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu())
        del output
        steady = recorder.record("steady", {"renderer": model}, dict(latent=latent, feature=feature))
        assert torch.isfinite(frame).all()
    report = {"scope": "Full production LIA FaceGenerator, random FP32 weights and synthetic features; no motion/TTS/checkpoint",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "loaded": loaded, "steady": steady, "layer_snapshots": layers}
    (args.output_dir / "renderer_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"loaded_mib": loaded["allocated_bytes"] / 2**20,
                      "steady_allocated_mib": steady["allocated_bytes"] / 2**20,
                      "steady_reserved_mib": steady["reserved_bytes"] / 2**20,
                      "frame_peak_mib": steady["peak_allocated_bytes"] / 2**20}, indent=2))


if __name__ == "__main__":
    main()
