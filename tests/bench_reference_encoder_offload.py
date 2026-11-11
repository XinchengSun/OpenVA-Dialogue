"""Measure the real reference encoders' storage; never claim full-pipeline fit."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.reference_encoder_fixture import ROOT, HEAD, load_encoders, load_offload, tensor_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu for tensor inventory only")
    torch.manual_seed(20261028)
    encoders = load_encoders()
    inventory = {name: tensor_inventory(module) for name, module in encoders.items()}
    report = {"scope": "Real production reference encoders, random FP32 weights; no trained checkpoint or full dialogue pipeline",
              "python": platform.python_version(), "torch": torch.__version__,
              "platform": platform.platform(), "device": args.device,
              "seed": 20261028, "image_shape": [1, 3, 512, 512],
              "inventory": inventory, "full_pipeline_vram_after_bytes": None,
              "full_pipeline_peak_reduction_bytes": None, "fits_12gib": None}
    if args.device == "cuda":
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        for module in encoders.values():
            module.to(device)
        with torch.no_grad():
            image = torch.randn(1, 3, 512, 512, device=device)
            anchor = encoders["motion_encoder"](image)[0].detach()
            encoders["face_encoder"](image)  # Warm the real convolution path.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        # The original worker left autograd enabled for this one-off pass.
        with torch.enable_grad():
            baseline_face = encoders["face_encoder"](image).detach()
        torch.cuda.synchronize(device)
        original_peak = torch.cuda.max_memory_allocated(device)
        expected_face, expected_anchor = baseline_face.cpu(), anchor.cpu()
        del baseline_face
        torch.cuda.empty_cache()
        offload_motion = load_offload()(torch, encoders, ("motion_encoder",))
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            face = encoders["face_encoder"](image).detach()
            torch.cuda.synchronize(device)
            candidate_peak = torch.cuda.max_memory_allocated(device)
            report["face_extraction_peak_allocated_bytes"] = {
                "original_grad_enabled": original_peak,
                "candidate_no_grad_and_motion_offloaded": candidate_peak,
                "reduction": original_peak - candidate_peak,
                "scope": "Encoder-only process with same image, anchor and weights; excludes all other models",
            }
            # Production drops this temporary before calling the offloader.
            image_bytes = image.numel() * image.element_size()
            del image
            offload_face = load_offload()(torch, encoders, ("face_encoder",))
            report["offload_by_stage"] = {"motion_encoder": offload_motion, "face_encoder": offload_face}
            report["encoder_allocated_freed_bytes"] = sum(
                stage["allocated_freed_bytes"] for stage in (offload_motion, offload_face))
            report["cuda_gpu"] = torch.cuda.get_device_name(device)
            report["cuda_runtime"] = torch.version.cuda
            report["source_image_cuda_bytes_released_separately"] = image_bytes
            report["cached_face_bytes"] = face.numel() * face.element_size()
            report["cached_anchor_bytes"] = anchor.numel() * anchor.element_size()
            report["cached_outputs_stay_on_gpu"] = face.is_cuda and anchor.is_cuda
            report["cached_face_max_abs_error"] = (face.cpu() - expected_face).abs().max().item()
            report["cached_anchor_max_abs_error"] = (anchor.cpu() - expected_anchor).abs().max().item()
            torch.testing.assert_close(face.cpu(), expected_face, rtol=0, atol=0)
            torch.testing.assert_close(anchor.cpu(), expected_anchor, rtol=0, atol=0)
    files = [ROOT / "dual_gpu_mic_realtime_ui_v8_playbuffer.py",
             HEAD / "LIA_3d/face_encoder.py", HEAD / "LIA_3d/motion_encoder.py",
             HEAD / "LIA_3d/util.py", HEAD / "LIA/modules.py",
             ROOT / "tools/visualization_0416/configs/head_animator_best_0506.yaml"]
    report["source_sha256"] = {str(path.relative_to(ROOT)).replace("\\", "/"):
                                hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    report["checkout_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["working_tree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    report["benchmark_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
