"""Profile the production-sized motion core separately from the full pipeline."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_memory import MemoryRecorder, prepare_motion_inference_memory, place_motion_for_inference
from tests.motion_memory_fixture import make_motion_core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--place", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENVA_MEMORY_PROFILE_DIR"] = str(args.output_dir)
    if not torch.cuda.is_available():
        parser.error("requires CUDA")
    torch.manual_seed(20261028)
    model, classes = make_motion_core()
    placement = place_motion_for_inference(torch, model, "cuda") if args.place else None
    if not args.place:
        model.cuda()
    recorder = MemoryRecorder("motion-core", torch, "cuda")
    baseline = recorder.record("loaded", dict(model.named_children()))
    preparation = None
    if args.prepare:
        preparation = prepare_motion_inference_memory(torch, model)
        torch.cuda.empty_cache()
        recorder.record("prepared", dict(model.named_children()))
    with torch.inference_mode():
        runner = classes["CUDAGraphGPTRunner"](
            model.blocks, model.output_norm, model.output_proj,
            hidden_size=768, batch_size=4, device="cuda")
        x = torch.randn(4, 94, 768, device="cuda")
        audio = torch.randn_like(x)
        anchor = torch.randn(4, 1, 768, device="cuda")
        mask = torch.triu(torch.full((94, 94), float("-inf"), device="cuda"), diagonal=1)
        prediction = runner.forward(x, audio, anchor, mask, mask)
        captured = recorder.record("captured", dict(model.named_children()),
                                   dict(graph_buffers=runner.static_buffers, x=x, audio=audio,
                                        anchor=anchor, mask=mask, prediction=prediction))
        if args.prepare:
            torch.cuda.empty_cache()
        outputs = [prediction.cpu()]
        for index in range(12):
            x.add_(0.001)
            audio.mul_(0.999)
            prediction = runner.forward(x, audio, anchor, mask, mask)
            outputs.append(prediction.cpu())
        steady = recorder.record("steady", dict(model.named_children()),
                                 dict(graph_buffers=runner.static_buffers, x=x, audio=audio,
                                      anchor=anchor, mask=mask, prediction=prediction))
        import numpy as np
        np.save(args.output_dir / "predictions.npy", torch.stack(outputs).numpy())
    report = {"scope": "Production motion core, random FP32 weights, no Wav2Vec/TTS/renderer",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "shape": {"layers": 12, "hidden": 768, "cfg_batch": 4, "sequence": 94},
              "loaded": baseline, "placement": placement, "preparation": preparation,
              "captured": captured, "steady": steady}
    (args.output_dir / "baseline_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"loaded_mib": baseline["allocated_bytes"] / 2**20,
                      "captured_mib": captured["allocated_bytes"] / 2**20,
                      "steady_allocated_mib": steady["allocated_bytes"] / 2**20,
                      "steady_reserved_mib": steady["reserved_bytes"] / 2**20}, indent=2))


if __name__ == "__main__":
    main()
