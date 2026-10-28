"""CUDA AudioVAE component check against an explicitly supplied official source."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference_memory import MemoryRecorder, enable_prompt_encoder_offload, cache_decoder_weight_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-decoder", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENVA_MEMORY_PROFILE_DIR"] = str(args.output_dir)
    source = args.source.read_bytes()
    spec = importlib.util.spec_from_file_location("openva_memory_audiovae", args.source)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    torch.manual_seed(20261028)
    vae = upstream.AudioVAE().eval().cuda()
    recorder = MemoryRecorder("audiovae", torch, "cuda")
    with torch.no_grad():
        audio = torch.randn(1, 1, 16000, device="cuda") * 0.01
        reference_audio = audio.cpu()
        features = vae.encode(audio, 16000)
        expected_features = features.cpu()
        del audio, features
        patches = [torch.randn(1, 64, 4, device="cuda") * 0.1 for _ in range(2)]
        def decode():
            with vae.streaming_decode() as stream:
                return torch.cat([stream.decode_chunk(patch).cpu() for patch in patches], dim=-1)
        # Warm kernels first, then measure the steady decoding phase separately
        # from reference encoding/startup workspaces.
        decode()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_audio = decode()
        before = recorder.record("before_offload", {"vae": vae})
        offload = enable_prompt_encoder_offload(torch, SimpleNamespace(audio_vae=vae))
        after = recorder.record("after_offload", {"vae": vae})
        decoder_cache = None
        if args.cache_decoder:
            decoder_cache = cache_decoder_weight_norm(torch, SimpleNamespace(audio_vae=vae))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            candidate_audio = decode()
        torch.testing.assert_close(candidate_audio, baseline_audio, rtol=0, atol=0)
        decode_steady = recorder.record("decode_steady", {"vae": vae})
        # A later reference encode must reload and offload automatically.
        audio = reference_audio.cuda()
        encoded = vae.encode(audio, 16000).cpu()
        again = vae.encode(audio, 16000).cpu()
        torch.testing.assert_close(encoded, expected_features, rtol=0, atol=0)
        torch.testing.assert_close(encoded, again, rtol=0, atol=0)
        torch.testing.assert_close(decode(), baseline_audio, rtol=0, atol=0)
        steady = recorder.record("steady", {"vae": vae})
    report = {"scope": "Official AudioVAE v2 architecture with random FP32 weights; no LM or trained speech checkpoint",
              "source_sha256": hashlib.sha256(source).hexdigest(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "decode_patches": 2, "latent_shape": [1, 64, 4],
              "offload": offload, "decoder_cache": decoder_cache,
              "before": before, "after": after, "decode_steady": decode_steady, "steady": steady,
              "streamed_audio_max_abs_error": (candidate_audio - baseline_audio).abs().max().item(),
              "automatic_reload_max_abs_error": (encoded - again).abs().max().item()}
    (args.output_dir / "audiovae_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"offload": offload, "before_mib": before["allocated_bytes"] / 2**20,
                      "after_mib": after["allocated_bytes"] / 2**20,
                      "before_reserved_mib": before["reserved_bytes"] / 2**20,
                      "after_reserved_mib": after["reserved_bytes"] / 2**20,
                      "steady_mib": steady["allocated_bytes"] / 2**20,
                      "steady_reserved_mib": steady["reserved_bytes"] / 2**20,
                      "audio_error": report["streamed_audio_max_abs_error"]}, indent=2))


if __name__ == "__main__":
    main()
