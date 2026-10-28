"""Exact CUDA/CPU lookup comparison for an explicitly supplied BF16 word table."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bits_equal(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("shape/dtype changed")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("non-finite output")
    if not torch.equal(reference.cpu().contiguous().view(torch.uint8), candidate.cpu().contiguous().view(torch.uint8)):
        raise ValueError("output bits changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new")
    args.output_dir.mkdir(parents=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    payload = args.table.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == metadata["tensor_sha256"]
    table = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(metadata["shape"])
    holder = torch.nn.Module()
    holder.base_lm = torch.nn.Module()
    embedding = torch.nn.Embedding.from_pretrained(table.cuda(), freeze=True).eval()
    holder.base_lm.embed_tokens = embedding
    holder.eval()
    before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    if args.offload:
        from voice_service.text_embedding_offload import offload_text_embedding
        prepared = offload_text_embedding(holder)
        assert prepared["enabled"]
    else:
        prepared = None
    setup_ms = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    hashes, rows = [], 0
    with torch.inference_mode():
        for offset in range(0, table.shape[0], 8192):
            ids = torch.arange(offset, min(offset + 8192, table.shape[0]), device="cuda")
            actual = embedding(ids).cpu()
            expected = table[offset:offset + len(ids)]
            bits_equal(expected, actual)
            hashes.append(hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest())
            rows += len(ids)
        del actual, expected, ids
        timings = {}
        for length in (32, 128, 512, 2048, 8192):
            ids = torch.arange(length, device="cuda").reshape(1, -1)
            for _ in range(10):
                embedding(ids)
            windows = []
            for _ in range(3):
                torch.cuda.synchronize()
                began = time.perf_counter()
                for _ in range(25):
                    embedding(ids)
                torch.cuda.synchronize()
                windows.append((time.perf_counter() - began) * 1000 / 25)
            timings[length] = {"window_ms": windows, "median_ms": statistics.median(windows)}
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    bits_equal(embedding.weight.cpu(), table)
    report = {"scope": "Trained VoxCPM2 word-table lookup only; no full LM/voice generation", "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "threads": torch.get_num_threads(),
              "offload": prepared, "setup_ms": setup_ms, "before_allocated_bytes": before,
              "after_allocated_bytes": after, "freed_bytes": before - after,
              "all_rows_checked": rows, "all_rows_bitwise_equal": True, "rng_preserved": True,
              "output_chunk_hashes": hashes, "timing": timings, "source_tensor": metadata,
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "freed_mib": (before-after)/2**20,
                      "latency_ms": {k: v["median_ms"] for k,v in timings.items()}}))


if __name__ == "__main__":
    main()
