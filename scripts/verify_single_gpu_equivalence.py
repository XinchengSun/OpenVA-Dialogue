#!/usr/bin/env python3
"""Compare five-branch and pruned CFG on the actual model, after graph warmup.

This is a motion numerical check, not an end-to-end throughput benchmark.
Use CUDA_VISIBLE_DEVICES to select exactly one idle physical GPU.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--ref-image', type=Path, default=os.getenv('DYSTREAM_REF_IMAGE'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=50)
    args = parser.parse_args()
    if args.frames < 1 or torch.cuda.device_count() != 1:
        parser.error('positive frames and exactly one visible CUDA GPU are required')
    if not args.audio.is_file():
        parser.error(f'audio file is missing: {args.audio}')
    if args.ref_image is None or not args.ref_image.is_file():
        parser.error('provide an existing --ref-image or DYSTREAM_REF_IMAGE')
    os.environ['DYSTREAM_FOLD_EMA'] = '1'
    # Use the same CPU/EGL-free portrait preprocessing as the single-card app.
    os.environ['DYSTREAM_SINGLE_GPU'] = os.environ.get('CUDA_VISIBLE_DEVICES') or '0'
    import app
    import librosa
    from PIL import Image

    torch.set_num_threads(1)
    app.load_dystream_model()
    app.load_visualization_model()
    _, _, anchor_cpu = app.process_image(Image.open(args.ref_image).convert('RGB'))
    model = app._dystream_model
    anchor = anchor_cpu.to('cuda').reshape(1, 1, -1)
    context = model.inpainting_length
    window = model.cfg.cbh_window_length
    total_frames = context + args.frames + 2
    waveform, _ = librosa.load(args.audio, sr=16000, mono=True)
    waveform = np.resize(waveform, total_frames * 640).astype(np.float32)
    audio = torch.from_numpy(waveform).reshape(1, -1).cuda()
    results = {}
    comparisons = {}

    with torch.inference_mode():
        for mode in ('speaker', 'listener'):
            self_audio = audio if mode == 'speaker' else torch.zeros_like(audio)
            other_audio = audio if mode == 'listener' else torch.zeros_like(audio)
            self_features = model.get_audio2face_fea(self_audio, None, total_frames)
            other_features = model.get_audio2face_fea_other(other_audio, None, total_frames)
            for prune in (False, True):
                os.environ['DYSTREAM_PRUNE_CFG'] = '1' if prune else '0'
                model.setup_cuda_graphs(num_inference_steps=1)

                def step(index, history):
                    return model.one_clip_only_inference_cuda_graph(
                        self_features[:, index:index + window], self_audio, None,
                        anchor, history, 1,
                        per_compute_audio_other_feature=other_features[:, index:index + window],
                        audio_other=other_audio, past_audio_other=None,
                        noise_scheduler=app._noise_scheduler, num_inference_steps=1,
                    )

                history = anchor.repeat(1, context, 1)
                step(0, history)  # capture consumes random numbers; exclude it
                torch.cuda.synchronize()
                torch.manual_seed(20260921)
                torch.cuda.manual_seed_all(20260921)
                torch.cuda.reset_peak_memory_stats()
                outputs = []
                started = time.perf_counter()
                for index in range(args.frames):
                    motion = step(index, history)
                    outputs.append(motion.clone())
                    history = torch.cat([history, motion], dim=1)[:, -context:]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                name = f'{mode}_{"pruned" if prune else "baseline"}'
                results[name] = torch.cat(outputs, dim=1).cpu()
                comparisons[name] = {
                    'motion_only_fps': args.frames / elapsed,
                    'elapsed_sec': elapsed,
                    'allocated_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
                    'cfg_branches': list(model._cuda_graph_branch_indices),
                }
            baseline, pruned = results[f'{mode}_baseline'], results[f'{mode}_pruned']
            delta = (baseline - pruned).abs()
            comparisons[f'{mode}_comparison'] = {
                'max_absolute_error': delta.max().item(),
                'mean_absolute_error': delta.mean().item(),
                'baseline_motion_std': baseline.std().item(),
                'close_rtol_1e-4_atol_1e-4': torch.allclose(baseline, pruned, rtol=1e-4, atol=1e-4),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
    np.savez_compressed(args.output.with_suffix('.npz'), **{key: value.numpy() for key, value in results.items()})
    print(json.dumps(comparisons, indent=2))
    if not all(comparisons[f'{mode}_comparison']['close_rtol_1e-4_atol_1e-4'] for mode in ('speaker', 'listener')):
        raise SystemExit('CFG equivalence tolerance exceeded; do not enable pruning by default')


if __name__ == '__main__':
    main()
