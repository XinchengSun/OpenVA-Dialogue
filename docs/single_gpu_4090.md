# Single RTX 4090 deployment

This profile puts DyStream motion, LIA rendering, and the official VoxCPM2
voice-cloning backend on **one physical NVIDIA GPU**. ASR runs on the CPU;
the conversation LLM remains the configured remote API. This is not a fully
offline local-LLM configuration. The existing multi-GPU Fish profile is unchanged.

The [4090 measurement record](single_gpu_validation_20260921.md) reports about
13.0 GiB sampled memory, 12.4 delivered FPS at 512×512, actual dialogue,
15-second continuous speech, and interruption recovery. These are server-local
measurements, not browser/public-network latency or a 25-FPS output claim.

## Configure and launch

Use the existing Linux/Python 3.11 realtime environment, the three DyStream
assets described in [weights.md](weights.md), and an official VoxCPM2 runtime.
The VoxCPM environment must import `voxcpm` or be given the official source's
`src` directory. A nano-only installation is insufficient for this profile.
Keep model weights and private credentials outside Git.

```bash
export FLASHAV2AV_DATA_ROOT=/data/openva-single-gpu
export PIPECAT_VENV=/path/to/pipecat
export PIPECAT_PYTHON="$PIPECAT_VENV/bin/python"

"$PIPECAT_PYTHON" scripts/configure_custom_runtime.py \
  --runtime-root "$FLASHAV2AV_DATA_ROOT" \
  --source-env /path/to/private-base.env \
  --single-gpu 0 \
  --prompt-wav /path/to/reference.wav \
  --prompt-text-file /path/to/reference.txt \
  --pipecat-python "$PIPECAT_PYTHON" \
  --voxcpm-python /path/to/official-voxcpm/bin/python \
  --voxcpm-official-source /path/to/VoxCPM/src

# The official VoxCPM2 weights must exist at:
# $FLASHAV2AV_DATA_ROOT/models/VoxCPM2
export ENV_FILE="$FLASHAV2AV_DATA_ROOT/config/custom_cascade.env"
bash scripts/flashav2av setup-single-gpu --check-only
bash scripts/run_demo.sh start
```

`--single-gpu` takes a **physical** GPU index. On a one-card PC it is normally
`0`; selecting `6` on a server still maps all three workers to logical `cuda:0`.
The launch checks reject mismatching TTS GPUs, a two-card Fish backend, a nano
memory pool, external/unmanaged TTS, or a non-CPU ASR configuration. They do not
claim that another unrelated process cannot also consume the selected GPU.

The generated environment enables `DYSTREAM_FOLD_EMA=1` and
`DYSTREAM_PRUNE_CFG=1`. Set either to `0` to independently disable that
optimization. Do not change CFG coefficients or Euler steps to reproduce a
speed comparison. `PIPE_FRAME_STRIDE=2`, the existing realtime default, produces
12.5 newly rendered frames/s from a 25-Hz motion timeline; use stride 1 only
after measuring sustained full-rate performance on the target machine.

`setup-single-gpu --check-only` checks a prepared environment without loading
models onto the GPU or downloading Fish weights. A fresh-host CUDA/PyTorch
and official VoxCPM installer is not provided by this change.

## What changed

1. Apply checkpoint EMA parameters to the motion model **on CPU once**, before
   moving it to CUDA. The inference worker no longer needs GPU copies of both
   EMA shadows and the original parameters for a session-long swap context.
2. Remove only CFG branches whose **final algebraic coefficient** is zero:

   ```text
   output = (1 - ws - wl - wr - wa) * unconditional
            + ws * self + wl * other + wr * anchor + wa * all
   ```

   With the repository's coefficients, anchor-only has zero coefficient,
   while unconditional has coefficient -1. The captured batch shrinks from
   five branches to four; the other/listener branch remains active. A changed
   weight or batch layout recreates the CUDA graphs. Floating-point rounding
   may differ when the GEMM batch shape changes, so a real-model comparison is
   required in addition to the algebra tests.
3. Place motion, renderer and official streaming TTS on the same visible card.
   Use the existing official backend's serialized generation and cached voice
   prompt instead of reserving a nano inference pool for concurrent users.
4. Bound cumulative Wav2Vec input during long replies, not just idle periods.
   The single-card profile sets `DYSTREAM_AUDIO_HISTORY_MAX_SEC=8.0` and retains
   4 seconds when compacting. Recurrent motion state is preserved. This bounds
   audio-attention work but changes distant acoustic context: unlike the EMA
   fold and zero-weight CFG pruning, it is **not numerically equivalent** to an
   unlimited-history run. Set the maximum to `0` to disable this bound.
5. Run reference-portrait landmark detection on CPU with EGL initialization
   disabled for that preprocessing step. MediaPipe otherwise opens a graphics
   context on physical GPU 0 even when CUDA is restricted to another card.
   The audit includes graphics contexts and resolves CUDA worker thread IDs,
   rather than checking only CUDA compute processes.

This profile keeps recurrent motion history, speaker/listener audio routing,
render resolution, interruption handling, and the existing denoising count.

## Research basis and limits

[DyStream, Appendix C](https://arxiv.org/html/2512.24408v2#S0.SS3) already reports
single-4090 core generation. Its reported memory and FPS exclude our complete
dialogue stack and must not be relabeled as this project's end-to-end results.

[FlashAttention](https://arxiv.org/abs/2205.14135) motivates reducing attention
memory traffic, but this implementation already uses PyTorch SDPA.
[PyGraph](https://arxiv.org/html/2503.19779v1) motivates measuring graph capture
per component instead of assuming every captured module is faster.
[StreamDiffusion](https://arxiv.org/html/2312.12491v2) motivates reuse of static
conditions; its approximate R-CFG and across-frame batching are not applied to
DyStream's autoregressive motion loop. No retraining or distillation is claimed.

## Reproduce validation

```bash
# CPU algebra and real EMA-loader tests
python -m unittest discover -s tests -p test_cfg_pruning.py
python -m unittest discover -s tests -p test_inference_ema.py
python -m unittest discover -s tests -p test_motion_audio_history.py
python -m unittest discover -s tests -p test_single_gpu_face_detector.py
python -m unittest discover -s tests -p test_benchmark_single_gpu.py
bash tests/test_single_gpu_launchers.sh
bash tests/test_single_gpu_bridge_reuse.sh
bash tests/test_single_gpu_setup.sh

# Numerical A/B on actual speaker and listener motion. Captures are warmed
# before reseeding, because the existing graph warmup consumes RNG values.
CUDA_VISIBLE_DEVICES=0 python scripts/verify_single_gpu_equivalence.py \
  --audio /path/to/real-speech.wav --ref-image /path/to/portrait.png --frames 50 \
  --output /data/results/cfg-equivalence.json

# Run on the same host/PID namespace as the service. Include the separate
# TTS process so the GPU-ownership check covers the complete local stack.
python scripts/benchmark_single_gpu.py \
  --base ws://127.0.0.1:7860 --gpu 0 \
  --server-pid "$(cat logs/pipecat_mse.pid)" \
  --server-pid "$(cat logs/voxcpm2_bridge.pid)" \
  --duration-sec 60 --max-memory-mib 24564 \
  --output-dir /data/results/single-gpu-run01
```

Add `--input-wav /path/to/mono-16k-pcm16.wav --turns 2` for actual
ASR/LLM/TTS dialogue. Each output directory must be new. The report separates
fMP4 arrival intervals, decoded playback FPS, exact repeated frames, GPU
memory, and response boundaries; decoded FPS is not model generation FPS.
GPU numerical tolerance, memory, sustained media delivery and dialogue must
all be checked before calling a hardware configuration validated.
