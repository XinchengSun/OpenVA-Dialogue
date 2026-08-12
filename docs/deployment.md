# FlashAV2AV deployment guide

This guide contains the host-specific setup details that do not belong in the
project landing page. The supported public entry point is always
`scripts/flashav2av`; the older launchers are implementation details.

## Tested host contract

The current Fish S2 Pro deployment was validated on Linux with:

| Component | Tested configuration |
| --- | --- |
| Python | 3.11 |
| PyTorch runtime | `2.8.0+cu128` |
| CUDA runtime | 12.8 |
| Media tools | `ffmpeg`, `ffprobe` |
| DyStream | two distinct NVIDIA GPUs |
| Fish S2 Pro | two additional GPUs, disjoint from DyStream |
| Browser output | 512 x 512 H.264 video + AAC audio over fMP4/MSE |

GitHub CI validates the release surface, configuration logic, lifecycle
ownership, and CPU-only tests. It does not run multi-GPU inference.

`requirements-pipecat.txt` and `scripts/check_pipecat_env.py` define the tested
realtime runtime. The root `requirements.txt` is a legacy/offline research
environment and is not the supported server installer.

## Prepare the Fish runtime

`scripts/flashav2av setup` installs the Pipecat environment, prefetches
Paraformer, downloads the pinned Fish/DyStream/LIA/Wav2Vec2 model assets, and
creates ignored runtime symlinks. It does **not** build SGLang-Omni or CUDA from
a blank host.

Before starting Fish, the following checked runtime must exist under
`$FLASHAV2AV_DATA_ROOT/runtime/fish-s2-pro` (or the equivalent paths selected
through `FISH_ROOT`, `SGLANG_OMNI_SRC`, `FISH_VENV`, and `FISH_MODEL`):

```text
runtime/fish-s2-pro/
  src/sglang-omni-ghproxy/       SGLang-Omni commit 2e607bc005c1...
  venv/bin/sgl-omni              working SGLang-Omni executable
  models/fishaudio-s2-pro/       pinned Fish S2 Pro snapshot
  references/                    local reference audio allow-list
```

The Fish launcher verifies the exact SGLang-Omni commit and applies
[`sglang_omni_fish_cross_gpu_host_staging.patch`](../patches/sglang_omni_fish_cross_gpu_host_staging.patch)
once. The patch routes cross-GPU stream tensors through host shared memory on
machines whose GPUs do not expose peer access.

## Primary Fish S2 Pro deployment

Create a private base env containing:

```dotenv
PIPECAT_LLM_API_KEY=your_private_key
PIPECAT_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DYSTREAM_REF_IMAGE=/absolute/path/to/avatar.png
```

The LLM key may instead be supplied through one of the compatible key names
accepted by `scripts/configure_custom_runtime.py`. Keep the reference image,
reference audio, and matching transcript outside Git.

```bash
export FLASHAV2AV_DATA_ROOT=/data/flashav2av

bash scripts/flashav2av setup

bash scripts/flashav2av configure \
  --source-env /absolute/path/to/private-base.env \
  --prompt-wav /absolute/path/to/reference.wav \
  --prompt-text-file /absolute/path/to/reference.txt \
  --dystream-gpus 0,1 \
  --fish-gpus 2,3 \
  --tts-backend fish_s2pro \
  --realtime-search-mode smart \
  --realtime-search-strategy turbo

bash scripts/flashav2av start
```

The generated private files are written with mode `0600` under
`$FLASHAV2AV_DATA_ROOT/config/`. The managed endpoints are:

| Endpoint | Purpose |
| --- | --- |
| `127.0.0.1:8001` | Fish S2 Pro HTTP service |
| `127.0.0.1:8771` | local streaming PCM bridge |
| `127.0.0.1:7860` | FlashAV2AV browser service |
| local `127.0.0.1:6008` | suggested SSH-forwarded browser port |

Remote browser access:

```bash
ssh -N -L 6008:127.0.0.1:7860 <user>@<server>
```

Open <http://127.0.0.1:6008/>, click **Start conversation**, and allow
microphone access.

## Alternative dialogue and TTS routes

### Native speech-to-speech

The native Qwen Audio route keeps the two-GPU DyStream renderer but replaces
the Paraformer/LLM/Fish cascade. It is retained for compatibility and A/B work:

```dotenv
PIPECAT_MSE_DIALOG_MODE=native_s2s
PIPECAT_S2S_API_KEY=your_private_key
DYSTREAM_REF_IMAGE=/absolute/path/to/avatar.png
CUDA_VISIBLE_DEVICES=0,1
MOTION_GPU=0
RENDER_GPU=1
```

### VoxCPM2 fallback

```bash
bash scripts/flashav2av setup-voxcpm2

bash scripts/flashav2av configure \
  --source-env /absolute/path/to/private-base.env \
  --prompt-wav /absolute/path/to/reference.wav \
  --prompt-text-file /absolute/path/to/reference.txt \
  --dystream-gpus 0,1 \
  --tts-backend voxcpm2 \
  --tts-gpu 2
```

## Lifecycle and readiness

```bash
bash scripts/flashav2av status
bash scripts/flashav2av restart
bash scripts/flashav2av stop
bash scripts/flashav2av setup --check-only
```

`DEMO_READY` means that the selected TTS stack, motion/render workers, dialogue
route, and decodable H.264/AAC media smoke test all passed.

For an interactive acceptance pass, verify at least three turns, one barge-in,
the Speaker-to-Listener return, and sustained playback without browser buffer
exhaustion. See [known issues](known_issues.md) and
[avatar/voice customization](../README_CUSTOMIZATION.md) for the remaining
runtime boundaries.
