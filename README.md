<div align="center">

<img src="docs/assets/flashav2av-banner.svg" alt="FlashAV2AV" width="100%">

# FlashAV2AV

**Low-latency, full-duplex audio-video conversational avatars.**

[![CI](https://github.com/XinchengSun/FlashAV2AV/actions/workflows/ci.yml/badge.svg)](https://github.com/XinchengSun/FlashAV2AV/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Linux](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=111)](#requirements)
[![Version](https://img.shields.io/badge/version-0.1.0-6C63FF)](VERSION)

[中文](README_zh-CN.md) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Benchmarks](#benchmarks) · [Weights](docs/weights.md) · [Customization](README_CUSTOMIZATION.md)

</div>

FlashAV2AV turns a browser microphone stream into a continuously rendered talking avatar. It combines streaming speech understanding, response generation, speech synthesis, DyStream motion, LIA rendering, H.264/AAC fragmented MP4, and browser MediaSource playback on one interruptible timeline.

> [!IMPORTANT]
> This is a research release, not a hosted service. Model weights and identity media are not stored in Git. The repository currently has no project-level open-source license; public visibility does not grant reuse rights. Review [Third-party notices](THIRD_PARTY_NOTICES.md) before use.

## Highlights

- **Full-duplex interaction** — microphone input remains live while the avatar is speaking, with barge-in cancellation and graceful media handoff.
- **Continuous Listener and Speaker motion** — both modes share the same recurrent DyStream state instead of restarting the avatar at every turn.
- **Two dialogue routes** — native speech-to-speech for a compact path, or a configurable Paraformer → LLM → cloned-TTS cascade.
- **Real-time information** — the cascade can route fresh-information queries to provider-backed web search without enabling LLM thinking mode.
- **Local customization** — portrait and reference-voice assets stay outside Git and can be selected through a local customization workflow.
- **One lifecycle CLI** — setup, configure, start, status, restart, and stop use ownership-aware scripts and media readiness checks.

## Architecture

```mermaid
flowchart LR
    MIC["Browser microphone"] --> WS["WebSocket PCM"]
    WS --> ROUTE{"Dialogue route"}

    ROUTE -->|native_s2s| S2S["Qwen Audio realtime S2S"]
    ROUTE -->|custom_cascade| VAD["Silero VAD"]
    VAD --> ASR["Paraformer streaming ASR"]
    ASR --> LLM["Streaming OpenAI-compatible LLM"]
    LLM -. fresh queries .-> SEARCH["Provider web search"]
    LLM --> TTS["VoxCPM2 cloned TTS"]

    S2S --> PCM["Assistant PCM"]
    TTS --> PCM
    PCM --> MOTION["DyStream motion · GPU 0"]
    WS --> LISTENER["Listener conditioning"]
    LISTENER --> MOTION
    MOTION --> RENDER["LIA renderer · GPU 1"]
    RENDER --> MUX["H.264 + AAC fMP4"]
    MUX --> MSE["Browser MediaSource"]
```

Pipecat owns dialogue orchestration, turn events, and cancellation. `server_mse.py` owns the continuous AV2AV state, motion/render workers, A/V boundaries, fragmented MP4 transport, and browser WebSocket session. See [architecture.md](docs/architecture.md) for the state contract.

## Deployment routes

| Route | Speech pipeline | GPU topology | Best for |
| --- | --- | --- | --- |
| `native_s2s` | Qwen Audio realtime speech-to-speech | 2 non-overlapping DyStream GPUs | Simpler runtime path and fallback deployment |
| `custom_cascade` | Silero → Paraformer → streaming LLM → VoxCPM2 | 2 DyStream GPUs + 1 separate VoxCPM2 GPU | Voice cloning, model selection, and real-time search |
| Fish S2 Pro candidate | Standalone SGLang-compatible TTS harness | 1–2 Fish GPUs | TTS research and quality/latency A/B; not wired into the default lifecycle |

Fish S2 Pro is not the default TTS. Its upstream license is research/non-commercial unless separately licensed.

## Requirements

The checked runtime contract is intentionally strict:

| Component | Checked configuration |
| --- | --- |
| Host | Linux, NVIDIA GPU, `ffmpeg`, `ffprobe`, `curl`, `flock`, `nvidia-smi` |
| Python | 3.11 |
| PyTorch | `2.8.0+cu128` |
| CUDA runtime | 12.8 |
| Avatar output | 512 × 512, H.264 video + AAC audio |
| DyStream | two distinct logical CUDA devices |
| Custom cascade | a third physical GPU for VoxCPM2; it must not overlap DyStream |

The release installer has been validated for structure and dependency checks, but a blank-host multi-GPU installation is not exercised by GitHub CI.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/XinchengSun/FlashAV2AV.git
cd FlashAV2AV

# Keep multi-gigabyte models, environments, and caches outside Git.
export FLASHAV2AV_DATA_ROOT=/data/flashav2av
bash scripts/flashav2av setup
```

`setup` prepares both supported dialogue routes: it creates the project-specific Pipecat and VoxCPM2 environments, prefetches Paraformer, downloads the required upstream DyStream/LIA/Wav2Vec2 assets recorded in `weights-manifest.json`, and creates ignored runtime symlinks. It therefore performs the cascade setup even when only `native_s2s` will be used.

The current installer is **not a universal CUDA bootstrap**: it expects the checked base runtime (`torch 2.8.0+cu128`, CUDA 12.8, MediaPipe, and the DyStream dependencies) to already be available to its `--system-site-packages` environment. Use it on the tested server image, or reproduce that base environment first. A container/lockfile for a blank host is still pending.

### 2A. Native speech-to-speech

Copy `.env.example` to a private `.env`, then set at least:

```dotenv
PIPECAT_MSE_DIALOG_MODE=native_s2s
PIPECAT_S2S_API_KEY=your_private_key
DYSTREAM_REF_IMAGE=/absolute/path/to/authorized-avatar.png

CUDA_VISIBLE_DEVICES=0,1
MOTION_GPU=0
RENDER_GPU=1
```

### 2B. Custom cascade

Prepare a private source env containing a DashScope/OpenAI-compatible key and an authorized reference voice, then generate isolated runtime envs:

```bash
bash scripts/flashav2av configure \
  --source-env /absolute/path/to/private-base.env \
  --prompt-wav /absolute/path/to/authorized-reference.wav \
  --dystream-gpus 0,1 \
  --tts-gpu 2 \
  --realtime-search-mode smart \
  --realtime-search-strategy turbo
```

The generated env files are written with mode `0600` under `$FLASHAV2AV_DATA_ROOT/config/`. LLM thinking is explicitly disabled in this low-latency path.

### 3. Start

```bash
bash scripts/flashav2av start
```

`DEMO_READY` means the workers, dialogue route, bridge, and decodable media smoke test passed. The service binds to loopback by default. From a remote machine:

```bash
ssh -N -L 6008:127.0.0.1:7860 <user>@<server>
```

Open <http://127.0.0.1:6008/>, keep one demo tab, click **Start conversation**, and allow microphone access.

Useful lifecycle commands:

```bash
bash scripts/flashav2av status
bash scripts/flashav2av restart
bash scripts/flashav2av stop
bash scripts/flashav2av setup --check-only
```

## Benchmarks

### Fish S2 Pro TTS-only candidate

Warm single-request measurements on an 8 × RTX 4090 host; the dual-GPU profiles used physical GPUs 5 and 6. Each profile contains 60 measured samples after 3 discarded warm-up samples. These values cover only the TTS bridge—not ASR, LLM, DyStream, encoding, network, or browser playback.

| Profile | First PCM P50 / P95 | Audible TTFA P50 / P95 | RTF P50 / P95 | Starved samples |
| --- | ---: | ---: | ---: | ---: |
| Single GPU, stride 20/10 | 1259 / 1298 ms | 1262 / 1392 ms | 0.592 / 0.612 | 0 / 60 |
| Dual GPU, stride 20/10 | 666 / 692 ms | 672 / 763 ms | 0.561 / 0.584 | 0 / 60 |
| Dual GPU, stride 10/10 | **426 / 436 ms** | **431 / 579 ms** | 0.564 / 0.584 | 0 / 60 |

Raw benchmark record: [`docs/fishspeech_2gpu_benchmark_20260811.json`](docs/fishspeech_2gpu_benchmark_20260811.json).

No reproducible microphone-to-visible-avatar E2E benchmark is published yet. Provider latency, endpointing, browser buffering, and GPU contention must be reported separately; TTS-only figures must not be presented as conversational latency.

## Model weights

Weights are downloaded directly from their official upstream repositories and are never committed to Git:

```bash
python scripts/setup_weights.py download
python scripts/setup_weights.py verify --deep

# Optional research candidate
python scripts/setup_weights.py --model fish-audio-s2-pro download
```

`verify --deep` checks recorded hashes where available. The current public manifest publishes no SHA-256 values, so it verifies file presence and recorded sizes; it does not claim cryptographic integrity. See [weights.md](docs/weights.md) for the inventory, revisions, paths, and license status.

## Avatar and voice customization

After startup, open the local-only customization page:

```text
http://127.0.0.1:6008/customize
```

Portraits, voice recordings, transcripts, generated latents, caches, and rollback snapshots remain private runtime assets. They are ignored by Git. See [README_CUSTOMIZATION.md](README_CUSTOMIZATION.md).

## Runtime verification

A demonstration should pass all of the following:

- motion and render workers are alive;
- the selected dialogue route reports ready;
- media probes decode H.264 and AAC;
- Listener, Speaker, and interruption stay on one continuous timeline;
- at least three consecutive turns complete;
- a barge-in returns to Listener and a new turn can start;
- long-running browser playback does not exhaust its media buffer.

```bash
bash scripts/flashav2av status
tail -f logs/pipecat_mse.log
```

## Repository layout

```text
pipecat_dystream/   dialogue, ASR, LLM, search, TTS, and MSE adapters
voice_service/      isolated VoxCPM2 and compatible PCM bridges
model/              DyStream motion model code
tools/              LIA renderer and preprocessing code
static/             browser demo and customization frontend
scripts/            setup, lifecycle, health checks, probes, and benchmarks
tests/              logic, protocol, cancellation, and media regressions
```

The supported release entry point is `scripts/flashav2av`. Legacy offline scripts require local sample media that is intentionally not distributed.

## Known limitations

- The renderer produces native 512 × 512 frames; enlarging the player cannot add model detail.
- Expression, emotion, nod, and blink controls are not exposed as stable production controls.
- Listener naturalness remains dependent on the source checkpoint and conditioning audio.
- Required model checkpoints are external and DyStream currently publishes no model card/license file.
- GitHub CI validates the release surface, shell syntax, Python syntax, manifest, and credential/weight exclusions; it does not run GPU inference.

See [known_issues.md](docs/known_issues.md) for the current list.

## Data, security, and licensing

Never commit API keys, `.env` files, portraits, reference voices, videos, model weights, virtual environments, logs, captures, or temporary public URLs. Put any public deployment behind authenticated HTTPS/WSS.

This repository currently does **not** declare a project-level open-source license. Third-party model and code terms still apply. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before downloading or redistributing any dependency.

## Acknowledgements

FlashAV2AV integrates work from [DyStream](https://github.com/XinchengSun/DyStream), [Pipecat](https://github.com/pipecat-ai/pipecat), [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), [FunASR/Paraformer](https://github.com/modelscope/FunASR), [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h), and the optional [Fish Speech S2 Pro](https://huggingface.co/fishaudio/s2-pro) candidate. Please cite and follow each upstream project's terms when using their work.
