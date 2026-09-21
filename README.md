<div align="center">

<img src="docs/assets/flashav2av-banner.svg" alt="OpenVA-Dialogue" width="100%">

# OpenVA-Dialogue

**An agent system for full-duplex real-time audio-visual dialogue.**

[![CI](https://github.com/XinchengSun/OpenVA-Dialogue/actions/workflows/ci.yml/badge.svg)](https://github.com/XinchengSun/OpenVA-Dialogue/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Linux](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=111)](#tested-setup)

[中文](README_zh-CN.md) · [Quick Start](#quick-start) · [Architecture](#how-it-works) · [Benchmarks](#measured-performance) · [Deployment](docs/deployment.md) · [Customization](README_CUSTOMIZATION.md)

</div>

OpenVA-Dialogue turns one portrait and one short reference recording into a
browser-based conversational avatar. Its primary pipeline streams microphone
audio through Paraformer, an OpenAI-compatible LLM, Fish Speech S2 Pro, and a
continuous DyStream/LIA renderer while preserving one interruptible audio/video
timeline.

**Keywords:** Full-Duplex Dialogue, Audio-Visual Dialogue, Real-Time Interaction,
Streaming Generation.

Previously named FlashAV2AV. Existing `scripts/flashav2av` commands,
`FLASHAV2AV_*` environment variables, and engine identifiers remain compatible.

> **License status:** source code is public, but a project-wide license has not
> been assigned. Public availability is not a blanket permission to reuse or
> redistribute all components. See [license status](LICENSE_STATUS.md).

Two deployment profiles are available: the existing multi-GPU Fish pipeline
and the opt-in single-4090 VoxCPM2 pipeline. Both use DyStream/LIA for the avatar.
The tested conversation input is microphone audio; this release does not claim
camera-based visual understanding.

## What it delivers

- **Portrait customization** from a single front-facing image.
- **Zero-shot voice cloning** with Fish Speech S2 Pro from a short reference
  recording and matching transcript.
- **Streaming conversation with barge-in**: microphone capture stays active
  while the avatar speaks, and a new turn can cancel the pending reply.
- **Continuous Listener and Speaker motion** on the same recurrent DyStream
  state instead of restarting the avatar at every turn.
- **Fresh-information queries** through optional provider-backed web search,
  with LLM thinking explicitly disabled on the tested low-latency path.
- **One managed lifecycle** for the Fish service, PCM bridge, AV2AV workers,
  media smoke test, restart, and cleanup.

## Preview

The public repository currently includes the software and a generated
DyStream preview, not a prerecorded end-to-end conversation. The preview shows
the native 512 x 512 avatar output; a reproducible microphone-to-avatar demo
capture is still being prepared.

<div align="center">
  <img src="docs/assets/flashav2av-avatar-preview.gif" alt="OpenVA-Dialogue generated avatar preview" width="384">
</div>

## Quick Start

Choose the profile before installing:

| Profile | Local GPU placement | Guide |
| --- | --- | --- |
| Single RTX 4090 / 24GB | DyStream + LIA + VoxCPM2 on one GPU; ASR on CPU; LLM via API | [Single-card setup](docs/single_gpu_4090.md) |
| Existing Fish / SGLang-Omni | two DyStream GPUs + two separate Fish GPUs | [Multi-card deployment](docs/deployment.md) |

The commands below are for the **multi-GPU Fish profile**, not the single-card
profile. Neither profile is a fully offline local-LLM setup.

The public installer targets the **tested Linux server image**. It downloads
the model assets and prepares the managed services, but it does not build CUDA
or SGLang-Omni from a blank host. Complete the short Fish runtime prerequisite
in the [deployment guide](docs/deployment.md#prepare-the-fish-runtime) first.

```bash
git clone https://github.com/XinchengSun/OpenVA-Dialogue.git
cd OpenVA-Dialogue

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

The private base env must provide `PIPECAT_LLM_API_KEY` (or a compatible
DashScope/OpenAI key) and `DYSTREAM_REF_IMAGE`. The two Fish GPUs must not
overlap the two DyStream GPUs. A successful start ends with `DEMO_READY`.

Forward the loopback browser service and open the local URL:

```bash
ssh -N -L 6008:127.0.0.1:7860 <user>@<server>
```

<http://127.0.0.1:6008/>

Full prerequisites, endpoint mapping, native speech-to-speech, and the VoxCPM2
fallback are documented in the [deployment guide](docs/deployment.md).

## How it works

```mermaid
flowchart LR
    MIC["Browser microphone"] --> VAD["Silero VAD"]
    VAD --> ASR["Paraformer streaming ASR"]
    ASR --> LLM["Streaming LLM"]
    LLM -. optional live search .-> SEARCH["Web search"]
    LLM --> TTS["Fish S2 Pro or VoxCPM2"]
    TTS --> MOTION["DyStream motion"]
    MIC -. voiced input .-> LISTENER["Listener conditioning"]
    REF["Authorized looping reference audio"] -. when microphone is silent .-> LISTENER
    LISTENER --> MOTION
    MOTION --> RENDER["LIA renderer"]
    RENDER --> MEDIA["H.264 + AAC fMP4"]
    MEDIA --> BROWSER["Browser MediaSource"]
```

Pipecat owns dialogue orchestration, turn events, and cancellation.
`server_mse.py` owns the continuous avatar state, motion/render workers,
audio/video boundaries, and fragmented-MP4 browser stream. The detailed state
contract and GPU placement are in [architecture.md](docs/architecture.md).

## Tested setup

An opt-in **single RTX 4090 / 24GB profile** is available with local VoxCPM2
TTS, CPU ASR, and an API-based LLM. See [single-card setup and validation](docs/single_gpu_4090.md).
Measured on a 4090: about 13.0 GiB sampled GPU memory and 12.4 delivered FPS
at 512×512; see [measurements and limits](docs/single_gpu_validation_20260921.md).
The Fish configuration below remains the existing multi-GPU profile.

| Component | Tested configuration |
| --- | --- |
| Host | Linux, Python 3.11, NVIDIA GPUs, `ffmpeg`/`ffprobe` |
| Realtime Python runtime | PyTorch `2.8.0+cu128`, CUDA 12.8 |
| Avatar | two distinct DyStream GPUs; native 512 x 512 output |
| Primary TTS | two additional Fish S2 Pro GPUs, disjoint from DyStream |
| Browser media | H.264 video + AAC audio over fMP4/MSE |

`requirements-pipecat.txt` defines the realtime service dependencies. The root
`requirements.txt` belongs to the legacy/offline research environment and is
not the supported server installer. GitHub CI does not run GPU inference.

## Measured performance

### Single-4090 system measurements

The [2026-09-21 measurement summary](docs/single_gpu_validation_20260921.md)
covers six completed dialogue turns across 90-, 180-, and 100-second runs on a
Linux server with an RTX 4090 and Xeon Gold 6530 CPU.

| Metric | Single GPU: DyStream + LIA + VoxCPM2 |
| --- | ---: |
| Sampled peak GPU memory | **13,320 MiB (13.008 GiB)** |
| Delivered video | **12.43–12.47 FPS, 512×512** |
| Voice end → server reply-media boundary, six turns | **1.68–2.26 s; mean 1.94 s** |

ASR runs on CPU and the LLM uses a remote API. Latency is measured at the
server-loopback client, excluding the public tunnel, browser buffering, and
sound-card playback. These small-sample measurements are not a latency SLA,
an hours-long stability result, or a perceptual naturalness/lip-sync benchmark.
The measurement tools and summary are public; raw reports and identity-bearing
test media are not included. See the linked record for reproduction and limits.

### Fish TTS bridge microbenchmark

This separate snapshot measures **warm, single-request TTS bridge latency**;
it is not microphone-to-visible-avatar latency. It was measured on
2026-08-11 at commit `29d1bc376fa2` on an 8 x RTX 4090 host. Fish used physical
GPUs 5/6, profile `low_ttfa_gapless` (`stream_stride=10`, follow-up stride 10),
3 warm-up samples were discarded, and 60 requests were measured.

| Metric | Dual-GPU Fish S2 Pro |
| --- | ---: |
| First PCM P50 / P95 | **426 / 436 ms** |
| Audible TTFA P50 / P95 | **431 / 579 ms** |
| RTF P50 / P95 | **0.564 / 0.584** |
| Playback starvation | **0 / 60** |

The full single/dual-GPU comparison is in
[`fishspeech_2gpu_benchmark_20260811.json`](docs/fishspeech_2gpu_benchmark_20260811.json).
ASR, endpointing, LLM, DyStream, encoding, network, and browser buffering are
excluded. No microphone-to-visible-avatar E2E number is published yet.

## Customize the avatar

After startup, open <http://127.0.0.1:6008/customize>. Upload a front-facing
portrait, a clean 10-20 second reference recording, and its matching transcript.
Activation updates the private runtime atomically and rolls back if validation
or restart fails. See [README_CUSTOMIZATION.md](README_CUSTOMIZATION.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Deployment](docs/deployment.md) | tested host, Fish runtime, ports, lifecycle, alternative backends |
| [Single GPU](docs/single_gpu_4090.md) | 4090 setup, optimization switches, validation commands |
| [Single-GPU measurements](docs/single_gpu_validation_20260921.md) | measured latency/memory/FPS, methodology, unverified scope |
| [Architecture](docs/architecture.md) | Listener/Speaker state, interruption, media boundaries, GPU placement |
| [Customization](README_CUSTOMIZATION.md) | portrait and zero-shot voice-cloning workflow |
| [Model weights](docs/weights.md) | pinned upstream assets and runtime paths |
| [Known issues](docs/known_issues.md) | current visual, expression, and CI boundaries |
| [Contributing](CONTRIBUTING.md) | development, tests, pull requests, benchmark reporting |
| [Security](SECURITY.md) | private vulnerability reports and deployment precautions |
| [License status](LICENSE_STATUS.md) | unresolved project license and upstream attribution |

## Project status

- Fish S2 Pro is the existing multi-GPU TTS; official VoxCPM2 powers the
  single-GPU profile. Native speech-to-speech remains an alternative route.
- The supported entry point is `bash scripts/flashav2av <command>`.
- Native output is 512 x 512. Enlarging it cannot add model detail.
- A complete fresh-host CUDA/runtime installer, browser-observed E2E latency
  benchmark, and standardized interaction-naturalness benchmark are not
  published yet. The server-side measurement tool is available.

## License and attribution

Project-level licensing is pending; do not assume MIT, Apache-2.0, commercial
permission, or permission to redistribute third-party weights. Consult
[LICENSE_STATUS.md](LICENSE_STATUS.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository does not
replace the licenses or attribution requirements of its upstream projects.

When referencing this implementation, include the repository URL and exact
commit SHA. A paper citation/DOI will be added when available; do not cite the
upstream DyStream paper as if it introduced OpenVA-Dialogue's system changes.

## Acknowledgements

OpenVA-Dialogue builds on [DyStream](https://github.com/XinchengSun/DyStream),
[Pipecat](https://github.com/pipecat-ai/pipecat),
[Fish Speech S2 Pro](https://huggingface.co/fishaudio/s2-pro),
[SGLang-Omni](https://github.com/sgl-project/sglang-omni),
[FunASR/Paraformer](https://github.com/modelscope/FunASR), and
[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h), and
[VoxCPM](https://github.com/OpenBMB/VoxCPM). Please also acknowledge the
upstream methods, model weights, and libraries used in your configuration.
