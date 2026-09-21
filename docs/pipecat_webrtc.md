# Pipecat v1.6 + DyStream WebRTC

## Scope

This entry point adds the first measurable full-duplex cascade without changing the
DyStream motion model, renderer, CUDA graph, or worker split. Listener training and
expression control remain out of scope.

```text
browser microphone
  -> Pipecat Silero VAD
  -> FunASR SenseVoiceSmall (16 kHz, local CPU by default)
  -> OpenAI-compatible streaming LLM endpoint
  -> Piper streaming TTS (local CPU)
  -> DyStream motion GPU 0 -> LIA3D render GPU 1
  -> Pipecat SmallWebRTC synchronized audio/video output
```

The microphone is also copied to the existing official `audio_other` input. Original
TTS audio is never sent directly to the browser: it is consumed by DyStream, then the
engine's already-aligned `(frames, audio)` output is converted to Pipecat image/audio
frames. This prevents a second, unsynchronized playback clock.

## Install without starting GPUs

```bash
cd "${FLASHAV2AV_ROOT:-$HOME/OpenVA-Dialogue}"
bash scripts/install_pipecat_env.sh
$HOME/.venvs/flashav2av/bin/python -m unittest tests.test_pipecat_bridge
```

The venv inherits the delivered DyStream/CUDA packages but keeps Pipecat-specific
packages separate. Model and package caches stay under `$HOME/.cache/flashav2av/pipecat`.
The installer pins protobuf 4.25.9 for the delivered MediaPipe build. Pipecat's
optional protobuf serializer is intentionally not part of this WebRTC path; the
targeted environment checker imports MediaPipe and every Pipecat provider used here.
It also preloads NLTK `punkt_tab`, preventing an implicit download during startup.
A narrow FunASR adapter keeps segmented audio as raw PCM, matching SenseVoice's
input contract instead of the segmented STT base class's WAV-container default.

## Configuration

Copy `.env.example` to `.env`. The first stack intentionally keeps providers
replaceable, but its initial measurable defaults are:

- ASR: `iic/SenseVoiceSmall`, Chinese, CPU;
- LLM: any OpenAI chat-completions compatible endpoint;
- TTS: Piper Chinese voice `zh_CN-huayan-medium`, CPU;
- avatar: existing DyStream workers on exactly two visible GPUs.

Required values:

```dotenv
PIPECAT_LLM_API_KEY=...
PIPECAT_LLM_BASE_URL=https://provider.example/v1
PIPECAT_LLM_MODEL=provider-model-name
```

For Qwen hybrid-thinking models, set `PIPECAT_LLM_ENABLE_THINKING=false`
to minimize conversational latency. Leave it unset for providers that do not
accept this parameter.

When the browser is not on the same host as the server, configure one or more
comma-separated STUN URLs so the server advertises public ICE candidates:

```bash
PIPECAT_ICE_SERVERS=stun:stun.miwifi.com:3478,stun:stun.chat.bilibili.com:3478
```

AutoDL commonly blocks inbound UDP even when STUN succeeds. For a browser reached
through SSH, install coturn once and tunnel TURN over TCP:

```bash
apt-get update && apt-get install -y coturn
```

```dotenv
PIPECAT_TURN_URL=turn:127.0.0.1:3478?transport=tcp
PIPECAT_TURN_USERNAME=choose-a-user
PIPECAT_TURN_CREDENTIAL=choose-a-long-random-password
```

The start script launches the local TURN relay when `PIPECAT_TURN_URL` is set.
Forward both HTTP and TURN from the browser machine:

```bash
ssh -L 6008:127.0.0.1:7860 -L 3478:127.0.0.1:3478 -p SSH_PORT root@SSH_HOST
```

Piper voice files are cached under `$HOME/.cache/flashav2av/pipecat/piper`. The
in-process Piper package is GPL-3.0; this is suitable for internal validation, while
a distributed product should review the license or switch this provider to an external
TTS service. No secret is committed.

## Start and stop after the GPUs are restored

```bash
cd "${FLASHAV2AV_ROOT:-$HOME/OpenVA-Dialogue}"
CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_pipecat_webrtc.sh
bash scripts/health_pipecat_webrtc.sh
bash scripts/stop_pipecat_webrtc.sh
```

The script rejects anything except two visible GPUs. Inside the process they are
logical GPU 0 (motion) and logical GPU 1 (render). Only one WebRTC conversation is
accepted because the two DyStream workers are shared state.

## Interruption contract

When Pipecat reports `UserStartedSpeakingFrame` or `InterruptionFrame`, the avatar
processor immediately:

1. invalidates queued/in-flight DyStream segments;
2. clears speaker audio and tail state in `RealtimeMSEEngine`;
3. requests a generation reset so stale motion/render frames are drained;
4. forwards the interruption so Pipecat cancels LLM/TTS and clears WebRTC playout.

A new TTS context begins a new DyStream generation. Late output from the previous
generation is rejected by an epoch check in the output client.

## GPU acceptance checklist

Do not run this section until two cards are available.

1. Start the server and confirm `/health` reports both workers alive.
2. Connect once at port 7860 and confirm audio and a 512x512 video track arrive.
3. Speak three short Chinese turns and verify the avatar never replays old audio.
4. Interrupt a long answer at least five times; stale audio/video must stop and the
   next question must be recognized.
5. Record VAD stop, ASR final, LLM TTFT, TTS first audio, DyStream first segment, and
   browser first media timestamps.
6. Confirm `nvidia-smi` shows only the two exposed cards used by the process tree.

The first run may download SenseVoice/Piper assets; warm them before measuring latency.
