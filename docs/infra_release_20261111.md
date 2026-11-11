# Realtime infra release · 2026-11-11

This release combines the infra history from `73a2d48` through `f39d918`
with the saved 2026-11-08 20786 dialogue service. It includes inference memory
reuse, reference-encoder offload, bounded media queues, conversation ownership,
streamed TTS clauses, startup diagnostics and continuous listener/speaker playback.

The 20786 update advances speech inference while the browser plays listener
frames. Browser feedback bounds queued speech and listener preparation. Both
roles use the same motion history and play at 1×. Interruptions gradually reduce
the producer's lead, and repeated interruption signals share the active fade.
The initial media reserve is 0.64 seconds. Completed TTS chunks retain their text
in conversation history when a later chunk fails or is interrupted.

The silent other-audio encoder cache stores exact results for matching lengths
on the CPU. Nonzero audio and unprimed lengths use the original encoder. Selected
priming lengths and live cache hits are checked for equality. This adds startup
work, and cache coverage is finite. Its benefits depend on the session length.

## Source provenance

The source archive was saved as `openva-20786-20261108-9bd0dc1b`. These hashes
identify its verified runtime files before publication.

| File | SHA-256 |
|---|---|
| `server_mse.py` | `9bd0dc1b52a018b54c558d4504c89fd5616f69dd56fd272ecc5ec012e3e337be` |
| `dual_gpu_mic_realtime_ui_v8_playbuffer.py` | `e3ff4ae35f1f229fa725bfa90d9ec67ad0a241523d4d680308402227cdc9307f` |
| `static/index.html` | `5888cd4fd673ee1318453b95437e93436d0ac5d354ff39ee0d46934926694e47` |

The published browser has an updated buffer comment. The local Git tests retain
newer reconnect and clause-splitting regressions than the saved archive.
The TTS adapter retains the newer explanatory comment with the same runtime logic.

## Reproduce the configuration

Follow [single-GPU setup](single_gpu_4090.md), keeping your deployment paths,
portrait, voice reference, GPU assignment and private API settings in `.env`.
After configuring the official VoxCPM2 single-GPU backend, add this line at the
end of that private `.env`, then use the existing `scripts/run_demo.sh start`
entry point from the repository root.

```bash
source configs/infra_20786.env.example
```

The preset records the tested Qwen3.6 Flash model name. The configured API
endpoint must support that identifier. The ASR pre-roll is 0.60 seconds, the VAD
stop interval is 0.45 seconds, and reply sentence limits are disabled. Rendering
uses every frame at 512 × 512 and about 25 fps. Listening audio conditioning and
post-generation expression controllers are disabled in this saved configuration.

## Historical GPU and browser measurements

The November 8 run used a 48 GiB RTX 4090 with MPS, CPU Paraformer, API LLM,
local official VoxCPM2 and the existing DyStream/LIA weights. Other service
copies also occupied the GPU. These measurements describe that recorded run.

| Measurement | Recorded result |
|---|---|
| End of microphone speech to first audible browser playback, three turns | 3.05 / 4.13 / 3.42 s, mean 3.53 s |
| Final story duration | 40.4 s, played through the ending |
| Interrupt to next listener motion submission, voice/button | 0.51 / 0.37 s |
| Interrupt to last audible old speech, voice/button | 2.20 / 1.82 s |
| Waiting after initial startup | One 107 ms listener wait |
| Peak simultaneous sampled memory of service-owned compute processes | 10,516 MiB, 10.27 GiB |

The first-sound measurement used browser playback timing with approximately
100 ms polling. It did not measure a physical speaker. The memory figure is a
simultaneous process sum and has not established operation on a physical 12 GiB
card. It cannot be subtracted from the older 13 GiB benchmark as a matched saving.
Remaining queued audio explains the measured delay before an interruption becomes
silent. Network/gateway delays still affect playback.

The separate November 9 listener projection experiments failed the combined
face-shape and motion-preservation requirements. Their weight edits remain
experimental. The source consolidation here follows the saved dialogue runtime.

Publication checks are recorded in `infra_release_validation_20261111.json`.
Those checks exercise CPU, transport and browser code. The historical GPU/browser
measurements above were obtained before this publication run.
