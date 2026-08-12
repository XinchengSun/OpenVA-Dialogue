# FlashAV2AV architecture

## Runtime data flow

```text
Browser microphone PCM
  -> WebSocket input
  -> dialogue route
       native_s2s:
         Qwen Audio realtime speech-to-speech
       custom_cascade:
         Silero VAD
         -> Paraformer streaming ASR
         -> resilient OpenAI-compatible LLM
         -> optional provider web search
         -> Fish Speech S2 Pro cloned TTS through SGLang-Omni
  -> assistant PCM (normalized by the AV2AV engine as required)
  -> continuous DyStream motion worker (logical GPU 0)
  -> fixed-source LIA render worker (logical GPU 1)
  -> H.264/AAC fragmented MP4
  -> browser MediaSource Extensions
```

Pipecat owns dialogue orchestration, turn events, and cancellation. The MSE
runtime owns the long-lived motion/render workers, A/V queues, media boundaries,
and browser live-tail behavior.

## Listener and Speaker contract

The official DyStream motion checkpoint has two audio-conditioning branches in
one recurrent model:

- Speaker: assistant audio is routed to `audio_self`; `audio_other` is zero.
- Listener: `audio_self` is zero; authorized Listener conditioning is routed to
  `audio_other`.

This is not a Speaker checkpoint plus a Listener checkpoint. Both modes preserve
the same recurrent motion state. A normal turn changes `turn_id` but does not
reload the model, reset the renderer, or create a new media timeline.

## Interruption contract

A barge-in cancels the active dialogue/TTS request and applies a short graceful
assistant tail. Stale queued data is fenced before it can re-enter the visible
stream. Ordinary interruption remains on the current FFmpeg/MSE timeline; a new
encoder epoch is reserved for genuine encoder or backlog recovery.

## Media boundary contract

Assistant-visible state changes are emitted only after the matching audio/video
unit has been accepted by the encoder path. The browser releases live assistant
audio at the corresponding safe media time, keeping mouth motion and audio on
one fMP4 timeline.

## GPU placement

- DyStream motion and LIA rendering use two distinct logical devices selected by
  `CUDA_VISIBLE_DEVICES`, normally logical `0` and `1`.
- The current deployment assigns two non-overlapping physical GPUs
  to Fish S2 Pro through SGLang-Omni: logical GPU 0 runs the TTS engine and
  logical GPU 1 runs the vocoder.
- The unified lifecycle manages the Fish HTTP service and local PCM bridge.
  VoxCPM2 remains a compatible single-GPU fallback selected explicitly during
  configuration.

## Output and buffering

The native renderer output is 512 × 512. The server encodes H.264 video and AAC
audio into fragmented MP4. The browser starts with a bounded media reserve and
uses low-water playback control to mask short production jitter. CSS enlargement
does not add model detail.
