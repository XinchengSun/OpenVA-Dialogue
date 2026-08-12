# Current Architecture

```text
Browser PCM
  -> Pipecat Silero VAD
  -> SenseVoice/FunASR
  -> resilient OpenAI-compatible LLM
  -> Qwen3 realtime TTS
  -> continuous DyStream motion worker (GPU 0)
  -> fixed-source DyStream render worker (GPU 1)
  -> H.264/AAC fragmented MP4
  -> browser MediaSource Extensions
```

Pipecat owns dialogue orchestration. The mature MSE transport, dual-GPU workers,
A/V queues, and browser live-tail behavior remain in the original DyStream
runtime.

Normal dialogue turns increment only `turn_id`; they preserve recurrent media
state. A real interruption increments `stream_generation`, allowing stale queued
items to be rejected without re-anchoring the renderer source motion.

See the root `README.md` for the full state contract, file ownership, root-cause
analysis, and measured acceptance results.
