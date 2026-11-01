# Listener continuity and complete replies (2026-11-02 local time)

Core change: `3697318`. Models, weights, precision, native 512×512 / 25 fps,
TTS reference audio and inference steps are unchanged. This follow-up corrects
playback/control policy and response limits; it does not claim a new model speedup.

## What the user observed

The user reported frequent Listener stalls, a frozen portrait when switching to
thinking, unusually fast-looking mouth motion, and a story that stopped after
announcing its title. The live story requests returned only about 25 characters
with provider `finish_reason=stop`; the corresponding TTS requests completed.
That particular trace does **not** prove that TTS discarded a longer reply.

Two additional limits were present: local code stopped all replies after three
sentences, and the provider budget defaulted to 256 tokens. In a separate raw API
probe, the old policy produced a 416-character story fragment ending with
`finish_reason=length`. The old local three-sentence cap could stop even earlier.

## Changes

- Remove the added speech-preroll `video.pause()` and its forward seek. Listener
  continues rendering during thinking. Initial startup also keeps the first
  encoded frame/audio rather than seeking away buffered content.
- Keep the entire live stream at 1×, including Listener and transitions. Remove
  idle catch-up speed changes and forward seeks. The two-second startup reserve,
  FIFO append queue and turn/generation/epoch audio gate remain.
- Preserve the feed deadline while waiting for backpressure to clear. The old
  retry assignment restarted the clock and could accumulate phase loss. The
  existing bounded catch-up and all queue limits remain; no audio/frame is
  intentionally skipped to create throughput.
- Allow `reply_max_sentences=0` to disable the sentence cutoff; negative values
  remain invalid and explicitly configured positive limits still work. The
  custom cascade defaults to zero and a 2,048-token provider budget. The prompt
  requests an actual complete story with an ending, about 150 Chinese characters
  when no length is specified. Longer explicit requests can exceed three sentences.
- Add opt-in feed timing records with mode and deadline lag, without audio/text.

## Validation

All tests use a file-backed synthetic microphone in isolated Edge with normal
autoplay policy. New tests invoke the button with a user gesture. The old code's
quiet 60-second Listener baseline was already approximately 25 fps without
post-start buffering events; it did **not** reproduce every reported stall.

| New run | Duration | Browser decode rate | Post-start waiting | Thinking playback |
|---|---:|---:|---:|---|
| Complete story | 75 s | 24.99 fps | one 30.7 ms Listener wait; none during speech | 2.7015 s elapsed / 2.7020 s media, 68 frames, no paused samples |
| Long spoken input then acknowledgement | 75 s | 25.01 fps | none | 2.5991 s elapsed / 2.5990 s media, 66 frames, no paused samples |

Both new runs had only 1× playback, no seeks, and no browser exceptions. They had
four and one dropped browser frames respectively. This is finite-sample evidence,
not a guarantee for every network or the user's embedded browser.

The real story contained **five sentences / 180 Chinese characters**. All five
sentence hashes reached TTS and all five requests completed; output PCM covered
**35.36 seconds**. AAC decoded from the captured browser media was transcribed
back from the opening about A-Ming and the injured bird through the final line
about gentleness/persistence and being seen. This verifies complete transport
audio, not a perceptual score or physical loudspeaker capture.

Three independent new-policy Qwen 3.8 API probes produced complete 156/160/166-
character stories with a situation, event and ending; all ended naturally.
These are response-quality spot checks, not a general language-model benchmark.
The raw API probe does not exercise the local sentence parser; a separate
regression confirms that a six-sentence story retains its ending with cap zero.

Captured H.264 streams remain native 25 fps. Audio/video track duration
differences were about 19 ms and 5 ms. **Track duration is not a lip-sync or
mouth-naturalness metric.** The story speaks roughly 5.1 characters per second.
The user's impression of fast mouth motion still needs to distinguish speech
pace from mouth motion/phoneme alignment; no blind mouth-motion filtering or
voice time stretching was introduced.

The simulated backpressure regression drives the actual production clock branch
with alternating 230/170 ms readiness. It preserves approximately 25 fps over
60 seconds instead of accumulating the retry debt. It is not a GPU benchmark.
The candidate CPU suite passed 118 tests (117 passed, one skipped), and 25 local
clock/startup/playback/transport/tracing tests passed. Frozen trained-model
judges and reference tensors were not modified; inference math code did not change.

The whole-device story sampler peaked at **9,749 MiB**; the later live ownership
audit also read 9,749 MiB. The long-input run was not covered end to end by that
sampler. Keep the earlier broader **9,885 MiB** observation as the conservative
observed capacity reference. Actual 12 GiB hardware and long-session stability
remain untested. Do not add component allocator numbers to estimate capacity.

Private evidence includes exact microphone fixtures, encoded browser MP4s,
stage hashes/timestamps, API probes, decoded audio/transcription and GPU samples.
No private media, credentials or deployment addresses are committed here.
