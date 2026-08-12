# Known limitations

- The LIA renderer produces native 512 × 512 frames. Enlarging the browser
  element cannot recover details that the model did not generate.
- Expression, emotion, nod, blink, and landmark controls are not exposed as
  stable production controls; earlier hand-written controls produced artifacts.
- Listener naturalness depends on the source checkpoint and conditioning audio.
  FlashAV2AV preserves the official dual-audio-branch model path rather than
  loading a separate Listener checkpoint.
- Real-time search depends on the configured provider and network. It improves
  freshness, not deterministic latency.
- The public repository contains no identity media. Legacy offline examples that
  reference local sample files are not supported from a clean clone.
- GitHub CI does not run GPU inference or a full browser conversation.
- No reproducible microphone-to-visible-avatar E2E benchmark is published yet.
- Required checkpoints live in upstream model repositories and are downloaded
  by the setup tooling instead of being stored in Git.
