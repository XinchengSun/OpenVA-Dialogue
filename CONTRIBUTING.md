# Contributing to OpenVA-Dialogue

Issues and pull requests in English or Chinese are welcome. Read
[LICENSE_STATUS.md](LICENSE_STATUS.md) before submitting code; project-wide
licensing is still pending. Submit only material you have the right to share,
and preserve all existing third-party attribution. No contributor license
agreement or blanket relicensing is implied by this guide.

## Report a problem

Search existing issues, then provide the commit SHA, selected profile, GPU/VRAM,
OS, Python/PyTorch/CUDA versions, minimal reproduction, and expected/actual result.
Distinguish the single-GPU VoxCPM2 profile from the multi-GPU Fish profile.
Remove API keys, access/admin tokens, private hosts, conversation transcripts,
and identifiable media from logs. Report vulnerabilities privately via
[SECURITY.md](SECURITY.md), not a public issue.

## Develop and test

Use a feature branch and keep changes scoped. Existing `scripts/flashav2av`
commands and `FLASHAV2AV_*` names are intentional compatibility interfaces.
Do not rename them without a documented migration path. Use a separate runtime
and ports for integration tests; do not restart a shared production service.

Lightweight source and configuration checks (Python 3.11, from repository root):

```bash
python -m compileall -q .
python -m unittest tests.test_weights_manifest tests.test_configure_custom_runtime tests.test_s2s_launcher_contract -v
python -m unittest tests.test_cfg_pruning tests.test_single_gpu_face_detector tests.test_benchmark_single_gpu -v
```

The complete CPU/shell release checks and exact test dependencies are maintained
in [.github/workflows/ci.yml](.github/workflows/ci.yml). Audio customization tests
need `ffmpeg`/`ffprobe`; EMA and history tests need CPU PyTorch, NumPy, and
`torch_ema`. Tests use temporary fixtures, not private portraits or credentials.
Do not use the legacy root `requirements.txt` as the realtime environment
installer; follow the [deployment guides](README.md#documentation).

CI checks source contracts and CPU regressions; it does **not** validate GPU
inference, perceptual quality, or browser/public-network latency. Runtime changes
also need the relevant [single-GPU checks](docs/single_gpu_4090.md#reproduce-validation)
or multi-GPU integration checks on authorized hardware. State which checks you
ran, which were skipped, and why.

## Pull request checklist

- Explain the issue, scope, and behavioral changes; link an issue if available.
- Add a regression test for fixes and document configuration changes.
- Run `git diff --check` and the relevant tests; attach sanitized results.
- Do not commit model weights, `.env` files, tokens, runtime caches, or private media.
- Preserve upstream headers and identify the source/license of new dependencies.
- Update both English and Chinese READMEs when changing user-facing capabilities.

## Report performance accurately

Include the exact commit, hardware, model revisions, profile, warm-up, run
duration, sample count, and measurement boundaries. Separate TTS first PCM,
server media-boundary latency, and browser-observed playback latency. Report
rendered/delivered FPS separately from model/motion FPS, and state whether GPU
memory is sampled whole-device memory or a process allocator statistic.
Do not label a short run as long-term stability, or visible movement as a
validated naturalness score. Use authorized or synthetic inputs and keep raw
private media outside Git.
