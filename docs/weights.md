# Model weights

OpenVA-Dialogue does not store model weights or private reference media in Git. The
machine-readable source of truth is [`weights-manifest.json`](../weights-manifest.json).
Every download goes directly to the model author's official Hugging Face or
ModelScope repository; this project does not operate a weight mirror.

## Install and verify

Run these commands from the repository root:

```bash
python scripts/setup_weights.py download
python scripts/setup_weights.py verify --deep
```

The default download installs all models marked `required`, including Fish
Speech S2 Pro, the primary TTS in the current deployment.

Use `--root /absolute/path` when model storage must live outside the checkout.
The unified `scripts/flashav2av setup` does this automatically under
`$FLASHAV2AV_DATA_ROOT/models/dystream-official` and creates local ignored
symlinks for the runtime.
Set the runtime model environment variables to those installed paths as needed.
`download` skips an existing target unless `--force` is supplied. `verify`
checks presence and recorded byte sizes; `--deep` additionally checks SHA-256
when the manifest contains one. A `null` hash means no trusted hash was
available during publication and is deliberately not guessed.

The OpenVA-Dialogue installer includes `huggingface_hub` in the Pipecat environment.
For a standalone downloader environment, install:

```bash
python -m pip install huggingface_hub modelscope
```

## Inventory and license policy

| ID | Official source | Runtime target | License status |
| --- | --- | --- | --- |
| `dystream-motion` | `robinwitch/DyStream` | `checkpoints/last.ckpt` | No upstream model card/license file was found. Download from the author's repository only; do not re-host. |
| `dystream-lia-renderer` | `robinwitch/DyStream` | `tools/pretrained_model/epoch=0-step=312000.ckpt` | Same DyStream license uncertainty and no-mirror rule. |
| `wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` | `tools/hf_models/wav2vec2-base-960h/pytorch_model.bin` | Apache-2.0. |
| VoxCPM2 | `openbmb/VoxCPM2` | `$FLASHAV2AV_DATA_ROOT/models/VoxCPM2/` | Apache-2.0; installed by `voice_service/setup_voxcpm2.sh`, not duplicated by this manifest. |
| `sensevoice-small` | `iic/SenseVoiceSmall` | `runtime/custom_cascade/cache/pipecat/modelscope/manual/SenseVoiceSmall/` | Apache-2.0; optional final-ASR/customization helper. |
| `fish-audio-s2-pro` | `fishaudio/s2-pro` | `runtime/fish-s2-pro/models/fishaudio-s2-pro/` | Primary TTS for the current Fish/SGLang-Omni deployment. Fish Audio Research License; research/non-commercial use only unless a separate commercial license is obtained. |

Paraformer is installed separately by `scripts/prefetch_custom_asr.sh` into the
exact ModelScope cache used by the Pipecat runtime. This avoids maintaining a
second manual copy that FunASR would ignore.
The exact upstream revision is pinned only where it was independently observed.
An unpinned (`null`) revision intentionally follows upstream's default branch;
record and review a commit before a reproducible public release.

## Private reference assets

Avatar images, voice-cloning recordings, prompt transcripts, Listener audio and
uploaded videos are user-provided identity material, not model weights. They are
outside this downloader and ignored by Git. Do not publish them without explicit
rights to the person's likeness, voice and source recording. Public demos should
use synthetic or separately licensed media and document consent.

## Why weights are not GitHub assets

Several files exceed GitHub's 100 MB ordinary-Git limit, and the DyStream motion
checkpoint also exceeds GitHub LFS's maximum single-file size. LFS or split
GitHub Release archives would duplicate upstream assets, worsen clone/download
behavior and cannot cure missing redistribution permission. Keep GitHub limited
to code, this manifest, checksums and download tooling.

After an authorized installation, keep the downloaded files untracked. Before
publishing, check with:

```bash
git status --short
git ls-files '*.ckpt' '*.pth' '*.pt' '*.safetensors' '*.bin' '*.onnx'
```

The second command should print nothing except an intentionally reviewed small
non-weight fixture, if the project ever adds one.
