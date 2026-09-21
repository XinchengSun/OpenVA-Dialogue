# Third-party notices

This file is an inventory, not legal advice and not a replacement for the
upstream license text. The large motion, rendering, and speech weights listed
below are downloaded from their named upstream repositories, not mirrored here.
The existing bundled MediaPipe task is an exception, identified below. Review
the license attached to each exact revision. The project-level license and
incomplete source-provenance audit are described in [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Source-code provenance

The motion code in `model/`, audio encoders, and rendering code in
`tools/visualization_0416/` include upstream research implementations, rather
than exclusively original OpenVA-Dialogue code. The starting project is
[DyStream](https://github.com/XinchengSun/DyStream), including its LIA-based
renderer. Preserve existing author headers and per-file notices.

This inventory is not a complete file-by-file license audit of the inherited
DyStream/LIA and other research utilities. Their rights and redistribution terms
must be verified before assigning a repository-wide license. A license for a
model checkpoint does not establish a license for its source code, or vice versa.

## DyStream motion and LIA renderer

- Upstream: <https://huggingface.co/robinwitch/DyStream>
- Recorded revision: `9ad5b1d3b0ef7aece4b9855d972c1f819f04dfbb`
- Files used: `checkpoints/last.ckpt` and
  `tools/pretrained_model/epoch=0-step=312000.ckpt`
- License status: no upstream model card or license file was found during the
  audit. Absence of a license is not permission to redistribute. OpenVA-Dialogue
  therefore downloads only from the author's official repository and does not
  mirror these files.

## Wav2Vec2 Base 960h

- Upstream: <https://huggingface.co/facebook/wav2vec2-base-960h>
- Recorded revision: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`
- Declared license: Apache License 2.0

## VoxCPM2

- Upstream: <https://huggingface.co/openbmb/VoxCPM2>
- Declared license: Apache License 2.0

## Paraformer streaming ASR

- Upstream: <https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online>
- Declared license: Apache License 2.0

## SenseVoiceSmall

- Upstream: <https://www.modelscope.cn/models/iic/SenseVoiceSmall>
- Declared license: Apache License 2.0

## Fish Audio S2 Pro

- Upstream: <https://huggingface.co/fishaudio/s2-pro>
- Recorded revision: `1de9996b6be38b745688de084d87a5633f714e4e`
- License: Fish Audio Research License
- Required upstream attribution: `This model is licensed under the Fish Audio
  Research License, Copyright © 39 AI, INC. All Rights Reserved.`
- The upstream license permits research and non-commercial use under its terms.
  Commercial use requires a separate written license from Fish Audio. It also
  imposes distribution, attribution, acceptable-use and other conditions; read
  the full upstream `LICENSE.md` before use or distribution.

Fish S2 Pro is the TTS for the existing multi-GPU profile. Official VoxCPM2
is the TTS for the opt-in single-GPU profile.

## Reference media

Reference portraits, cloned-voice recordings, transcripts, Listener recordings
and customization uploads are not distributed. Rights to a model do not grant
rights to a person's voice, image or recording.

## MediaPipe Face Landmarker

- Upstream: <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker#models>
- Bundled task file: `tools/visualization_0416/utils/face_landmarker.task`
- Pinned original: <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
- Size: `3758596` bytes. SHA256:
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
  The included file matched this official version byte-for-byte on 2026-09-21.
- License: Apache License 2.0, as stated on page 1 of each component's official
  model card: [BlazeFace Short Range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf),
  [Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf),
  and [Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf).
- A copy of the license is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
  This attribution and license apply to that bundle, not the whole repository.
