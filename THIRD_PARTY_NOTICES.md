# Third-party model notices

This file is an inventory, not legal advice and not a replacement for the
upstream license text. FlashAV2AV's repository does not include the model weights
listed below. Users download them directly from the named upstream repositories
and must review the license version attached to the downloaded revision.

## DyStream motion and LIA renderer

- Upstream: <https://huggingface.co/robinwitch/DyStream>
- Recorded revision: `9ad5b1d3b0ef7aece4b9855d972c1f819f04dfbb`
- Files used: `checkpoints/last.ckpt` and
  `tools/pretrained_model/epoch=0-step=312000.ckpt`
- License status: no upstream model card or license file was found during the
  audit. Absence of a license is not permission to redistribute. FlashAV2AV
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

Fish S2 Pro is the primary TTS in the current deployment. VoxCPM2 is retained
as a compatible fallback.

## Reference media

Reference portraits, cloned-voice recordings, transcripts, Listener recordings
and customization uploads are not distributed. Rights to a model do not grant
rights to a person's voice, image or recording.

## MediaPipe Face Landmarker

- Upstream: <https://developers.google.com/mediapipe/solutions/vision/face_landmarker>
- Bundled task file: `tools/visualization_0416/utils/face_landmarker.task`
- Project license: Apache License 2.0. Review the model-card terms and notices
  distributed by the upstream project for the exact task bundle.
