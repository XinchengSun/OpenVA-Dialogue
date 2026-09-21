# README style study

This note records the 2026-08-12 README review used to redesign the
OpenVA-Dialogue (formerly FlashAV2AV) landing page. The historical observations
below refer to the FlashAV2AV README at that time. It is not a ranking of the projects.

## Projects reviewed

| Project | Useful README pattern | Pattern not copied |
| --- | --- | --- |
| [LiveKit Agents](https://github.com/livekit/agents) | precise two-sentence definition, early install and docs | long framework tutorial in the landing page |
| [Pipecat](https://github.com/pipecat-ai/pipecat) | clear value proposition, visible quick-start action, real examples | a large provider matrix |
| [vLLM](https://github.com/vllm-project/vllm) | short product promise and a deliberately compact README | omitting application-level deployment context |
| [SGLang](https://github.com/sgl-project/sglang) | concrete capability summary and performance links | a long News section before the product explanation |
| [Fish Speech](https://github.com/fishaudio/fish-speech) | model identity, benchmark context, direct quick-start routes | vanity counters and promotional badge volume |
| [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat) | demo-first digital-avatar presentation and a clear module table | latency claims without enough measurement context |
| [MuseTalk](https://github.com/TMElyralab/MuseTalk) | visual input/output proof, hardware-scoped speed claims | mixing training and product setup in the main path |
| [LivePortrait](https://github.com/KlingAIResearch/LivePortrait) | showcase before installation and numbered success-oriented setup | paper/author metadata dominating a product landing page |
| [CosyVoice](https://github.com/QwenAudio/CosyVoice) | version/model entry points, evaluation context, separate FAQ | putting every historical model generation before Quick Start |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | feature proof, tested-environment matrix, platform-specific routes | too many badges and long release history on the front page |

## FlashAV2AV issues found

The previous README had useful technical detail, but its hierarchy was closer
to an internal runbook than a public project page:

1. Architecture and three deployment routes appeared before the primary
   Quick Start.
2. Fish S2 Pro was described as primary, while the example env still selected
   native speech-to-speech.
3. The Quick Start looked like a fresh-host installer even though the checked
   CUDA/SGLang-Omni runtime must already exist.
4. Native S2S, Fish, and VoxCPM2 were all expanded in the main path, so readers
   could not see which one to run.
5. The product had no visual proof. A decorative banner was doing the job of a
   real avatar preview.
6. The headline used strong full-duplex wording where the verified product
   behavior is more precisely described as streaming conversation with
   barge-in.
7. The TTS-only 426 ms result was carefully disclaimed, but its method and
   scope were separated from the number.
8. Runtime ports, permissions, compatibility variables, validation checklists,
   and fallback setup made the landing page longer than necessary.
9. The version badge implied a tagged release even though no Git tag existed.
10. The English and Chinese pages mixed product terminology and deployment
    detail inconsistently.

## Rules used for the rewrite

- Explain the product and primary stack before implementation detail.
- Show a real generated-avatar preview; do not fabricate an end-to-end demo.
- Keep one Fish S2 Pro happy path in the main README.
- Put tested-host prerequisites and alternative backends in deployment docs.
- Keep one compact architecture diagram because this is a multi-service AV
  application rather than a single Python package.
- Place the hardware, profile, sample count, date, commit, and exclusions next
  to every latency number.
- Do not add a license badge, paper citation, online demo, or performance claim
  that does not yet exist.
