# 回复首声等待审计：2026-10-30

本次对照 OpenVA、LiveTalking、OpenAvatarChat 和项目锁定的 Pipecat 1.6.0 源码，
并重放实际文本切分函数、上游结束策略和浏览器脚本。结论是：**“用户说完才提交这一轮”
是正常的轮次边界；当前实现还在提交前、首句交给 TTS 前、媒体播放前三处等待。**
已确认这些等待存在，但没有取得当前运行服务的逐段时间戳，不能断言其中哪项占线上延迟最多。

本次仅新增诊断脚本和报告，没有改变生产代码、分句、模型、精度、采样参数、画面或播放缓冲。
所有重放均使用合成文本或伪媒体 API，不构成语音质量、模型效果或浏览器端到端验收。
结构化结果及源码校验值见 [reply_latency_audit_20261030.json](reply_latency_audit_20261030.json)。

## 1. 提交前：ASR 定稿不会取消所有等待

模板选择 `PIPECAT_MSE_DIALOG_MODE=custom_cascade`，实际服务器是否使用此配置尚未核实。
该路线使用 Paraformer 流式 ASR，默认音频块为 600 ms；说话期间已经识别，
VAD 停止后只需刷新剩余音频和 ASR 状态。**600 ms 音频块不是每轮说完后固定追加 600 ms。**

本地 VAD 默认停止门限 200 ms，随后 `SpeechTimeoutUserTurnStopStrategy` 的
`user_speech_timeout=0.45` 是独立的“允许用户继续说”的等待。项目锁定版本的源码明确规定：
`finalized=True` 只提前结束 STT 安全计时器，不提前结束这个 450 ms 策略计时器。
因此，在 ASR 足够快、调度正常、默认配置的理想条件下，从实际语音停止到允许触发回复，
仅这部分就可能接近 **200 + 450 = 650 ms**。这是配置推导，不是麦克风实测。

实际执行上游类的计时方法，替代的只有基础帧类型和 asyncio 任务管理适配器：

| 人为安排的 ASR 定稿时机 | VAD 停止→定稿实测 | VAD 停止→轮次结束实测 | 定稿后继续等待 |
| --- | ---: | ---: | ---: |
| VAD 后 50 ms | 57.42 ms | 446.42 ms | 388.99 ms |
| VAD 后 700 ms | 720.70 ms | 720.76 ms | 0.06 ms |

这说明两项主要重叠，不能把 ASR 收尾时间与 450 ms 顺序相加。
Windows 定时器和调度存在偏差，表中不是模型速度或服务 P50/P95。
现有代码注释还记录了较短等待误切用户思考停顿、随后取消刚启动请求的问题，
因此没有盲目把 450 ms 调小。

本地依据：`pipecat_dystream/custom_cascade.py::_vad_params/_create_user_aggregator_params`、
`funasr_streaming.py::run_stt/_finalize_utterance`、`bot.py::create_user_aggregator_params`。

## 2. 提交后：拿到首个 token，仍然不能开始 TTS

OpenVA 的 `RecoveringOpenAILLMService._push_llm_text` 缓冲流式文本。
它通常只在 `。！？!?`、英文句号和省略号等完整句边界释放，逗号、分号和冒号不会释放。
此外，即使句号已到达，只要它处于当前块末尾，也会继续等下一块或 EOF，
以便把晚到的引号、括号、标点及空白附着在原句上。无标点回答会一直缓冲到自然 EOF。

`LocalPCMTTSService` 已设置 `TextAggregationMode.TOKEN`，避免 Pipecat 再做一次句子聚合，
**但上面的 LLM 层缓冲依然存在**。因此“请求开启 stream=True”不能等同于首个 token 马上播出。

对照开源设计：

| 实现 | 应用层何时提交文本给 TTS | 实际差别及限制 |
| --- | --- | --- |
| OpenVA 当前自定义级联 | 完整句，且边界处保留下一块前瞻 | 上下文较完整，但长首句和边界等待会推迟首声 |
| LiveTalking `llm.py` | 遇到逗号、分号等标点且累计长度大于 10 时提交 | 可以先说一个分句；不是无标点、无限短文本也立即提交 |
| OpenAvatarChat 本地 CosyVoice handler | 以逗号等标点切分 | 分块更早，但该实现还过滤字符、追加句号；不能照搬后宣称语音不变 |
| OpenAvatarChat 百炼 CosyVoice handler | 每个文本块直接调用同一合成器的 `streaming_call` | 应用支持连续追加文本；供应商内部何时发声仍需测量 |

本机重放执行 OpenVA 原始解析函数，以及 LiveTalking 原始 `llm_response` 函数，
没有重新手写它们的切分规则。下表的时间是**人为设定的文本到达时间轴**，不是在线模型延迟：

| 合成输入轨迹 | OpenVA 首次释放 | LiveTalking 首次释放 | TTS 文本请求数 OpenVA / LiveTalking |
| --- | ---: | ---: | ---: |
| 逗号分句依次在 0、300 ms 到达，句号在 600 ms，下一句在 850 ms | 850 ms | 0 ms | 2 / 4 |
| 14 字完整句在 0 ms 到达，EOF 在 400 ms | 400 ms | 0 ms | 1 / 1 |
| 无标点回答在 0 ms 到达，EOF 在 700 ms | 700 ms | 700 ms | 1 / 1 |
| 引号跨块到达 | 80 ms | 80 ms | 2 / 1 |

四组重放的拼接文本逐字相同，SHA-256 一致。这只证明文字没有丢失或变化，
**不证明改变 TTS 分块后声线、韵律、连贯性、首包或 RTF 一定不变**。
另一次真实 asyncio 等待测试中，注入 300 ms EOF 停顿后，OpenVA 到 303.23 ms 才释放完整句。
这个实验确认前瞻等待能够挡住已经完整到达的句子；不能把 303 ms 当成线上平均浪费时间。

## 3. 语音生成后：还要等待媒体时间轴

当前文本句子通过 `LocalPCMTTSService.run_tts` 发送 `synthesize(text=完整文本)`，
响应音频可以分块返回；这和允许在同一次合成中不断追加文本是两种能力。
此处还要测实际 TTS 的排队、预填充、首个 PCM、首个非静音 PCM，不能用接口的“流式”名称代替测量。
每次句子请求还会建立本机 WebSocket，开销未实测，不能直接认定它是大头。

PCM 到达后才进入数字人运动、渲染、编码和浏览器。源码默认 `PIPE_FRAME_STRIDE=2`、
25 FPS 源网格、`PIPE_GOP=3`，对应输出 12.5 FPS、一个 GOP 约 240 ms；
渲染和编码耗时需另测，不能把帧长当成每一层固定累加的延迟。

实际执行完整浏览器生产脚本，用伪媒体 API 验证 5 种状态：

| 状态 | 已缓冲媒体 | 是否调用播放 |
| --- | ---: | --- |
| 首次启动 | 0.59 s | 否 |
| 首次启动 | 0.60 s | 是，起播后目标保留 0.48 s 媒体 |
| 替换播放器 | 0.15 s | 否 |
| 替换播放器 | 0.16 s | 是，起播后目标保留 0.08 s 媒体 |
| 已经起播 | 0.10 s | 是，不重新套用首次起播门限 |

**0.60 s 是媒体量门限，不等于每轮回复固定等 600 ms 墙钟时间。**
客户端还会等媒体播放时钟到达 `assistant_media_boundary` 才解除静音。
伪 API 测试不覆盖解码、实际发声、网络抖动或音画同步；本次没有降低缓冲。

仓库有 2026-08-11 的 Fish bridge 首包记录（不同配置 P50 约 426–1259 ms），
但它是旧环境的单独 TTS 测试，不是本轮实测，也不能套用于当前 VoxCPM2 或完整服务。

## 4. 为什么现有日志还不能回答“谁是线上大头”

- `mse_session.py` 的 `first audio latency_ms` 从接收 `TTSStartedFrame` 开始计时，
  不是从用户真实停止说话开始；也不能未经核对事件位置就把它当作纯 TTS 模型耗时。
- 引擎的 `PIPELINE LATENCY ... after_engine_pcm_ms` 从首段 PCM 进入引擎开始，
  不包含 ASR、轮次等待和之前的大模型等待。
- `probe_custom_e2e.py` 能观测用户停止到首音频/媒体边界，但目前不能拆开 ASR 定稿、
  LLM 请求、首 token、首句释放；媒体边界到达也不等于浏览器已实际播放。
- 没有本次线上逐段轨迹，因此不发布整轮延迟占比、优化百分比或 GPU 加速结论。

下一步按同一个轮次 ID 连起：真实语音停止、VAD 停止、ASR 定稿、LLM 请求、
首个正文 token、首句提交、TTS 首个非静音 PCM、首帧及浏览器实际解除静音。
优先验证已经复现的完整句前瞻等待；逗号提前分块、调整结束策略和降低媒体缓冲，
必须分别检查韵律、误切轮次、卡顿及音画同步，不合并改参数后再猜收益。

## 5. 来源、复现和验证边界

审计生产基线：`9bae6c04de8786216dd39c6a7395bcbf01e508c2`。
上游只读快照和原始输出在仓库外
`D:/桌面/codex_outputs/openva_infra_20261026/evidence/latency_audit_20261030/`。

- [LiveTalking 固定版本](https://github.com/lipku/LiveTalking/blob/c7269433a2b6ecebcd9342445d2ecc882c98b0da/llm.py)
- [OpenAvatarChat LLM 固定版本](https://github.com/HumanAIGC-Engineering/OpenAvatarChat/blob/8b7b3b45bca28ae9ab5fa72ee31bc03c5a99b08b/src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py)
- [OpenAvatarChat 本地 CosyVoice 固定版本](https://github.com/HumanAIGC-Engineering/OpenAvatarChat/blob/8b7b3b45bca28ae9ab5fa72ee31bc03c5a99b08b/src/handlers/tts/cosyvoice/tts_handler_cosyvoice.py)
- [OpenAvatarChat 百炼双向流式 TTS 固定版本](https://github.com/HumanAIGC-Engineering/OpenAvatarChat/blob/8b7b3b45bca28ae9ab5fa72ee31bc03c5a99b08b/src/handlers/tts/bailian_tts/tts_handler_cosyvoice_bailian.py)
- [Pipecat 1.6.0 实际提交中的结束策略](https://github.com/pipecat-ai/pipecat/blob/08e871599904080cedad7ce5683676ab8481fa59/src/pipecat/turns/user_stop/speech_timeout_user_turn_stop_strategy.py)

Pipecat 的 tag 对象是 `35b946f...`，解引用后的代码提交是 `08e8715...`，
本次使用后者，不以会漂移的 `main` 替代项目锁定版本。

```powershell
$auditRoot = 'D:/桌面/codex_outputs/openva_infra_20261026/evidence/latency_audit_20261030'
python scripts/audit_reply_release.py `
  --livetalking-source "$auditRoot/livetalking_llm.py" `
  --pipecat-source "$auditRoot/pipecat_speech_timeout.py" `
  --output "$auditRoot/policy_replay.json"
node scripts/audit_browser_startup.js
node tests/test_media_buffer_bounds.js
node tests/test_mic_resampler.js
```

已运行：4 组文本重放、2 组上游计时策略重放、300 ms EOF 停顿墙钟确认、
5 组浏览器门限重放、现有 6 组媒体状态检查、44.1/48 kHz 麦克风重采样检查。
本机没有 Pipecat 运行时，因此文本和策略测试按脚本说明加载实际 AST 节点，
没有运行完整 Pipecat 管线。未调用外部模型 API、未启动 GPU 服务、未读取私人会话文本。
