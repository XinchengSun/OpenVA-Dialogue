# ASR、LLM 与 TTS 分段计时

`OPENVA_LATENCY_TRACE_DIR` 非空时，服务把单调时钟时间戳写到指定目录的
`stages-PID.jsonl`。默认关闭；诊断不修改音频、生成参数、分句或播放策略。
记录文本长度及 SHA-256，不写原始问题、回复、音频或凭据。
写文件失败时跳过诊断，保持会话工作。

```bash
export OPENVA_LATENCY_TRACE_DIR=/absolute/path/to/diagnostic-run/trace
# 在同一启动环境中运行原有服务入口；测试后取消该变量。
```

| 事件 | 含义 |
| --- | --- |
| `vad_start` / `vad_stop` | ASR 收到 VAD 开始/停止帧 |
| `asr_infer_start` / `asr_infer_end` | 每块流式识别及最终尾块，含样本数、是否 final、耗时 |
| `asr_final_ready` | 一个 VAD 片段的识别结果已可下发 |
| `llm_request` / `llm_first_content` | 提交上下文、收到第一段正文 |
| `llm_sentence_released` | 完整句子送往下游；含句子序号、文本哈希 |
| `llm_stream_closed` | 开始关闭该供应商流，不等于全部音频已经合成或播放 |
| `tts_request` / `tts_connected` | 提交一次句子合成、连接本地 bridge |
| `tts_first_pcm` / `tts_done` | 收到该请求的首段 PCM、合成结束 |

同一长问题可能含多个 VAD 片段，用户聚合器会拼接它们；不能把最后一个
`asr_final_ready.characters` 当作整句字数。LLM 请求编号与 TTS 请求编号
属于不同命名空间，通过时间窗口、句子哈希、TTS context 对照。
跨主机不能直接相减 `monotonic_ns`；同机同 PID 的各阶段可以直接比较。

如果 `tts_request` 早于 `llm_stream_closed`，说明该轮已经在 LLM 流结束前
启动合成。如果第一句很长、始终没有结束标点，现有句子缓冲可能一直积攒到
自然 EOF；这与 ASR 是否流式是两件事。`TextAggregationMode.TOKEN` 在本项目
接收的是上游已经组成的句子，不会把 VoxCPM2 变成增量文本输入模型。

客户端工具 `scripts/probe_custom_e2e.py` 另测语音结束到首 PCM、服务端媒体边界
及媒体结束；媒体边界不等于浏览器或声卡实际开始播放。需保留原配置、输入、
代码版本、设备身份和原始 trace；计时 A/B 两侧应使用相同诊断设置。

参考 [ResearchBoost-Agent 方法](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/main/docs/methodology.md)
固定工作负载、精度与语义边界，并把正确性检查和性能测量分别保留。
[LiveKit 的提前生成方案](https://docs.livekit.io/agents/logic/turns/tuning/)需要处理
转写变化后的取消；[Pipecat 的聚合说明](https://reference-server.pipecat.ai/en/stable/api/pipecat.services.tts_service.html)
也提示文本粒度会影响语音质量。这里仅增加可观测性，没有启用提前猜答或更细粒度切句。

`tests.test_latency_trace` 检查默认无 I/O、并发记录完整和写入失败隔离；
ASR、LLM、TTS 原有回归检查继续验证输出与取消契约。具体服务器实验以独立
run 的原始回执为准，本说明不承诺性能收益或 12 GiB 支持。
