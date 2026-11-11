# 基础设施改动的反例审查与当前结论

用户要求：先证明改动有依据，不能凭猜测调参，再安排其他 GPU 服务器测试。
本轮没有使用 GPU、远程服务器或付费模型接口。

**结论：第一版 `958aee3` 不能视为正确的部署候选。** 本次找到并修复了其行为回归，
撤回了未经测量的默认阈值，并纠正了“局部队列限制等于整条链路背压”的错误推断。
原始基线仍为 `73a2d4846172ad1e1ea1037b5294b922113e8de9`。

## 发现的反例与处理

| 反例 | 第一版或中间版的错误行为 | 当前处理 | 可执行证据 |
| --- | --- | --- | --- |
| 合法的长 metadata 前缀后才出现答案 | 基线接受；第一版 64 分片上限在答案到达前直接报错 | 删除该上限，保留原有首答案期限，验证 metadata、答案、finish、usage 顺序 | `tests.test_llm_backpressure.HedgeBackpressureTests.test_valid_long_metadata_prefix_is_preserved_before_answer` |
| 第一条语句发送失败，第二条沿用同一轮 context | 第一版释放整轮上下文，第二条无法生成；基线可以继续 | 单句失败只关闭本句生成器；显式 cancel / release 才作废整轮 | `tests.test_voxcpm2_infra.VoxCPM2InfrastructureTests.test_failed_sentence_does_not_poison_next_official_sentence` |
| 正常消费者等待时，先调度 64 条诊断日志，再提交媒体边界 | 第一版在输出队列仍为空时拒绝边界并关闭健康连接 | 日志独立排队、独立计数；关键媒体/控制保持原 64 容量和 FIFO | `tests.test_media_backpressure.WebSocketLifetimeTests.test_ready_consumer_survives_log_burst_and_receives_boundary` |
| 浏览器同步追加失败或媒体连接被服务端关闭 | 第一版只关闭画面，麦克风继续采集、Start 禁用，无法正常重启 | 结束采音、重置音频与时间线状态、恢复 Start；可安全回收旧缓冲时重试原字节块 | `node tests/test_media_buffer_bounds.js`，执行完整生产脚本及真实停止/开始函数 |
| SDK 底层关闭尚未完成就强制超时 | 真实 OpenAI/httpx 的 response 已标记 closed，但合成 transport 未释放；重试 close 也无效 | 移除强制 1 秒超时，保留清理任务所有权，待实际关闭后传播取消 | `tests.test_llm_backpressure.HedgeBackpressureTests.test_real_sdk_close_waits_for_transport_release_after_cancellation` |
| 移除 64 上限后，primary 连续产生已缓冲 metadata | 中间版可能持续抢共享队列，已就绪 hedge 等待 primary 前缀耗尽 | 生产者显式让出调度，验证就绪 hedge 不被长前缀拖住 | `test_losing_metadata_is_not_emitted_when_hedge_wins`；按读取数量验证，不宣传为 API 加速倍数 |

旧“发送失败必须释放整轮”的测试期待本身是错的，已经更正。因此旧报告中的
“六项基线失败”不能整体当成六个正确修复的证明。当前保留其中五个有效的基线回归重放，
并新增同一轮恢复、显式取消、日志隔离、真实 SDK 清理等反例测试。

## 依据来自真实接口契约

- [Pipecat v1.6.0 TTS 源码](https://raw.githubusercontent.com/pipecat-ai/pipecat/v1.6.0/src/pipecat/services/tts_service.py)
  在一轮内复用 context；普通 ErrorFrame 不等价于结束整轮。本项目的 `LocalPCMTTSService`
  采用这个约定，因此按请求清理与按轮次失效必须分开。
- [Pipecat v1.6.0 FrameProcessor 源码](https://raw.githubusercontent.com/pipecat-ai/pipecat/v1.6.0/src/pipecat/processors/frame_processor.py)
  的普通模式使用无界输入队列。TTS 变慢不会自动把压力传回 hedge 内部队列；
  本项目主 `custom_cascade` 的 hedge 默认也未启用。
- [Python 3.11 asyncio.Queue](https://docs.python.org/3.11/library/asyncio-queue.html)
  的正容量限制队列内容，却不会自动限制人为创建的等待 `put()` 任务数量。
  所以本次既限制诊断消息入队，又限制跨线程等待执行的回调；多余诊断日志计数后丢弃。
- [Python 3.11 aclosing](https://docs.python.org/3.11/library/contextlib.html#contextlib.aclosing)
  支持异常或提前退出时在同一任务上下文清理异步生成器。这支持 TTS 的显式关闭改动，
  不能据此推断真实模型 GPU 显存已经回收。
- [W3C MSE Prepare Append](https://w3c.github.io/media-source/#sourcebuffer-prepare-append)
  说明 quota 情况可移除旧数据以便重试。当前沿用原有 12 秒历史保留边界，一次重试同一字节块；
  没有可安全移除范围或重试失败则结束对话并恢复 Start。不是任意丢弃 fMP4 字节后继续播放。
- [websockets 14.1 send](https://websockets.readthedocs.io/en/14.1/reference/asyncio/server.html#websockets.asyncio.server.ServerConnection.send)
  不建议取消阻塞发送后继续复用连接。可选 TTS 超时会退休连接；该策略仍属实验，默认关闭。

PARD、Harpagon 等论文只提供研究方向，不能为以上每一行代码的正确性背书。
论文资料保留在[第一版历史记录](infra_optimization_20261026.md)，本轮依据是接口契约和可重放反例。

## 默认行为与实验选项

| 项目 | 当前状态 | 证据边界 |
| --- | --- | --- |
| TTS 生成器及加载任务生命周期修复 | 默认启用 | 验证请求/线程任务所有权和可用 close 钩子；实际 CUDA 回收未验证 |
| 官方 TTS PCM 队列上限 | `VOXCPM2_PCM_QUEUE_CHUNKS=0`，默认不启用限制 | 设正值才进入实验；候选值 8 未经生产 profile，不称最优 |
| TTS 新增发送超时 | `VOXCPM2_SEND_TIMEOUT_SEC=0`，默认关闭 | 正值启用实验；有界队列实验仍要求有限停滞期限，避免独占模型锁 |
| 媒体新增发送超时 | `PIPE_WEBSOCKET_SEND_TIMEOUT_SEC=0`，默认关闭 | 原本的编码入队 5 秒/重置通知 2 秒保留，只修复超时任务未取消 |
| 浏览器 8 MiB / 256 块上限 | 已删除 | 尚不能声称浏览器待追加内存有界 |
| hedge 内部队列 | 上限 16，并让出调度 | 仅约束内部等待分片，不是整条链路内存界限，也没有证明 16 为最佳参数 |
| 关闭 SDK 流 | 等待真实清理结束后传播取消 | 胜出首答案不等败选流关闭；退出仍依赖 SDK 合作，没有硬性退出期限 |

真实 TTS 路径是 `LocalPCMTTSService.run_tts → MSE adapter.push_pcm → enqueue_speaker_audio`，
最终 `_speaker_chunks` 仍无容量等待。因此“端到端背压已解决”与“整体时延已改善”都不成立。
完整 Pipecat 集成、模型数值/画质、真正的解码/播放、GPU 吞吐仍待后续验收。

## 本轮验证方式

本轮汇总见 [infra_review_validation_20261026.json](infra_review_validation_20261026.json)：
187 项 Python 测试中 169 项通过、18 项因平台/缺少 Pipecat 跳过；6 个完整页面脚本场景通过。
这些计数只说明覆盖范围和执行结果，正确性依据仍是前面的具体反例和不变量。

除假的模型和浏览器 API 场景，新增两个真实本机 aiohttp WebSocket 集成测试：
空闲连接关闭后注销；64 条日志突发下关键边界仍到达，后续二进制消息与结束边界保持顺序。
它们使用实际握手、收发和关闭逻辑，仅替换 GPU/编码器边界，不声称验证了 MP4 解码。

```bash
python -m unittest tests.test_llm_backpressure tests.test_voxcpm2_infra -v
python -m unittest tests.test_media_backpressure tests.test_media_socket_integration -v
python -m unittest tests.test_benchmark_provenance tests.test_benchmark_single_gpu -v
node tests/test_media_buffer_bounds.js
python tests/bench_media_backpressure.py
```

本地 SDK 验证使用 OpenAI 2.35.1、httpx 0.28.1 的实际方法和合成 transport，无供应商调用。
完整页面脚本测试使用假的 DOM/MSE，不是真实浏览器解码验收。AST 提取测试验证生产方法，
无法替代完整进程启动和依赖集成。所有统计与跳过项必须随报告一起看。

比较工具新增测量代码 SHA-256 检查：即使硬件、输入、参数一样，换了测量实现也不能直接标记
相同记录条件。服务提交仍是人工核对后的声明，权重清单摘要仍不等于已加载权重校验。

## 上卡前后分别能确认什么

上卡前可以确认：反例确实触发、修复保持已验证的正常消息/PCM 顺序、资源清理调用关系正确、
未验证阈值默认关闭、代码和测试可追踪。不能用测试数量代替这些具体事实。

上卡后再比较：同硬件、同权重、同输入、同测量工具，分别运行原始基线和修订候选；
先验证正常对话与打断不退化，再分阶段测量首包、帧龄、真实播放、显存、长回复和慢客户端。
一次只启用一个实验选项，记录成功/失败请求及质量变化。没有通过实机对照的参数，不进入默认配置。

本轮代码状态是**经反例审查修订的测试候选**，不是“保证全部正确”或“已经加速部署”的结论。
