# 2026-10-26 基础设施优化与溯源记录

> 历史候选记录：下文描述第一版 `958aee3`，其中部分默认策略和测试期待已被证实存在问题。
> 请以[反例审查与当前结论](infra_correctness_review_20261026.md)为准，勿按下列旧默认值部署。
> 旧测试通过数量不构成正确性认证；保留本文用于追踪前后判断与撤回内容。

本次以公开仓库 `73a2d4846172ad1e1ea1037b5294b922113e8de9` 为基线，
优化流式队列、慢客户端隔离、取消回收和性能测量。代码审查、CPU 合成负载和回归测试
支持下述结论；尚无本次改动的真实 GPU 加速比例、浏览器端到端时延或感知质量结论。

## 来源与采用边界

胡一涛 / Yitao Hu 的身份和研究归属由[天津大学教师页](https://cic.tju.edu.cn/info/1099/5228.htm)
及其链接的[作者主页](https://flashserve.org/zh.html)核对。检索日期为 2026-10-26。
以下区分论文方法、工程借鉴和本项目独立修复；本次没有复制这些论文项目的代码。

| 来源 | 原工作关注的问题 | 本次采用与限制 |
| --- | --- | --- |
| [PARD, EuroSys 2026, arXiv v1](https://arxiv.org/html/2602.08747v1) | 结合流水线上下游状态和整体期限，避免注定超时的请求继续消耗资源 | 借鉴减少无效工作、显式观测丢弃的原则。实际改动是有界队列、失效请求回收和慢客户端隔离；未实现其时延预测或主动丢弃调度算法 |
| [Harpagon, INFOCOM 2025, arXiv v1](https://arxiv.org/html/2412.06161v1) | 用模块 profile 和延迟预算选择推理流水线配置 | 补齐同条件基准的身份记录和比较约束，保留端到端媒体边界指标；未照搬集群批处理策略，也未声称获得论文中的收益 |
| [InstaInfer, SoCC 2024, 作者大学出版记录](https://researchwith.stevens.edu/en/publications/pre-warming-is-not-enough-accelerating-serverless-inference-with-/) | 容器预热之外的库/模型初始化成本 | 用于审查初始化资源的生命周期；修复取消模型加载后的回收。当前服务已有实际模型健康检查，未实现 serverless 预加载系统 |
| [PAT, arXiv v3](https://arxiv.org/abs/2511.22333v3) / [官方代码](https://github.com/flashserve/PAT) | 共享前缀的 LLM decode attention | 当前 LLM 为外部 API，客户端不能替换供应商内核；未集成 |
| [Mosaic, arXiv](https://arxiv.org/abs/2601.06562) / [官方代码](https://github.com/flashserve/Mosaic) | diffusion LLM 动态内存峰值 | 工作负载与当前 DyStream/LIA/VoxCPM 不同；未集成，未借用其性能数字 |

项目继续遵守 [LICENSE_STATUS.md](../LICENSE_STATUS.md) 和
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，研究参考不改变项目或模型授权。

## 问题、改动和证据

| ID | 基线问题 | 实际改动 | 验证入口 |
| --- | --- | --- | --- |
| INF-01 | LLM hedge 使用无上限事件队列；下游停止消费后上游仍可读完整回复 | 队列限 16；每个候选回答前最多 64 个分片，超限明确失败；取消败选流并有界清理 | `tests/test_llm_backpressure.py`，实际方法的 CPU 行为测试 |
| INF-02 | 官方 TTS 同步生产线程可将整轮 PCM 提前投递到事件循环 | PCM 信用额度覆盖队列及待执行回调，默认 8 块，另留终止消息槽位；取消可解除等待 | `tests/test_voxcpm2_infra.py`，慢消费、样本保真、满队列错误、缓存提交测试 |
| INF-03 | 断线发生在发送处时，后端生成器可能延迟到垃圾回收才关闭；取消加载遗失已分配模型 | 显式关闭 bridge / Nano 生成器，释放未完成上下文；取消加载后等待线程返回并回收模型 | 同上，包含同任务关闭、加载重复取消和资源所有权测试 |
| INF-04 | 引入背压后，仍连接但不读取的客户端可能长期持有全局 TTS 模型锁 | 生产端停滞、WebSocket 发送默认各限 5 秒，超时终止请求并回收；第二上下文可继续 | 同上，两个上下文及 PCM/错误消息发送同时阻塞的回归 |
| INF-05 | 媒体队列虽有上限，后台 `queue.put` 任务仍可无限增长；超时投递未取消 | 跨线程回调数量有界；日志按 best effort 丢弃并计数；关键控制无法投递时关闭该客户端；取消超时 future | `tests/test_media_backpressure.py`、`tests/bench_media_backpressure.py` |
| INF-06 | 空闲日志 WebSocket 未消费对端关闭；初始化失败和阻塞发送可能遗留资源 | 同时监听关闭和输出，发送限 5 秒，退出时取消并等待任务、注销客户端 | `tests/test_media_backpressure.py` |
| INF-07 | 浏览器 `pendingSegments` 无上限；追加失败时原实现先移除字节块再继续 | 待追加内容限 8 MiB / 256 块；成功追加才出队；失败关闭整条媒体流并提示重新连接 | `tests/test_media_buffer_bounds.js`，CPU JavaScript 场景，无真实浏览器解码 |
| INF-08 | 基准缺少代码、配置、输入身份；手工比较容易忽略运行失败或条件变化 | 增加 provenance、帧龄/媒体边界分布及运输计数白名单；独立比较器标记条件缺失/不一致 | `tests/test_benchmark_provenance.py` 和原单卡基准测试 |

INF-04 是本次交叉审查发现并修复的背压副作用。LLM 清理也经过慢 `close()` 场景审查：
不能为了等待败选流关闭，拖住已经到达的胜出答案。

## 可复现的 CPU 压力证据

```bash
python tests/bench_media_backpressure.py
```

实验从 Git 中读取上述固定基线与当前 `server_mse.py`，输出两份源码 SHA-256。
同样投递 10,000 条合成日志且不设消费者时：

| 指标 | 基线 | 改后 |
| --- | ---: | ---: |
| 队列中日志 | 64 | 64 |
| 等待入队的异步任务 | 9,936 | 0 |
| 显式计数的 best-effort 日志丢弃 | 原实现未提供 | 9,936 |

这是诊断日志过载实验，没有音频或编码视频字节参与；不能换算为 GPU 吞吐或端到端加速比例。
LLM 的慢消费测试使用 2,000 个合成分片，检查上游读取有界；TTS 使用合成 PCM、队列为 2，
检查一块已交付时最多再有两块排队和一块在生产端。测试同时验证正常流顺序、样本不丢失和终止错误。

## 回归命令

本地结果见 [infra_validation_20261026.json](infra_validation_20261026.json)：174 项 Python
测试中 156 项通过、18 项因平台或缺少 Pipecat 跳过；3 个媒体前端场景、8 项定制页面测试和
麦克风重采样测试通过。另将 6 项新增行为测试运行在原始基线上，均出现预期断言失败、无导入错误，
同样测试在优化版通过，确认测试确实覆盖了原问题。

Python 3.11 为支持的服务器版本。CPU 测试依赖与 GitHub CI 中的固定版本保持一致；本地
Windows 验证使用 Python 3.12.3 / CPU PyTorch 2.10.0，因此不能替代 Linux 运行时验收。

```bash
python -m unittest tests.test_llm_backpressure tests.test_benchmark_provenance tests.test_benchmark_single_gpu -v
python -m unittest tests.test_voxcpm2_infra tests.test_official_voxcpm2_backend tests.test_voxcpm2_tts -v
python -m unittest tests.test_media_backpressure -v
node tests/test_media_buffer_bounds.js
node tests/test_mic_resampler.js
python -m compileall -q .
git diff --check
```

部分测试直接提取生产源码的类/方法，用假的模型或网络驱动实际分支，不导入完整 GPU 模型栈。
它们验证生命周期和队列语义；完整 Pipecat 集成测试缺依赖时会跳过，必须在报告中单列。
CI 已增加这些回归入口；本地通过不表示托管 CI 已运行。

## 同条件 GPU 验收与溯源

在有完整权重、支持运行时和足够显存的独立服务上，先完成预热并记录条件；基线和优化版各用
一个干净提交。固定输入录音、提示词、模型/权重、分块和 stride，重复运行并覆盖打断、长回复、
慢客户端、断线重连。不要在共用服务上未经协调重启进程或占用其麦克风会话。

```bash
# 服务端本机、同一 PID namespace；替换 PID、路径和 GPU 索引。
# 先将 SERVICE_COMMIT 设置为人工核实过的完整服务提交 SHA。
python scripts/benchmark_single_gpu.py \
  --base ws://127.0.0.1:7871 --gpu 0 \
  --server-pid 1234 --server-pid 2345 \
  --profile single_gpu_voxcpm2 \
  --service-commit "$SERVICE_COMMIT" \
  --runtime-config /data/openva/runtime.env \
  --duration-sec 180 --input-wav /data/openva/authorized-test.wav --turns 6 \
  --output-dir /data/openva/bench/candidate-01

python scripts/compare_benchmarks.py \
  /data/openva/bench/baseline-01/report.json \
  /data/openva/bench/candidate-01/report.json
```

`--output-dir` 必须是新目录。`provenance.json` 在采集前写入，最终报告也携带该对象，记录
测量端 Git 提交/脏工作区、包版本、配置与输入 SHA-256、权重清单摘要和测量参数。
不保存配置正文、提示词正文或进程环境；原有基准仍会保存采集的媒体，使用已授权测试输入。
权重清单摘要不是已加载权重校验，测量脚本所在提交也不是服务端代码的自动证明，必须另行核对。
基线服务也使用同一份新版测量工具采集；用 `--service-commit` 分别声明人工核对过的服务提交，
避免把测量工具版本当作服务版本。旧工具生成且缺少溯源字段的报告会要求复核。
私有配置全文的摘要在密钥轮换后也会变化；比较器会要求人工复核，不应为消除差异而公开配置。

比较器同时输出 delivered FPS、采样显存峰值、帧龄 P95、语音结束到服务端媒体边界的
样本数/P50/P95。失败、未溯源、未记录配置、不同硬件/输入/测量参数均标记 `review_required`
并退出 2；字段匹配也只表示已记录条件相符，不证明统计显著性、浏览器时延或感知质量。

## 配置取舍与尚未验证项

- TTS 队列按块而非按字节计数，不限制模型内部状态、每轮累积特征或全部连接总内存。
  `VOXCPM2_PCM_QUEUE_CHUNKS=8`、`VOXCPM2_PCM_STALL_TIMEOUT_SEC=5`、
  `VOXCPM2_SEND_TIMEOUT_SEC=5` 是默认值；过慢请求明确失败，健康请求保持 PCM 样本。
- LLM 64 个回答前分片上限针对已有关闭 thinking 的低延迟路径。合法但长时间只输出 reasoning
  的候选也可能失败；这是有意的可用性/资源边界，不适合作为无限 reasoning 服务。
- 日志可丢弃且有计数；关键控制拥塞、编码投递超时、浏览器缓冲溢出会结束该连接。
  不随意丢弃 fMP4 字节或 PCM 来继续同一条受损时间线。重新连接会获得新的初始化段。
- 已检查 ASR generation fence/cache 替换、现有 EMA 折叠、CFG 零系数裁剪、音频历史界限和
  渲染 stride。未改变循环模型步骤、采样步数、精度或画质参数，也未把已有优化归为本次贡献。
- ASR 的取消无法停止正在运行的 `to_thread` 调用；同模型是否允许重叠调用仍需目标依赖下的
  复现。没有凭这一疑点宣称 FunASR 不安全或加入未验证的串行化性能改动。
- 真实 GPU 首包/吞吐/显存、长时间稳定性、唇音同步、浏览器播放和公网链路仍待实机验收。
  当前机器仅有 8GB RTX 4060，CPU 回归不能证明完整单卡 4090 配置在该设备上可运行。
