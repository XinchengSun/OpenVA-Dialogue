# 推理显存核查与第二轮优化

基线：`c6220ee5c618af2af95117d70d625e818c27dcf7`，已包含上一轮头像编码器卸载。
本轮新增动作模块选择性加载、位置表共享、SPADE 临时张量复用、官方 VoxCPM2 AudioVAE
参考编码器卸载及解码器固定权重复用。没有修改分辨率、精度、模型参数值、去噪步数、
音频历史或上下文长度。**尚未验证完整管线可以在 12 GiB 卡上运行。**

## 已确认的占用与改动

| 占用来源 | 源码依据与处理 | 生效范围 |
| --- | --- | --- |
| 动作网络不参与 forward 的层 | `Audio2FaceGPTBlock.forward` 不调用 `cross_attn_rope/norm2`；`DiffusionHead.forward` 不调用 `past_proj/anchor_proj`。保留 checkpoint 中的原权重，直接留在 CPU，避免先在 CUDA 分配再卸载。实际配置张量载荷共 147.22 MiB。 | 单卡配置、CPU 加载并折叠 EMA 后的固定 eval 模型；保留有活动 EMA 的旧路径 |
| 多层完全相同的位置表 | 活跃 RoPE/PE 表逐元素比较相等后共享存储，在 CPU 上建立别名并保持 GPU 传输别名；共去重 102 MiB。表长度不变，不合并数值不同的表。 | 同上，CUDA Graph 捕获之前 |
| CUDA Graph 与碎片 | 测量区分默认池、Graph 私有池和 inactive split；固定 94 帧上下文的 GPT 图实测没有随生成帧数不断新增。采用选择性加载避免碎片。 | 可选诊断；没有删除图或改为低精度 |
| 渲染器高分辨率 SPADE 中间结果 | 原实现同时持有 `normalized`、上采样条件、activation、gamma/beta 和算术临时结果。提前释放局部引用，并在 eval 且关闭梯度时复用内部输出存储，保持加、乘、加的计算顺序。 | LIA SPADE 推理；训练及有梯度 eval 保留原算术路径 |
| 参考音频编码器 | 官方 AudioVAE `encode` 使用 encoder；`decode/streaming_decode` 只使用 decoder。缓存参考特征后将 encoder 参数、buffer 和 legacy weight_norm 的普通 Tensor 属性都移回 CPU；再次 encode 自动重载，用完再卸载。 | 官方 VoxCPM2 后端；独立 encoder/decoder、eval、CUDA |
| 音频解码器双份权重 | legacy weight_norm 同时保留 `weight_g/weight_v` 和展开后的有效 `weight`。固定推理复用相同有效权重，原参数保留在 CPU，checkpoint 键和值保持兼容。重载/修改参数、切换 dtype/device 会更新缓存；训练/梯度计算恢复原 hook。 | 官方后端 eval decoder 的 legacy weight_norm；无版本计数、共享参数或已有梯度时跳过 |

实现集中在 [`inference_memory.py`](../inference_memory.py)，接入点为
[`app.py`](../app.py)、[渲染/动作 worker](../dual_gpu_mic_realtime_ui_v8_playbuffer.py)
及[官方语音后端](../voice_service/official_voxcpm2_backend.py)。上述适用路径默认生效；
显存诊断默认关闭。实验性 diffusion condition 复用仍默认关闭。

## 本机真实 CUDA 组件测量

2026-10-28，Windows，RTX 4060 Laptop 8188 MiB，驱动 566.24，PyTorch 2.8.0+cu126。
使用真实生产组件类、随机 FP32 权重和固定种子合成输入；没有训练权重、真实头像、完整
VoxCPM2 LM 或整套并发服务。MiB 为 2²⁰ 字节。下表都是进程内 PyTorch 分配器统计。

| 范围与口径 | 修改前 MiB | 修改后 MiB | 差值 MiB |
| --- | ---: | ---: | ---: |
| 动作核心，稳态 allocated | 1161.33 | 904.11 | 257.22 |
| 动作核心，稳态 reserved | 1286 | 1066 | 220 |
| LIA 512×512 单帧 allocated 峰值 | 959.83 | 799.20 | 160.62 |
| LIA 预热后 reserved | 1308 | 1136 | 172 |
| AudioVAE，流式解码后 allocated | 718.33 | 175.58 | 542.75 |
| AudioVAE，流式解码阶段 allocated 峰值 | 864.65 | 198.02 | 666.63 |

AudioVAE 其中参考编码器卸载释放 369.36 MiB allocated；解码器随后再减少约 173.39 MiB。
最终记录 reserved 为 218 MiB，参考音频再次编码后仍为 218 MiB；再次编码会临时重载
encoder，该阶段 allocated 峰值为 678.63 MiB，不等于常规语音解码峰值。
这些数值依赖当前 FP32 架构及工作负载，不替代目标服务配置的测量。

动作模块也保留了失败路线的证据：普通 CUDA 加载后再卸载，allocated 虽降约 257 MiB，
reserved 仅从 1286 降到 1266 MiB；inactive split 从约 104.67 增到 341.77 MiB。
改为 CPU 上先去重、选择性传输后，reserved 才降到 1066 MiB。

**以上各组件的 allocated/reserved/峰值不能直接相加，不能从历史 13.008 GiB 中直接相减。**
它们的生命周期、上下文、工作区和分配器不同；历史记录是整卡采样，当前是独立组件测量。

## 输出与验证边界

- 动作：实际 12 层、hidden=768、CFG batch=4、sequence=94，第一次 GPT Graph 输出加
  12 次变化输入 replay，共 13 组输出逐元素一致，最大绝对误差 0。该基准加载 diffusion head
  权重但不运行完整去噪循环；回归另验证 diffusion head 前后输出一致、卸载层未被调用。
- 渲染：3 帧预热、1 帧测量，完整 LIA 输出 `(1,3,512,512)` 的浮点数组逐元素一致，
  最大绝对误差 0；输入未被修改，训练/有梯度 eval 的输出及梯度一致。
- 语音：官方 AudioVAE v2 默认完整结构，2 个 `(1,64,4)` latent patch 的流式解码波形，
  以及参考编码器自动重载前后的特征，最大绝对误差均为 0。覆盖真实 `inference_mode`
  解码、权重重载、dtype/device 往返、梯度回退及后端启动顺序。
- 回归共 122 项：108 通过、14 因未安装 Pipecat 跳过、0 失败；源码编译、diff 检查通过。
  这不是完整训练模型的音色、口型、画质或长期稳定性验收。远端 CI 未运行。

原始记录、数组与日志在本地 `evidence/memory_audit`（位于本 checkout 的父目录）保留。
仓库中的[测量摘要](inference_memory_validation_20261028.json)包含原始证据 SHA-256、
候选源码 SHA-256、环境、分配器分项、输出比较及固定上游版本。
AudioVAE/调用链核查使用
[VoxCPM cce58a5](https://github.com/OpenBMB/VoxCPM/tree/cce58a5b59303c9bd63d12f23afe6a49a4a80c59)，
服务器实际版本仍需匹配；未将上游源码或权重复制进发布仓库。

## 复现

在满足仓库推理依赖的 CUDA 环境中，从仓库根目录运行。每次使用新的输出目录，动作和
渲染的前后版本分开进程执行，避免旧池污染；不要与其他 GPU 测量并发运行。

```bash
python tests/bench_motion_memory.py --output-dir /data/mem/motion-before
python tests/bench_motion_memory.py --place --output-dir /data/mem/motion-after
python tests/bench_renderer_memory.py --baseline --output-dir /data/mem/render-before
python tests/bench_renderer_memory.py --output-dir /data/mem/render-after

# 固定官方源文件；SHA-256 必须匹配摘要中的 source_sha256。
curl -L https://raw.githubusercontent.com/OpenBMB/VoxCPM/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/modules/audiovae/audio_vae_v2.py -o /data/mem/audio_vae_v2.py
python tests/bench_prompt_encoder_memory.py --source /data/mem/audio_vae_v2.py \
  --cache-decoder --output-dir /data/mem/vae
python -m unittest tests.test_motion_memory tests.test_spade_memory tests.test_tts_memory -v
```

语音基准还需要 `pydantic`、`einops`、`numpy`；当前记录使用的官方实现并不编译 AudioVAE。
动作基准不包括 Wav2Vec、音频投影/embedding 等外围模块。单帧/短音频组件测试不提供
长期运行容量保证。

## 完整服务的剩余核查与 TODO

1. 有效目标服务器、项目及权重路径到位后，在独立端口对基线与候选进行同输入 A/B。
   保持同头像、音色、随机种子、模型 revision、精度、分辨率、历史长度和去噪步数。
   当前本地无完整 checkpoint，已有连接未能建立，未修改或重启远端服务。
2. 在目标服务的私有环境中设置 `OPENVA_MEMORY_PROFILE_DIR=/data/mem/ledgers`，分别
   传给 motion/render/官方 TTS 服务，再启动测试实例。会在加载、预热及第 1/2/4/8…
   次阶段输出各进程 JSONL，记录参数、buffer、普通 Tensor 缓存、KV/static buffer、
   默认/Graph 池、inactive split 与未归类差额。只记录尺寸和标签，不记录输入内容。
   诊断会同步 CUDA，测延迟时应关掉该变量另跑。
3. 对仍参与每次生成的两路 Wav2Vec、motion 活跃层、flow/generator、VoxCPM2
   base/residual LM、特征网络、KV cache，以及编译/图工作区做完整账本。
   本轮没有这些训练权重和并发运行记录，不能声称已量清其真实整机占比。
   普通 Tensor 存储清点不能覆盖全部 CUDA 库/驱动分配，必须结合整卡采样。
4. 用 `scripts/benchmark_single_gpu.py` 在同服务器采样目标卡和服务所属 PID，验证冷启动、
   Listener/Speaker 切换、长句、多轮和打断期间的整卡峰值、无 OOM、输出等价与延迟。
   最后在实际 12 GiB GPU 上验收启动和持续生成；单靠分配器限额或组件节省量不算通过。

分配器口径参考：[PyTorch 2.8 CUDA memory management](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-memory-management)。
