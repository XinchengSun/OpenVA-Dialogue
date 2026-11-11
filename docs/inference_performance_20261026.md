# 推理热路径优化：计算依据、对照与边界

本轮基线为 `40153f85414c505238c6e948ce0aa45e5be076c6`，独立于上一轮队列和生命周期修复。
改动针对正常推理时的重复计算、设备标量读取和媒体复制。未改权重、模型选择、提示词、
采样参数、去噪步数、音频历史、分辨率或帧率；没有实现投机解码。

## 已接入的路径

| 路径 | 原行为 → 当前行为 | 可执行依据 |
| --- | --- | --- |
| LIA `Direction` | 每帧对同一 512×64 权重 QR 分解 → 固定权重首次计算后复用 | `tests.test_render_constant_cache.DirectionCacheTests`；保留原 QR、diag_embed、matmul、sum 顺序 |
| LIA 四层 `WSConv3d` | 每帧重复标准化同一卷积权重 → 推理时复用标准化结果 | `StandardizedConvolutionCacheTests`；均值、标准差、除法及卷积顺序不变 |
| LIA 恒等网格 | 每帧把 16×64×64×3 FP32 网格搬到 GPU → 复用同设备副本 | `IdentityGridCacheTests`；保留原 FP32 语义与 grid_sample 参数 |
| 运动模型时间步 | `torch.full(..., timestep_tensor)` 隐式读取标量 → `to(long).expand(bs)` | `tests.test_motion_timestep_batch`；实际 WanTimeEmbedding 数值对照，10 次构造的 `_local_scalar_dense` 从 10 次降到 0 |
| 媒体打包 | 已为 uint8 的片段再次整块复制 → 复用连续数组 | `tests.test_media_copy_performance`；各类输入输出字节一致，异步队列仍复制拥有的数据 |

渲染缓存只在 `eval + no_grad/inference_mode` 使用。训练及开启梯度时执行原计算。
权重更新、替换、load_state_dict、类型和设备切换会失效；`_apply` 主动清空，覆盖
`half().float()` 中间没有 forward 的往返和 CPU offload。缓存不进入 checkpoint。

当前 FP32 渲染配置额外常驻缓存张量为 **19,718,144 字节，约 18.80 MiB**：
四层标准化卷积权重 18,800,640 字节、QR 基 131,072 字节、设备网格 786,432 字节。
这项用内存换重复计算；没有宣称省显存。数字不含 CUDA 分配器、图池等开销。

## 已实现、等待 GPU 验收的条件投影复用

一帧内 GPT 输出不随 Euler 时间步变化。`gpt_proj` 和六个 `adaLN_modulation2`
只依赖这个固定输出，因此可各计算一次。10 步时，这七个线性模块由各 10 次降到各 1 次；
其余依赖噪声和时间步的计算照常执行。每帧刷新条件，不能跨帧复用。

CPU 实际 DiffusionHead 多步输出、训练输出/梯度对照通过。CUDA 实现使用条件图加逐步图，
两图联动尚未上卡，所以 **`DYSTREAM_PRECOMPUTE_DIFFUSION_CONDITION=0` 默认关闭**。
设为 1 才启用；开关、批大小、设备、dtype 或 CFG 签名变化会重建图。
测试包含不同 CFG 批大小和逐帧更新；不能把“实现了候选”写成“默认已经获得该项收益”。

## 本机 CPU 组件实测

原始样本、环境、参数、预热次数及源码摘要见
[performance_validation_20261026.json](performance_validation_20261026.json)。
环境为 Windows、Python 3.12.3、Torch 2.10.0+cpu、NumPy 2.1.2；Torch 组件单线程，
固定随机权重和输入，预热后交替执行原算法与候选。没有下载或加载真实模型权重。

| 测量边界 | 基线中位数 | 当前中位数 | 说明 |
| --- | ---: | ---: | --- |
| Direction(512,64)，单个方向投影 | 0.275730 ms | 0.053249 ms | 16 个输入逐元素一致；非完整渲染 |
| 首个 WSConv3d，输入 1×512×8×8×8 | 86.3395 ms | 28.9225 ms | 实际首层形状，3 个输入逐元素一致；非完整渲染 |
| DiffusionHead，6 层、CFG batch 4、10 步 | 253.5858 ms | 218.6869 ms | 条件复用候选，减少约 13.76%；多步最大误差 0 |
| 4 帧 512×512，stride 2 的 CPU 媒体打包 | 1.460400 ms | 0.526450 ms | 实际生产方法，编码器入队处替换为接收端；不包含编码/网络 |

首个卷积的 8³ 输入来自 4³ 体积第一次上采样；16×64×64 是后续层体积。
DiffusionHead 的 hidden=768 来自真实构造链：`app.py` 只把 cfg 传给
`utils.instantiate_motion_gen`，该函数没有展开 cfg；构造函数使用 hidden_size=768 默认值，
而非 YAML 中未应用的 512 字段。这里没有修改这一既有行为。

媒体测试每片段、每客户端少复制 3,145,728 字节；帧堆叠与音频准备还去掉了同 dtype 的冗余复制。
CPU 组件结果不能加总或换算为 GPU、首音、整段对话延迟。计时有波动，全部原始样本一并保留。

## 验证与复现

本轮合并验证：**123 项中 115 通过、8 跳过、0 失败**。跳过的是 4 项 CUDA 检查和
4 项 Windows 无法执行的 Linux 检查；编译和 diff 检查通过。CI 已接入新增测试，未实际运行远端 CI。
CPU 测试执行生产 Torch 类及真实 timm / diffusers 模块；为避开模型下载，部分测试从源码提取类。
它们不能替代整模型加载、实际音视频质量或真实浏览器播放验收。

```bash
python -m unittest tests.test_render_constant_cache tests.test_render_stride_and_d2h -v
python -m unittest tests.test_diffusion_condition_reuse tests.test_motion_timestep_batch tests.test_cfg_pruning -v
python -m unittest tests.test_media_copy_performance tests.test_media_backpressure tests.test_media_socket_integration -v
python scripts/benchmark_render_constants.py --component direction
python scripts/benchmark_render_constants.py --component wsconv3d --warmup 3
python -m tests.bench_diffusion_condition
python tests/bench_media_copies.py --baseline 40153f85414c505238c6e948ce0aa45e5be076c6
```

依赖使用项目已有版本 `timm==1.0.24`、`diffusers==0.36.0`。CI 的 CPU Torch 2.8.0
与本机 2.10.0 不同，不能把本机通过等同于目标 CUDA 2.8.0 环境通过。

GPU 到位后先执行上述 CUDA 条件测试，再在原基线和候选间固定同一权重、头像、音频、seed、
配置及测量工具做实际输出对照。首先比较默认启用项，之后单独开启条件复用；采集首次与稳态耗时、
峰值显存、生成帧/声音质量、首音及实际播放延迟。发现数值或画质异常时回到基线，不通过降画质补性能。

接口依据：[NumPy astype 的默认复制行为](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)、
[ascontiguousarray 的复用语义](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html)、
[PyTorch 2.8 Module._apply 实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/module.py)。
优化依据来自明确的数据依赖和原算法对照，论文中的加速数字未用于本项目收益估计。
