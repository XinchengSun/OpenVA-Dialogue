# 按 RBA 方法继续优化渲染器

基线：`a7d47a9e555f60dc7f53f93b5709236a0d924299`。
参考 [ResearchBoost-Agent df4378d](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/df4378df1949761d2d5f051cc2d3f5e01e3cfabb)
的分层 profiling、冻结验收、反例自测及独立进程对照方法。本轮完成两项可开关的渲染
显存优化，**没有测出达到门槛的稳定加速，也没有完成整机 12 GiB 验收**。

## 两项改动与依据

1. **固定谱归一化权重缓存**。完整 FaceGenerator 有 18 个 legacy SpectralNorm 层，
   两帧 profiling 确认执行了 36 次矩阵向量乘和 36 次点积。eval 且权重及 `u/v` 不变时，
   每次归一化结果相同。现在保留原 GPU 算法计算出的有效权重，将原始权重及 `u/v`
   移到 CPU，后续复用有效权重。参数/缓冲变化、checkpoint 重载、dtype/device 往返会
   使缓存失效；训练/有梯度计算回到原 hook。复测两帧的上述 `mv/dot` 调用均为 0。
2. **缩短翘曲中间张量生命周期**。逐行清点发现，进入高分辨率 `up_1` 时，局部变量仍
   持有已无后续用途的 `z_emb/condition/z/deltas/warping_field`、8 MiB 的翘曲特征体积
   及遮挡图。现在分别在最后一个消费者之后删除这些局部引用，保留输入、所需的 `seg`
   和外观特征。没有改变任何计算公式或帧数。

实现位于 [`inference_memory.py`](../inference_memory.py)、
[`FaceGenerator`](../tools/visualization_0416/utils/model_0506/model/head_animation/LIA_3d/face_generator.py)，
由 [`app.py`](../app.py) 在原 checkpoint 加载完成后接入。
两个开关默认关闭，开启后才生效：

```bash
export DYSTREAM_CACHE_SPECTRAL_NORM=1
export DYSTREAM_RELEASE_RENDER_INTERMEDIATES=1
# 使用原有 scripts/flashav2av 启动入口；也可将两项写入该实例的私有环境文件。
```

关闭相应开关并重启该测试实例即可回到基线路径。没有修改共享服务或私有运行配置。
默认关闭遵循 RBA 的逐项开关方法，并为后续真实头像及目标服务器 A/B 保留原路径。

## 真实训练权重的本机验证

2026-10-28，Windows，RTX 4060 Laptop 8188 MiB，Python 3.12.3，
PyTorch 2.8.0+cu126，cuDNN 9.10.2。保持原 FP32 权重及 math modes：
matmul TF32=false、cuDNN TF32=true、cuDNN benchmark=false；没有新开低精度选项。

从[固定上游模型版本](https://huggingface.co/robinwitch/DyStream/tree/9ad5b1d3b0ef7aece4b9855d972c1f819f04dfbb)
下载 LIA checkpoint，1,579,672,001 字节，核对 LFS SHA-256：
`a9f06f55912dfa12ea18d77315f1c1675c5f67669097db8f155b4a4d75f9d31d`。
该校验值也已补进 [`weights-manifest.json`](../weights-manifest.json)，权重仅保存在 D 盘，未提交 Git。
通过 `torch.load(weights_only=True, mmap=True)` 读取，没有放开任意 pickle 加载。

测试实际加载 FaceEncoder、MotionEncoder、FlowEstimator、FaceGenerator 的训练权重，
使用合成 512×512 RGB 参考图，提取外观特征和动作锚点，再生成四组轻微变化的动作输入。
一次性编码后将编码器移回 CPU，常驻 GPU 的生成路径为 flow 和 generator。
每个变体使用独立进程，相同权重、输入及配置。

| 配置 | 帧返回时 allocated MiB | 单帧 allocated 峰值 MiB | reserved MiB |
| --- | ---: | ---: | ---: |
| 原版 | 458.52 | 807.10 | 1044 |
| 仅谱归一化缓存 | 340.72 | 690.68 | 778 |
| 仅缩短临时张量生命周期 | 456.64 | 795.64 | 1026 |
| 两项同时开启 | 340.72 | **679.35** | 778 |

**组合开启的峰值实测减少 127.75 MiB**，是同一渲染子管线的实际对照，未将两项数据相加。
两个独立原版进程的峰值分别为 808.10 / 807.10 MiB，说明有约 1 MiB 工作区波动；
上表使用较低原版值计算节省。`reserved` 包括分配器缓存，不等于永久张量载荷。
谱归一化缓存将 116.29 MiB 原始状态移到 CPU，两个观测的安装成本为 34.84 / 36.26 ms，
属于一次性初始化成本，不放进稳态计时。

两个独立原版进程、两项各自开启及组合开启，均完成 4 帧浮点输出和 116 条中间层记录的
逐位对照，误差为 0；原 checkpoint 键和值保持一致。输出标准差约 0.543，并非全零结果。
开启/关闭遮挡分支的完整渲染器也通过对照。

上游 checkpoint 缺少当前类中未被 forward 调用的 `warp_layer.weight`。生产入口原本
使用 `strict=False` 保留初始化值；本测试仅允许这一个已确认的缺失键，其余严格匹配，
并将该初始化值一起纳入所有变体的 state hash。首次严格加载失败日志保留，未忽略
其他缺失参数，也未修改数值比较规则来接纳候选。

## 耗时：未通过收益门槛

先冻结 judge 和反例检查，原版在两个进程中复现后再改生产代码。计时单独关闭 profiler，
固定种子，20 帧预热，每进程 3 个窗口，每窗口 35 帧；仅在窗口边界同步。
按“原版→缓存→原版→生命周期→原版→组合→原版”运行七个进程，各项只改变一个开关组合。
judge 哈希在运行前校验，环境、权重 state hash、输入形状、math modes 与源代码指纹一致。

这组计时使用随机 FP32 权重及生产形状合成特征，**不是训练权重下的整套对话延迟**：

| 变体 | 两侧原版窗口中位数 ms/帧 | 候选中位数 ms/帧 | 结论 |
| --- | ---: | ---: | --- |
| 谱归一化缓存 | 108.085 | 108.972 | 未证实加速 |
| 缩短临时生命周期 | 107.902 | 108.282 | 未证实加速 |
| 两项组合 | 107.809 | 106.084 | 约 1.6%，未过 2% 门槛 |

收益门槛为 `max(1.02, 1 + 2 × 两侧原版窗口相对范围)`，并要求候选每个窗口都通过。
三项均不满足，因此只保留显存优化结论。谱归一化所占耗时较小，原 profile 仍以
cuDNN 卷积为主；不能从消除少量重复计算推导整体加速。本机是共享显示卡，也不满足
正式 RBA 独占 GPU 环境要求，不将本轮称作官方 RBA 完整任务验收。

## 正确性、证据与复现

- 验收反例能拒绝 1 ULP 改动、丢帧、NaN、dtype 改动、正负零位差、输入配置变化及中间层变化。
- 覆盖参数及 `u/v` 修改、参数替换、重载、dtype/device 往返、训练梯度与 power iteration。
- 回归 136 项：122 通过、14 因缺 Pipecat 跳过；权重清单另 6 项通过，共 128 通过。
  编译与 diff 检查通过；远端 CI 未运行。
- 原始结果、输出数组、profile trace、失败日志、固定协议及哈希在父目录的
  `evidence/rba_renderer_round3`；[验证摘要](rba_renderer_validation_20261028.json)记录其 SHA-256。
  Windows profiler 写中文 trace 路径失败后，使用 D 盘 ASCII 临时文件导出并验证最终 trace 存在。

```bash
python -m unittest tests.test_spectral_norm_cache tests.test_renderer_memory_integration tests.test_renderer_reuse_judge -v
python tests/bench_renderer_reuse.py check --output-dir /data/reuse/base
python tests/bench_renderer_reuse.py check --variant both --reference /data/reuse/base/outputs.npy --output-dir /data/reuse/both
python tests/bench_renderer_reuse.py time --variant baseline --output-dir /data/reuse/time-base
python tests/bench_renderer_reuse.py time --variant both --output-dir /data/reuse/time-both
# 对正式计时补上后置原版，独立进程运行；每个输出目录必须是新目录。

python tests/bench_renderer_checkpoint.py --checkpoint /data/weights/renderer.ckpt \
  --expected-sha256 a9f06f55912dfa12ea18d77315f1c1675c5f67669097db8f155b4a4d75f9d31d \
  --output-dir /data/reuse/trained-base
python tests/bench_renderer_checkpoint.py --checkpoint /data/weights/renderer.ckpt \
  --expected-sha256 a9f06f55912dfa12ea18d77315f1c1675c5f67669097db8f155b4a4d75f9d31d \
  --variant both --reference /data/reuse/trained-base/outputs.npy --output-dir /data/reuse/trained-both
```

## TODO

1. 在目标服务用获授权的真实头像、相同权重和输入做开关 A/B，检查完整输出、切换、打断及长期稳定性。
2. 同时加载 DyStream、VoxCPM2 和渲染进程，分别测冷启动与稳态的整卡峰值、端到端延迟。
3. 在实际 12 GiB 显卡验收。当前只证明上述渲染路径的数值一致及显存收益；真实人物画质、
   音色和模型任务效果尚未做整套验收，不能用本表推算新整机显存。
