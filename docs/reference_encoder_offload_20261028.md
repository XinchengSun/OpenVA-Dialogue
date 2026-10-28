# 头像编码器用后卸载

本轮基线为 `5f19575152afd68d417ad51c0f0bd13769bf4110`。目标是在保持原有 FP32
计算、头像特征、模型参数和生成配置的条件下，缩短初始化模块在 GPU 上的驻留时间。

## 改动

`render_worker` 每次在独立 spawn 进程中初始化一个固定头像：

1. `process_image` 提取头像动作后，立即将 `motion_encoder` 移回 CPU。
2. 使用原设备、原精度的 `face_encoder` 提取外观特征，关闭这次不参与训练的梯度记录。
3. 保留 GPU 外观特征，删除源头像的 GPU 临时张量，将 `face_encoder` 移回 CPU。
4. 释放 CUDA 分配器中空闲的缓存，再进行 LIA 预热和后续渲染。

持续渲染只使用缓存的外观特征和动作锚点，以及 `flow_estimator`、`face_generator`；
这两个生成模块保留在原设备。编码器权重仍以原精度保存在 CPU，没有删层、量化、
修改分辨率、帧率、去噪步数或音频历史。CPU 将额外持有对应编码器的约 90.85 MiB 张量。
头像切换继续沿用原有的工作进程重启流程，新进程重新提取特征。

## 容量清点

按仓库实际 512×512 配置，实例化生产编码器类并清点 FP32 参数及缓冲：

| 模块 | 参数个数 | 参数及缓冲字节 |
| --- | ---: | ---: |
| FaceEncoder | 2,148,480 | 8,604,800 |
| MotionEncoder | 21,665,312 | 86,662,144 |
| 合计 | 23,813,792 | 95,266,944，约 90.85 MiB |

源头像 GPU 张量另有 `1×3×512×512×4 = 3,145,728` 字节，即 3 MiB。
本表是张量大小清点；实际 CUDA 活跃分配、缓存保留量和整卡占用采用不同口径，不能混写。

## 验证方法

数值和显存记录见 [reference_encoder_offload_validation_20261028.json](reference_encoder_offload_validation_20261028.json)。

本机 RTX 4060 Laptop、驱动 566.24、Python 3.12.3、Torch 2.8.0+cu126 的实际 CUDA 组件测量：

| 指标 | 结果 | 范围 |
| --- | ---: | --- |
| 两个编码器活跃分配释放量 | 95,307,264 字节 / **90.89 MiB** | 含 CUDA 分配粒度；保持原 FP32 |
| 源头像临时张量 | 3,145,728 字节 / **3 MiB** | 单独删除，与编码器释放量不重复 |
| 外观特征提取阶段峰值 | **934.00 → 331.24 MiB** | 提前卸载动作编码器 + no_grad，减少 602.76 MiB |
| 缓存外观特征和动作锚点差异 | **最大绝对误差 0** | 留在 GPU 上；与原梯度开启路径对照 |

以上使用真实生产编码器结构、实际 512×512 配置、固定随机 FP32 权重。
没有加载训练 checkpoint、完整动作/渲染/TTS 模型，也没有测量整卡对话峰值。
阶段峰值已经包含该阶段发生的常驻量变化，不能与 93.89 MiB 再相加。

CUDA 回归 7 项全部通过，包括 CPU→GPU 权重往返后的实际编码器输出逐元素一致；
CPU 回归 29 项中 26 通过、3 项 CUDA 检查跳过。编译与 diff 检查通过，远端 CI 尚未执行。
CUDA 包在独立 D 盘虚拟环境安装，并验证官方 wheel SHA-256，原 CPU 环境未替换。

- 运行完整生产 `render_worker` 控制流程的 CPU 回归：在编码器卸载后禁止再次调用它们，
  后续两段共 8 帧仍与未卸载路径逐字节一致。该测试使用小型数值组件替代完整人物网络。
- 真实生产编码器类使用实际配置与随机 FP32 权重，执行 CUDA 编码、卸载和重新加载对照；
  不下载训练权重或私人人物素材，不将组件验证标为整模型画质验收。
- 同一个编码器测量进程对比原先开启梯度的外观提取与提前卸载动作编码器后的 no_grad 提取，
  记录阶段峰值。该阶段峰值差不等于完整对话管线的峰值差。
- 生产卸载函数记录 `memory_allocated` 的活跃张量释放量，以及 `empty_cache` 后的
  `memory_reserved` 变化。后者包含临时工作区，后续推理可能重新申请，不能全部记作永久节省。

```bash
python -m unittest tests.test_reference_encoder_offload tests.test_render_stride_and_d2h -v
python tests/bench_reference_encoder_offload.py --device cpu --output inventory.json
python tests/bench_reference_encoder_offload.py --device cuda --output offload_cuda.json
```

## 12 GiB 目标

历史单卡完整对话记录约 13.01 GiB，尚无当前修改后的完整模型测量。
两个编码器的实际活跃分配加源图像约 93.89 MiB，单靠这部分不足以填平约 1 GiB 的差距。
初始化阶段还可因关闭梯度和提前释放缓存降低峰值，但实际整条管线能降低多少，必须在
同硬件、同权重、同输入的完整对照中记录；当前不宣称已经适配 12 GiB 显卡。

分配器口径依据：[PyTorch 2.8 CUDA memory management](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-memory-management)。
