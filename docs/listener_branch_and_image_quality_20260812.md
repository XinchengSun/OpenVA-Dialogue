# Listener 双分支边界与画质审计（2026-08-12）

## Listener 修复

正式实时路径不是 Speaker/Listener 两套整模 checkpoint。`last.ckpt` 内部包含
`audio_self` 与 `audio_other` 两套音频编码/投影分支，LIA3D checkpoint 负责把
motion 渲染为人脸。角色切换必须留在同一 recurrent stream 中完成：

- Speaker：`self=TTS PCM, other=0`；
- Listener：`self=0, other=virtual/mic`；
- 打断：保留既有 graceful Speaker tail，同时 `other=mic/virtual`。

此前实验提交 `0876e65` 加入了低 RMS 入口和 0.4 秒人为渐入，已由
`f72e94b` 精确撤回。新修复只处理正常回复最后一个不足 200 ms 的 hop：若
最后真实 Speaker 长度为 `r`，则逐样本路由为：

```text
[0:r)       self=Speaker PCM, other=0
[r:3200)    self=0,           other=连续 virtual PCM
```

它不选择低 RMS 位置、不跳转或重置 virtual cursor、不重置 `past_motion` 或
两路 `past_audio`，也不消费任意短窗的 mic。真实 barge-in 仍走原 full-hop mic
路径。远端测试覆盖 `r=1/1377/3199`，并验证 normal Listener 的旧 80 ms
virtual/mic source fade 保持不变。

真实一轮记录：最后 Speaker hop 为 258 samples，padding 为 2942 samples；日志
出现 `strict_branch_handoff=1`，随后进入 `LISTENER_VIRTUAL`，未出现 Worker reset、
queue full、FFmpeg 错误或 `self=0,other=0` padding。

## 栅格化定位

当前画质有两层上限：

1. 模型/LIA3D 的原生输出是 `512x512`，而桌面页面最大显示约 820 CSS px，
   即放大约 1.60 倍；这是主要的像素感来源。
2. 原编码为 `libx264 ultrafast + yuv420p + 1800k`，会额外损失发丝和皮肤细纹，
   但不是模型分辨率下降的根因。

使用真实、未编码的 turn frame 做单变量离线 A/B：

| 编码 | PSNR | SSIM |
|---|---:|---:|
| 1800k / 3600k buffer | 46.75 dB | 0.9902 |
| 4000k / 8000k buffer | 56.45 dB | 0.9985 |

因此候选实例提升到 4000k/8000k；动态 media smoke 解码为 H.264 + AAC、
512x512，收到约 995 KB，媒体队列和编码器无告警。该改动只减少编码损失，
不会把 512 插值成伪 1024，也不会恢复模型从未生成的细节。若要明显突破当前
上限，需要更高分辨率 renderer/checkpoint 或经过严格实时预算验证的面部增强；
不能靠 MSE、CSS 或上采样解决。

页面原先错误地让视频继承 820 px 的主容器宽度。现已将 `.avatarStage` 的
border-box 上限设为 514 CSS px；扣除左右各 1 px 边框后，视频内容区桌面最大
恰好为 512x512 CSS px，并居中显示。窄屏只允许向下缩小，不再向上放大。

离线 A/B 产物位于：
`$FLASHAV2AV_ROOT/artifacts/pixel_audit`。
