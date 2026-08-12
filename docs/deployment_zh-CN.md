# FlashAV2AV 部署说明

本文放置不适合塞进项目首页的服务器部署细节。公开支持的入口始终是
`scripts/flashav2av`；其他旧启动脚本属于内部实现。

## 已测试主机条件

当前 Fish S2 Pro 部署在以下 Linux 环境验证：

| 组件 | 已测试配置 |
| --- | --- |
| Python | 3.11 |
| PyTorch runtime | `2.8.0+cu128` |
| CUDA runtime | 12.8 |
| 媒体工具 | `ffmpeg`、`ffprobe` |
| DyStream | 两张不同的 NVIDIA GPU |
| Fish S2 Pro | 额外两张与 DyStream 不重叠的 GPU |
| 浏览器输出 | 512 x 512 H.264 + AAC，通过 fMP4/MSE 播放 |

GitHub CI 会检查发布文件、配置逻辑、进程生命周期和 CPU 单测，但不会运行多卡推理。

`requirements-pipecat.txt` 与 `scripts/check_pipecat_env.py` 定义受支持的实时运行时。根目录 `requirements.txt` 属于旧离线研究环境，不是服务器安装入口。

## 准备 Fish 运行时

`scripts/flashav2av setup` 会安装 Pipecat 环境、预取 Paraformer、下载固定版本的 Fish/DyStream/LIA/Wav2Vec2 权重，并创建 Git 忽略的运行软链接。它不会在空白主机上自动安装 CUDA 或构建 SGLang-Omni。

启动前，`$FLASHAV2AV_DATA_ROOT/runtime/fish-s2-pro` 下需要存在以下运行时；也可以用 `FISH_ROOT`、`SGLANG_OMNI_SRC`、`FISH_VENV`、`FISH_MODEL` 指向等价位置：

```text
runtime/fish-s2-pro/
  src/sglang-omni-ghproxy/       SGLang-Omni commit 2e607bc005c1...
  venv/bin/sgl-omni              可用的 SGLang-Omni 命令
  models/fishaudio-s2-pro/       固定版本 Fish S2 Pro 模型
  references/                    本地参考音频白名单目录
```

Fish launcher 会校验 SGLang-Omni commit，并只应用一次 [`sglang_omni_fish_cross_gpu_host_staging.patch`](../patches/sglang_omni_fish_cross_gpu_host_staging.patch)。该补丁用于在不支持 GPU P2P 的机器上通过 host shared memory 传递跨卡流式张量。

## Fish S2 Pro 主部署

先创建私有 base env：

```dotenv
PIPECAT_LLM_API_KEY=your_private_key
PIPECAT_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DYSTREAM_REF_IMAGE=/absolute/path/to/avatar.png
```

头像、参考音频和匹配逐字稿均放在 Git 之外。

```bash
export FLASHAV2AV_DATA_ROOT=/data/flashav2av

bash scripts/flashav2av setup

bash scripts/flashav2av configure \
  --source-env /absolute/path/to/private-base.env \
  --prompt-wav /absolute/path/to/reference.wav \
  --prompt-text-file /absolute/path/to/reference.txt \
  --dystream-gpus 0,1 \
  --fish-gpus 2,3 \
  --tts-backend fish_s2pro \
  --realtime-search-mode smart \
  --realtime-search-strategy turbo

bash scripts/flashav2av start
```

生成的私有 env 会以 `0600` 权限写到 `$FLASHAV2AV_DATA_ROOT/config/`。

| 地址 | 用途 |
| --- | --- |
| `127.0.0.1:8001` | Fish S2 Pro HTTP 服务 |
| `127.0.0.1:8771` | 本地流式 PCM bridge |
| `127.0.0.1:7860` | FlashAV2AV 浏览器服务 |
| 本机 `127.0.0.1:6008` | 建议使用的 SSH 转发端口 |

```bash
ssh -N -L 6008:127.0.0.1:7860 <user>@<server>
```

打开 <http://127.0.0.1:6008/>，点击“开始对话”并允许麦克风权限。

## 兼容路线

### 原生语音到语音

原生 Qwen Audio 路线保留两张 DyStream GPU，但不使用 Paraformer/LLM/Fish 级联：

```dotenv
PIPECAT_MSE_DIALOG_MODE=native_s2s
PIPECAT_S2S_API_KEY=your_private_key
DYSTREAM_REF_IMAGE=/absolute/path/to/avatar.png
CUDA_VISIBLE_DEVICES=0,1
MOTION_GPU=0
RENDER_GPU=1
```

### VoxCPM2 回退

```bash
bash scripts/flashav2av setup-voxcpm2

bash scripts/flashav2av configure \
  --source-env /absolute/path/to/private-base.env \
  --prompt-wav /absolute/path/to/reference.wav \
  --prompt-text-file /absolute/path/to/reference.txt \
  --dystream-gpus 0,1 \
  --tts-backend voxcpm2 \
  --tts-gpu 2
```

## 生命周期与验收

```bash
bash scripts/flashav2av status
bash scripts/flashav2av restart
bash scripts/flashav2av stop
bash scripts/flashav2av setup --check-only
```

`DEMO_READY` 表示所选 TTS、motion/render worker、对话链路和 H.264/AAC 解码 smoke 均通过。人工验收至少应覆盖三轮对话、一次插话打断、Speaker 返回 Listener，以及浏览器持续播放不耗尽缓冲。其他边界见[已知问题](known_issues.md)和[形象/音色定制](../README_CUSTOMIZATION.md)。
