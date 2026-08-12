# FlashAV2AV

FlashAV2AV 是一套低延迟、可定制的实时音视频对话系统。浏览器持续上传麦克风音频；对话流水线完成语音理解、回复生成和流式语音合成；DyStream 在同一条 recurrent motion 流中生成 Listener 与 Speaker 状态；服务端最后输出 H.264 + AAC fMP4，由浏览器通过 MSE 连续播放。

项目保留两种对话后端：

- `custom_cascade`：VAD → Paraformer ASR → 流式 LLM → VoxCPM2 → DyStream。适合定制音色和接入实时工具。
- `native_s2s`：原生语音到语音模型 → DyStream。组件更少，适合快速部署或作为回退路径。

Pipecat 负责对话组件编排、回合和打断事件；`server_mse.py` 负责 AV2AV 状态、连续 motion/render、媒体边界和浏览器 WebSocket。发布入口不会修改这些已经验证的推理路径。

## 系统要求

- Linux
- NVIDIA GPU 与可用的 CUDA 环境
- Python 3.11
- `ffmpeg`、`ffprobe`、`curl`、`flock` 和 `nvidia-smi`
- DyStream motion、renderer 和 wav2vec2 权重
- 使用 `custom_cascade` 时还需要 VoxCPM2 模型及独立运行环境
- 对话模型所需的 API 凭据

DyStream 默认使用两个不同的逻辑 GPU。VoxCPM2 可以按部署情况使用另一张 GPU；具体映射全部通过私有环境文件配置。

## 用户入口

所有生命周期操作都从同一个 CLI 进入：

```bash
bash scripts/flashav2av setup
bash scripts/flashav2av configure --prompt-wav /absolute/path/to/reference.wav --tts-gpu 4
bash scripts/flashav2av start
bash scripts/flashav2av status
bash scripts/flashav2av restart
bash scripts/flashav2av stop
```

`start/status/restart/stop` 内部复用已经验证的 `scripts/run_demo.sh`，因此仍保留进程归属检查、组件健康检查和可解码媒体 smoke。

### 1. 准备环境

```bash
git clone <flashav2av-repository-url> FlashAV2AV
cd FlashAV2AV
bash scripts/flashav2av setup
```

`setup` 会：

1. 检查主机基础命令；
2. 在不存在时从 `.env.example` 创建权限为 `0600` 的 `.env`；
3. 调用项目已有的 Pipecat、Paraformer 和 VoxCPM2 安装器；
4. 按 `weights-manifest.json` 从上游模型仓下载并校验 DyStream motion、LIA renderer 和 Wav2Vec2 权重。

`setup` 不会猜测 API Key、克隆声音或头像。完成后编辑私有 `.env`，设置
`DYSTREAM_REF_IMAGE=/absolute/path/to/authorized-avatar.png` 和 LLM 凭据，随后运行：

```bash
bash scripts/flashav2av configure \
  --source-env /absolute/path/to/private-base.env \
  --prompt-wav /absolute/path/to/authorized-reference.wav \
  --tts-gpu 4 \
  --dystream-gpus 0,1
bash scripts/flashav2av start
```

`configure` 只写入 `$FLASHAV2AV_DATA_ROOT/config/` 下权限为 `0600` 的私有 env；
之后统一 CLI 会自动使用它。

安装会下载较大的 Python 包和 VoxCPM2 模型。只检查当前机器而不安装：

```bash
bash scripts/flashav2av setup --check-only
```

默认数据目录是 `${XDG_DATA_HOME:-$HOME/.local/share}/flashav2av`。如需放到数据盘：

```bash
export FLASHAV2AV_DATA_ROOT=/path/to/flashav2av-data
bash scripts/flashav2av setup
```

### 2. 模型权重

大模型文件不提交进 Git。`setup` 会从 `weights-manifest.json` 记录的上游地址下载到：

```text
checkpoints/last.ckpt
tools/pretrained_model/epoch=0-step=312000.ckpt
tools/hf_models/wav2vec2-base-960h/pytorch_model.bin
```

也可以单独执行下载器；`--include-optional` 会额外下载 SenseVoice 与 Fish S2 Pro 候选：

```bash
python scripts/setup_weights.py download
python scripts/setup_weights.py --include-optional download
python scripts/setup_weights.py verify --deep
```

文件可以是实际文件，也可以是指向本机模型目录的软链接。不要提交本机绝对路径软链接。完整清单见 [docs/weights.md](docs/weights.md)。

### 3. 配置私有环境

编辑 `.env`，至少配置所选对话后端需要的凭据和模型。不要提交、粘贴或在日志中输出 `.env`。

常用配置：

```dotenv
PIPECAT_MSE_DIALOG_MODE=custom_cascade
PIPECAT_LLM_API_KEY=
PIPECAT_LLM_BASE_URL=
PIPECAT_LLM_MODEL=
VOXCPM2_ENV_FILE=/path/to/private/voxcpm2.env

CUDA_VISIBLE_DEVICES=0,1
MOTION_GPU=0
RENDER_GPU=1
```

已有独立运行配置时，不必复制到仓库：

```bash
ENV_FILE=/path/to/private/flashav2av.env \
  bash scripts/flashav2av start
```

### 4. 启动与访问

```bash
bash scripts/flashav2av start
```

看到 `DEMO_READY` 表示模型、对话组件和媒体出口均已通过启动检查。默认服务仅绑定 loopback。远程机器可使用 SSH 转发：

```bash
ssh -N -L 6008:127.0.0.1:7860 <user>@<server>
```

然后访问：

```text
http://127.0.0.1:6008/
```

浏览器只保留一个演示标签页，点击“开始对话”后允许麦克风权限。

## 定制头像和声音

在本地安全入口访问：

```text
http://127.0.0.1:6008/customize
```

上传正脸图片和单人参考音频；视频会先提取音轨。定制过程生成的头像、音频、latent、缓存和回滚文件属于私有运行资产，不应提交到 Git。完整说明见 [README_CUSTOMIZATION.md](README_CUSTOMIZATION.md)。

## 运行状态与验收

```bash
bash scripts/flashav2av status
```

正式演示至少确认：

- motion worker 与 render worker 均 alive；
- 所选对话后端 ready；
- 媒体可被识别为 H.264 + AAC；
- Listener、Speaker 和打断共享连续时间轴；
- 连续完成至少三轮问答；
- 打断后能够恢复 Listener，并继续下一轮；
- 服务端持续输出目标帧率，浏览器缓冲不耗尽。

## 目录

```text
pipecat_dystream/   对话组件、ASR、LLM、TTS 和实时搜索适配
voice_service/      VoxCPM2 与兼容语音服务
model/              DyStream motion 模型代码
tools/              renderer 与预处理代码
static/             演示和定制前端
scripts/            安装、生命周期、检查和探针
tests/              逻辑、协议和媒体回归测试
```

## 数据与安全边界

以下内容不得提交：

- `.env`、API key、访问令牌；
- 用户头像、参考声音、视频和克隆缓存；
- 模型权重、虚拟环境和下载缓存；
- `logs/`、PID、抓帧、探针结果和运行快照；
- 临时公网地址和带令牌的访问 URL。

公开部署应放在 HTTPS/WSS 反向代理之后，并单独配置身份验证。头像、声音和视频必须取得相应授权。

## 排障

查看受控状态：

```bash
bash scripts/flashav2av status
```

查看服务日志：

```bash
tail -f logs/pipecat_mse.log
```

常见启动失败原因：

- `.env` 未配置所选后端需要的凭据；
- 三个 DyStream 权重缺失；
- `CUDA_VISIBLE_DEVICES` 没有暴露两个不同 GPU；
- VoxCPM2 私有 env 或模型目录不存在；
- `ffmpeg` 缺少 `libx264` 或 AAC 编码器；
- 目标端口已被不属于本项目的进程占用。

生命周期脚本只管理它能确认属于当前仓库的进程，不会清理未知 GPU 任务或占用端口的其他服务。
