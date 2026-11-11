# VoxCPM2 本地语音桥

该服务只负责把文本变成流式 `PCM S16LE`，运行在独立 Python 3.11
环境中。主 Pipecat 进程通过 `ws://127.0.0.1:8770` 调用，因此不需要把
Nano-vLLM、FlashAttention 等依赖装进 DyStream/Pipecat 环境。

## 独立安装与国内加速下载

默认仅写入 `${FLASHAV2AV_DATA_ROOT:-$HOME/.local/share/flashav2av}`，不会激活或改写现有
Pipecat 环境。脚本先检查 Python、GPU、CUDA/NVCC 和磁盘，
只有 NVCC 精确对应 `12.1 -> cu121` 或 `12.4 -> cu124` 时才安装；其他版本会
停止并要求人工确认环境，而不会猜 wheel。PyPI 和 PyTorch wheel 默认走阿里云，
模型走 ModelScope 国内源；相同目录中的未完成模型会续传，完整模型会直接跳过。

```bash
# 只读体检，不创建目录、不安装、不下载
bash voice_service/setup_voxcpm2.sh --check-only

# 只下载/续传模型，不安装完整 TTS 运行时
bash voice_service/setup_voxcpm2.sh --download-only

# 完整安装：独立 venv + 模型
bash voice_service/setup_voxcpm2.sh
```

默认产物：

- 运行环境：`$FLASHAV2AV_DATA_ROOT/venvs/voxcpm2`
- 模型：`$FLASHAV2AV_DATA_ROOT/models/VoxCPM2`
- ModelScope/Pip 缓存：`$FLASHAV2AV_DATA_ROOT/cache/`

完整性判断要求模型目录同时具有非空 `config.json`、`audiovae.pth` 和权重；若有
safetensors index，则其中引用的每个分片都必须存在且非空。下载或安装失败只会
留下上述隔离目录供下次继续，不会停止、
重启或修改 `native_s2s` 服务。脚本不会读取、回显或使用 API Key；公开模型下载时
还会从子进程中移除常见 ModelScope/Hugging Face token 环境变量。

## 必需环境变量

```bash
export VOXCPM2_MODEL_PATH=/path/to/VoxCPM2
export VOXCPM2_PROMPT_WAV=/path/to/ref.wav
export VOXCPM2_PROMPT_TEXT_FILE=/path/to/ref.txt
# 推荐把独立进程限制到一张物理卡；进程内它会重新编号为逻辑卡 0。
export CUDA_VISIBLE_DEVICES=2
export VOXCPM2_DEVICES=0
```

生产启动脚本要求显式设置 `CUDA_VISIBLE_DEVICES`，它填写物理卡映射；完成映射后，
`VOXCPM2_DEVICES` 只填写进程内逻辑卡号，并默认使用逻辑 `0`。例如
`CUDA_VISIBLE_DEVICES=2` 与 `VOXCPM2_DEVICES=0` 表示使用物理卡 2。最终卡号和
显存占用必须以远端 `nvidia-smi` 为准，不能照抄示例；与 DyStream 共卡时也不能
使用默认 `0.90` 显存比例，必须先做余量检查和单独压测。

也可以用 `VOXCPM2_PROMPT_TEXT` 直接提供参考音频逐字稿，但不能与
`VOXCPM2_PROMPT_TEXT_FILE` 同时设置。逐字稿允许留空：此时服务启动时
只执行一次 `encode_latents`，走 ref-only 音色克隆；提供准确逐字稿时走
`add_prompt` 的 prompt continuation，音色和韵律通常更稳定。后续可先由 ASR
生成逐字稿，再人工校正。可调项包括：

- `VOXCPM2_INFERENCE_TIMESTEPS`，默认 `10`
- `VOXCPM2_GPU_MEMORY_UTILIZATION`，默认 `0.90`
- `VOXCPM2_MAX_BATCHED_TOKENS`，默认 `8192`
- `VOXCPM2_MAX_NUM_SEQS`，默认 `16`
- `VOXCPM2_BRIDGE_PORT`，默认 `8770`

## 安全启动、停止和状态检查

建议把上述配置写入独立的 `voice_service/.env.voxcpm2`（权限设为 `600`），或把
另一个文件作为第二个参数传入。生命周期脚本会 source 该文件，但关闭 xtrace，
不会打印任何变量值或 secret：

```bash
chmod 700 voice_service/run_bridge.sh
chmod 600 voice_service/.env.voxcpm2

bash voice_service/run_bridge.sh start
bash voice_service/run_bridge.sh status
bash voice_service/run_bridge.sh restart
bash voice_service/run_bridge.sh stop

# 使用显式配置文件
bash voice_service/run_bridge.sh start /secure/path/voxcpm2.env
```

脚本把 PID 和追加日志固定放在仓库的 `logs/voxcpm2_bridge.pid` 与
`logs/voxcpm2_bridge.log`。`status` 和 `start` 都会通过 localhost WebSocket
执行真实 `health` 握手，而不是仅检查端口或 PID。停止前会同时校验 PID 的
`/proc/<pid>/cwd` 和精确的 `-m voice_service.voxcpm2_server` 命令行；PID 被复用
或属于别的进程时会拒绝发送信号。`SIGTERM`/`SIGINT` 会先进入 Python 的优雅退出
路径并等待 `backend.stop()` 回收 Nano worker 和 GPU；超时也不会误用 `SIGKILL`
杀掉未确认的进程。

该脚本是**显式的 custom cascade 组件**。`native_s2s` 的启动和回退流程不会自动
启动 VoxCPM2 bridge，因此仅运行 `native_s2s` 时不会额外占用 GPU。直接执行
`python -m voice_service.voxcpm2_server` 只适合开发调试，不具备 PID 所属校验。

提供准确逐字稿时，服务启动只调用一次 `add_prompt`，后续每句合成都复用
同一个 `prompt_id`；未提供逐字稿时，启动只调用一次 `encode_latents`，后续
复用同一份参考 latent。输出采样率来自模型的
`get_model_info().output_sample_rate`，随后通过 WebSocket `start` 消息传给
Pipecat，不在桥接层硬编码。

Pipecat 侧可用 `VOXCPM2_BRIDGE_URI` 覆盖默认地址，并用
`VOXCPM2_CONNECT_TIMEOUT_SEC` 设置启动健康检查超时。TTS service 的
`start()` 会先完成 `health` 握手，桥接服务未 ready 时不会开始接收麦克风。

## 启动热身屏障

`VOXCPM2_WARMUP_TEXT` 非空时，服务会在监听 WebSocket 端口之前完整合成一次短句；
`VOXCPM2_WARMUP_TIMEOUT_SEC` 控制这一步的超时。这样 CUDA/Nano 的一次性冷启动开销
发生在服务启动阶段，`health=ok` 只会在热身完成后出现，首位用户请求不会承担冷启动延迟。
把 `VOXCPM2_WARMUP_TEXT` 设为空可显式关闭热身，但不建议用于实时演示。

真实 PCM 和首包延迟可用以下命令复测：

```bash
python scripts/probe_voxcpm2_bridge.py \
  --url ws://127.0.0.1:8770 \
  --output logs/voxcpm2_probe.wav
```

## WebSocket 协议

每条连接只承载一句合成：

1. 客户端发送 `{"type":"synthesize","request_id":"...","text":"..."}`。
2. 服务端发送带真实采样率的 JSON `start`。
3. 服务端连续发送二进制 mono PCM16 数据块。
4. 服务端发送 JSON `done`。

打断时客户端在原连接发送
`{"type":"cancel","request_id":"..."}`。桥接服务立即取消正在消费的
Nano async generator，Nano 自身的 `finally` 会把对应 `seq_id` 取消，旧音频
不会进入下一轮。

## 流式队列与退出回收

官方 `official_prompt_cache` 后端的 `VOXCPM2_PCM_QUEUE_CHUNKS` 默认 `0`，保留原有
无上限队列。设为正整数才启用待消费 PCM 的背压实验；`8` 仅是待实测的候选值，没有证据表明
它适合所有部署。启用时，信用额度覆盖队列及待执行回调，另有一个生产中的块和一个消费中的块。
此限制只在 bridge 内部成立：Pipecat 会继续把音频送入 MSE 的 `_speaker_chunks`，后者没有
等待或容量限制，因此不能将本地队列有界称为端到端积压或总内存有界。

`VOXCPM2_PCM_STALL_TIMEOUT_SEC` 默认 `5` 秒，仅在有界队列启用时生效。启用队列时必须
提供有限正数，防止停止读取的消费者一直占用模型锁；停滞超时会终止该句且不提交部分缓存。
`VOXCPM2_SEND_TIMEOUT_SEC` 默认 `0`（禁用），设为有限正数才限制每次 WebSocket 发送。
超时会关闭该连接，关闭握手还受 WebSocket 自身的 `close_timeout` 控制；该策略尚未经过真实
传输和 GPU 负载验收。`0` 默认值保留原有调度与超时行为，不能据此声称解决所有慢客户端问题。

桥接层在发送失败、请求取消时显式关闭后端生成器；Nano 包装层也同步关闭底层生成器，
避免依赖垃圾回收才取消推理。单句发送或后端失败只清理该请求；Pipecat 在一轮内复用上下文，
不能因此作废整轮。显式 `cancel` 或 `release_context` 才使上下文失效，防止旧轮次重新进入。
回归覆盖同一上下文中首句发送失败、第二句继续成功，以及显式取消后不能恢复旧上下文。
官方模型加载期间收到停止信号时，Python 线程不能强制中止 CUDA/模型加载，因此先等待线程
结束并调用可用的模型 `close()` 钩子，再完成退出。假模型测试验证生命周期所有权，不证明
实际官方模型、CUDA 缓存或显存已经释放；这些仍需真实部署验证。

CPU 回归（使用假模型和合成 PCM，不代表 GPU 性能验证）：

```bash
python -m unittest tests.test_voxcpm2_infra tests.test_official_voxcpm2_backend tests.test_voxcpm2_tts -v
```
