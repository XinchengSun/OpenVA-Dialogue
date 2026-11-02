# OpenVA：CPU 迁移后的 GPU 离线排查

核验时间：2026-11-02 11:49 CST。此记录不宣称推理恢复，也不改变历史验收结果。

连接续查：用户提醒后，重新以单关键词和反斜杠 cwd 检索 SiYuan，读到迁移会话 `01a17140-674e-7861-8ca6-637ec87b5e7b` 的 2026-11-02 11:44:40 CST 记录：新机直连已经改为 `127.0.0.1:2226`。本轮随后实际通过 2226 核对主机名与 PID `498497`、健康接口 503/gpu_stopped；后续全部使用新机直连。此前部署虽在新机执行，但控制命令仍借旧机 Jupyter 跳转，漏查了这条直连更新，不能把这一路径算作连接依赖迁移完成。

迁移窗口同日 11:40:03 已记录 OpenVA GPU 作业停止；11:53:58 的“原 4090 批处理续跑”属于空间音频任务，不是 OpenVA 恢复，也不能将那张卡占来使用。来源：[迁移会话](siyuan://blocks/20261101193924-xq94spu)。当前请求没有授权修改 SiYuan 检索器，本轮仅调整检索方法。

## 已确认的问题类

CPU 网页代理生命周期与 GPU 推理作业生命周期分离；迁移代理时沿用了旧 GPU 地址，静态入口没有呈现后端作业状态。检查范围为旧 CPU、新 CPU 的 OpenVA 18786 代理、原 GPU 作业平台状态及已有 OpenVA 资源归属记录，未扫描或操作其他项目。

- 旧页面仍指向 `chliu-cpu-copy2`，实例 `3e5ee1e1-b086-4752-bf0e-fe35b7cf5dc2`。
- 新 CPU：`sunxc-v1`，实例 `bfe61be3-29c6-41f0-a274-5fe56eb0405a`；OpenVA 代理在迁移后已经运行，但仍转发到 `10.246.1.141:18788`。
- 原 GPU 作业 `job-d6d6c808-330f-49ae-8d4a-e87b72631956`：平台两次核验均为 `job_stopped`，`finished_at=1793567659000`，即 2026-11-02 05:14:19 CST。
- **停止原因未确认。** 不能用旧 supervisor 的计划结束时间解释本次停止。
- 没有核实到新的 OpenVA 专用 GPU 分配；共享账号下其他任务不能视为可用资源。

## 本次修改与实测

新增 `scripts/serve_worker_proxy.py`，保留原代理的 HTTP 流和 WebSocket 双向转发；增加可原子更新的后端状态文件。状态为离线时，页面返回中文原因，健康/媒体接口保持 HTTP 503；配置缺失或损坏也不会冒充就绪。状态改为 ready 后仍必须访问真实 worker，不把配置中的 ready 当作健康检查通过。

状态文件：部署根 `availability_20261102/backend.json`；当前原因 `gpu_stopped`。以后 worker 换 IP 时可原子更新文件，无需继续改源码中的地址。只有明确核验新作业归属、真实后端健康后，才可设置 `status=ready` 和 `upstream=http://实际地址:18788`；浏览器真实对话需要随后单独验收。

`scripts/recover_cpu_proxy_20261102.py` 保存本次一次性部署操作：核对新机 hostname、原代理完整 argv、cwd、进程启动时间，使用 pidfd 仅停止新机上我们的代理。新 PID `498497`，替换新机旧 PID `142707`；旧开发机代理没有停止，其他任务不动。此脚本绑定该次 PID/启动时间，**不能作为日常重启脚本反复执行**。

- 本机真实 HTTP/WebSocket 测试 **5/5 通过**；同版代码服务器环境复验也通过。
- 覆盖停止作业、缺失/损坏配置、配置 ready 但连接失败、worker 自身 503 / 拒绝 WebSocket、恢复后 HTTP 查询和二进制分块/文本及二进制 WebSocket 原样转发；确认浏览器 Cookie/Authorization 不转发给 worker。
- 新机 `/`（Accept: text/html）、`/health`、`/ws/media` 实际均返回 503；页面为中文离线说明，JSON 原因为 `gpu_stopped`。
- 没有修改模型、采样、分辨率、TTS、参考音色、权重或依赖安装。此改动属于入口和运行状态处理，不产生新的模型性能/效果结论。

## 新入口与剩余项

[sunxc-v1 OpenVA 入口](https://nat2-notebook-inspire.sii.edu.cn/ws-6e6ba362-e98e-45b2-9c5a-311998e93d65/project-7e0957fb-eaa7-4ded-8dca-dd508b2ae01d/user-ef4936dd-0231-4485-ba30-34e92bf3ea53/vscode/bfe61be3-29c6-41f0-a274-5fe56eb0405a/5d733ed2-b06a-4531-b4a0-7c2b8c727128/proxy/18786/)

此入口取自新实例访问回执；验证的是新机本地代理响应，**未做用户浏览器登录态/公网路由验收**。当前也没有声音/人像可测试。

1. 需要用户指定明确授权的现有 GPU worker，或明确同意新申请资源；本次未新建 GPU 作业，也未占用其他作业。
2. 在确认归属的 worker 恢复此前模型、MPS、VoxCPM2 参考音色和原生 25fps/512 配置；健康通过后更新代理状态，重新进行完整故事、长问题、Listener/思考连续播放和整卡显存验收。不得沿用旧 PID/IP 当作运行证据。
3. 镜像清理、环境迁移与旧 CPU 进程清理由另一迁移工作继续负责；本次不声明这些工作已全部完成。没有重新登录。

证据本地：`D:/桌面/codex_outputs/openva_infra_20261026/evidence/availability_20261102/`；服务器：`/inspire/ssd/project/video-generation/public/sunxincheng/openva_dialogue_20261030/availability_20261102/`。主要文件 `platform_jobs_summary.json`、`new_cpu_probe.json`、`new_cpu_acceptance.json`、`backend.json`、原代理备份 `dev_proxy.before.py`。
