# 本地 HTML 打开导致启动失败：2026-11-02

用户日志显示 WebSocket URL 使用了不支持的 `file:` 协议。页面的连接地址由 `location.href` 构造；直接打开本地 HTML 时，该路径没有 HTTP 服务来源。此错误发生在建立服务器连接之前。

本次修改仅在前端增加来源检查：本地文件打开时显示中文提示并禁用开始按钮；启动回调及 WebSocket URL 构造也执行同一检查。HTTP/HTTPS 的代理路径处理保持原样，没有调整模型、音色、画质或推理参数，没有重启模型服务。

已部署服务提交：`1d79b5f8e4d987e2b8b76ea7f1d5caa765730ca6`。

## 验证

- `python -m unittest tests.test_browser_startup -q`：2 项通过，覆盖来源错误在连接/麦克风获取之前显示。
- `node tests/test_ws_proxy_path.js`：15 个代理路径案例及 2 个不支持的来源案例通过。
- 本地文件浏览器验证：显示“当前打开的是本地页面文件”提示，无 WebSocket 构造异常，无媒体连接。
- 16:39:44–16:40:00 的 HTTPS 浏览器启动验证：226 个媒体数据块，最后采样已解码 292 帧，浏览器报告丢帧为 0、播放速率为 1；人像可见。麦克风连接和 AudioWorklet 初始化成功（测试使用静音虚拟麦克风）。没有浏览器异常。

[本次验证入口](https://nat2-notebook-inspire.sii.edu.cn/ws-6e6ba362-e98e-45b2-9c5a-311998e93d65/project-7e0957fb-eaa7-4ded-8dca-dd508b2ae01d/user-ef4936dd-0231-4485-ba30-34e92bf3ea53/vscode/bfe61be3-29c6-41f0-a274-5fe56eb0405a/5d733ed2-b06a-4531-b4a0-7c2b8c727128/proxy/18786/?client=file-origin-fix-20261102)

原始证据在 `D:/桌面/codex_outputs/openva_restore_20261102/`：`file_scheme_fix.json`、`browser_file_origin_guard.json`、`browser_file_origin_https.json`、对应的 `page_*.png`。

这是启动检查，不是整段对话或长期稳定性验收。服务日志在连接初期仍有配对队列满提示；此前记录的卡顿、长连接断线问题仍未解决，详见 [音画与网络复核](av_sync_network_20261102.md)。不恢复此前撤回的 localhost 隧道入口。
