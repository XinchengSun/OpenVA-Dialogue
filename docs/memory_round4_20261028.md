# 默认启用渲染优化，继续清点语音显存

基线提交：`28ef493fa8fc9995a025338ef6bf1090055a2816`。
继续参考用户提供的 [ResearchBoost-Agent 方法](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/df4378df1949761d2d5f051cc2d3f5e01e3cfabb/docs/methodology.md)：
固定输入与比较规则、反例自测、独立进程原版→候选→原版，以及明确记录未验证范围。

## 已直接启用的两项

根据用户本轮明确要求，`app.py` 的默认值及 `.env.example` 均改为：

```bash
DYSTREAM_CACHE_SPECTRAL_NORM=1
DYSTREAM_RELEASE_RENDER_INTERMEDIATES=1
```

不设置环境变量时也会启用；显式 `0` 仍用于原版对照。GPU 加载集成测试覆盖了显式
开、关、两项环境变量均缺省三种情况，并检查参数值及 checkpoint 键一致。
上一轮真实 LIA 训练权重下，两项组合的渲染子管线峰值为 807.10→679.35 MiB，
4 帧输出及 116 条中间层记录逐位一致；本轮没有更改这两项的计算实现。
历史 round-3 文档、JSON 和原始哈希保留其当时的默认关闭记录，不改写历史证据。

本地没有私有 `.env`，未连接或重启远端服务。已有服务器环境若显式写了 `0`，需要
在该实例配置中改成 `1`；本轮“启用”指本 checkout 的源码默认值和新配置模板。

## 剩余大头：区分权重清点与运行实测

检查固定 [VoxCPM2 模型版本 32279ef](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915)
的 safetensors 头部，核对固定 config 的 Git blob；没有把下面的权重大小冒充 GPU 峰值：

| 部分 | 大小 | 核查口径 |
| --- | ---: | --- |
| base LM，含文本词表 | 2.74 GiB | checkpoint 张量字节清点 |
| residual LM | 0.70 GiB | checkpoint 张量字节清点 |
| 局部特征 encoder + decoder | 0.78 GiB | checkpoint 张量字节清点 |
| 两套 KV cache | 288 MiB | 官方 StaticKVCache 类，28/8 层、BF16、8192 长度的独立 CUDA 分配实测 |
| base LM 文本词表 | **286.90625 MiB** | 真实训练词表的独立 CUDA 分配/卸载实测；已包含在第一行中 |

该 `model.safetensors` 不含另外加载的 AudioVAE，表中也不含其显存、编译图、工作区和
驱动上下文。活跃 LM/特征网络每个音频块都参与计算，本轮没有卸载它们，也没有缩短
KV 容量。完整并发运行的实际分布仍需目标服务器测量。

## 新改动：低频词表留 CPU，向量仍回原 GPU

源码依据为固定 [VoxCPM 实现 cce58a5](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py)：
文本 lookup 发生在 LM prefill 前，后续生成循环使用向量；编译的是 `forward_step` 等
路径，未编译这个词表查询。它没有与输出头共享参数。

新增 [`text_embedding_offload.py`](../voice_service/text_embedding_offload.py)：

- 原始 BF16 词表留在 CPU，按原 token ID 查出所需行后传回原 GPU；后续缩放和 LM 运算
  留在原设备，词表、dtype、token 顺序和上下文容量均不变。
- checkpoint 键和值不变；用零字节、非持久化 buffer 跟踪 `.to()` 的逻辑设备。
  训练/有梯度、CPU 索引和不兼容设备情况走原始路径；设备/dtype 转换及重载已测试。
- 共享权重、`max_norm`、已编译 embedding、自定义 embedding 等不满足条件时跳过。
  只在串行后端初始化中安装，未改动上游包。

后端开关 `VOXCPM2_CPU_TEXT_EMBEDDING=0` **仍默认关闭**，因为只验证了词表及适配器，
没有完整训练模型的 PCM/音色验收。可在独立测试实例设为 `1`，不能作为“已证明整套
语音效果不变”的结论。用户要求默认启用的原有两项不受此开关影响。

## 真实训练词表的实测

Windows、RTX 4060 Laptop 8188 MiB、Python 3.12.3、PyTorch 2.8.0+cu126、14 CPU 线程。
从固定版本 safetensors 的字节范围 `[71504,300914511]` 读取唯一词表张量，严格核对
HTTP 206 Content-Range、长度、形状和 dtype，记录该张量 SHA-256：
`7b7e11c548e6e9e36777326e5c2bba02ec9cd87999bd1df5730613be861c6452`。
未下载全部 4.58 GB 文件，因此**未声称验证了整个 checkpoint 的 LFS SHA-256**。

冻结 judge 后，三个独立进程按原版→CPU lookup→原版运行。每个进程检查全部
73,448×2,048 个 BF16 数值；全部词条的返回值逐位一致，CPU/CUDA RNG 状态不变。
反例检查拒绝数值、NaN、shape 和 dtype 改动。参数更新、重载、设备/dtype 往返、梯度
回退、共享参数拒绝和后端加载/生成接入也已验证。

候选初始化后 CUDA allocated 从 300,843,008 降到 0 字节，释放 **286.90625 MiB**。
这是仅含词表的进程统计，实际后端仍有其他 GPU 张量；需要额外约同量 CPU 内存。
本次准备耗时约 61.9 ms，属于一次性 CPU 搬运成本。

连续 25 次 lookup 的均摊 wall time，预热 10 次、3 个窗口，profiler 关闭：

| 查表输入长度 | 原版前/后均摊 ms | CPU lookup+回传 ms |
| --- | --- | ---: |
| 128 | 0.01448 / 0.01424 | 0.27227 |
| 512 | 0.01269 / 0.01832 | 0.44659 |
| 2048 | 0.01843 / 0.01772 | 1.19613 |
| 8192 | 0.30992 / 0.32858 | 6.14432 |

输入长度包括参考/续接前缀里的 padding，不只新增文本。这项优化用少量查表搬运时间
换常驻显存；它不是加速，不能从此表推出整句 TTFA/RTF 的变化。

## 更完整的流式显存探针

原探针在整句生成结束后记录，无法看到生成器暂停期间仍保留的前缀张量。
现在设置 `OPENVA_MEMORY_PROFILE_DIR` 时，会在每句第 1/2/4/8… 块音频处额外记录，
有界地读取暂停生成器及嵌套生成器的 CUDA 张量引用，不推进生成器。
可清点 `enc_outputs`、prefill KV tuple、组合特征等仍存活的存储，结合原有参数、KV、
Graph pool 和未归类差额，定位权重之外的占用。

诊断默认关闭，开启会同步 CUDA，不能混入正常延迟测试。只记录代码变量名和尺寸，
不保存张量值、文本、音频或任意字典键。实际 4 MiB CUDA 存储及别名去重、隐私、
不推进生成器、后端首块采集与取消优先级均有回归覆盖。

## 验证与 TODO

回归 165 项：150 通过、15 跳过（14 项缺 Pipecat、1 项 Linux symlink 测试在 Windows
跳过）；后续针对改动的 TTS/流式回归 39 项通过。编译及 diff 检查通过，远端 CI 未运行。
[结构化证据](memory_round4_validation_20261028.json)记录 judge、原始结果和源码 SHA-256；
完整原始文件在父目录 `evidence/rba_round4`，训练词表在父目录 `weights/voxcpm2-embedding`。

```bash
python -m unittest tests.test_renderer_memory_integration tests.test_text_embedding_offload tests.test_text_embedding_judge tests.test_generator_memory_inventory -v
python tests/bench_text_embedding_memory.py --table /data/embed_tokens.bf16 \
  --metadata /data/embedding_download.json --output-dir /data/embedding-baseline
python tests/bench_text_embedding_memory.py --table /data/embed_tokens.bf16 \
  --metadata /data/embedding_download.json --offload --output-dir /data/embedding-candidate
# 独立进程补跑后置 baseline；每个输出目录必须是新目录。
```

下一步在实际官方 TTS 环境做同种子 PCM、音色和 TTFA/RTF 对照，再决定是否默认启用词表
卸载；同时用流式探针量出完整前缀/编译图工作区，最后验证 12 GiB 卡的冷启动及持续多轮。
**当前优化后整机显存仍未知，不能把词表、渲染器和 checkpoint 大小相加减来认定 12 GiB 达标。**
