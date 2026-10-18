# macOS 使用与验证

## 环境与安装

项目要求 Python 3.11+。先检查 `python3 --version`，不要直接使用低版本系统 Python。本文使用 uv 和 Python 3.13；uv 会寻找或准备该版本解释器。

```bash
cd /path/to/SpireSight
uv sync --python 3.13 --locked
.venv/bin/python --version
.venv/bin/python main.py --help
```

也可用已安装的 Python 3.11+ 创建虚拟环境：

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e .
```

在 macOS 上使用 `.venv/bin/python`，Windows 对应 `.venv\Scripts\python.exe`。安装后也可激活环境 `source .venv/bin/activate`，使用 `sts2-advisor` 和 `sts2-ingest` 入口。

若安装成功后 `sts2-advisor` 或 `python -m sts2_advisor.ingest` 仍报 `ModuleNotFoundError`，检查 `ls -lO .venv/lib/python*/site-packages/*.pth`。本次环境中虚拟环境的 `.pth` 文件带有 macOS `hidden` 标记，Python 3.13 会忽略这些文件，导致 editable 安装的 `src/` 路径没有加载。确认是此问题后，清除这些路径配置文件的隐藏标记：

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth
.venv/bin/sts2-advisor --help
```

这是虚拟环境修复；直接从项目运行 `main.py` 不受此问题影响。

## 连接游戏

本项目只包含建议器，不包含 STS2MCP 模组。按你使用的模组版本安装说明，在 macOS 游戏中启用模组并启动游戏，然后运行：

```bash
.venv/bin/python test_api.py --timeout 2
.venv/bin/python main.py --backend rule --rag json
.venv/bin/python main.py --watch --backend rule --rag json
```

`test_api.py` 会 GET 检查多个只读端点；建议器运行时只 GET `/api/v1/singleplayer`。没有游戏或模组时，连接失败是预期结果，不能算作真实游戏联调通过。先用规则后端确认链路，再使用模型后端；规则建议不代表高水平策略。

Ctrl+C 停止监听。相同局面稳定读取后只输出一次建议。默认记录至 `recordings/`；用 `--no-record` 关闭。

## Apple Silicon 模型推理

安装模型依赖：

```bash
uv sync --python 3.13 --locked --extra model
.venv/bin/python -c 'import platform, torch; print(platform.machine()); print(torch.__version__); print("MPS:", torch.backends.mps.is_available())'
```

Apple Silicon 应使用原生 arm64 Python。设备选择依据当前 PyTorch 的可用性检测：`auto` 优先 CUDA，其次 MPS，最后 CPU。指定不可用设备会报错，并提示可切换 CPU。CPU 使用 float32，MPS/CUDA 使用 float16。MPS 模型和输入张量都会放到 Apple GPU；CUDA 保留现有自动映射。

程序离线加载，不会替你下载权重。可以提前将完整权重、配置和 tokenizer 放在本地目录：

```bash
.venv/bin/python main.py --watch --backend transformers \
  --model "/path/to/local model" --device auto
```

若使用原有 Hugging Face 缓存结构，`--model-cache MiniCPM5_2B` 默认会从 `MiniCPM5_2B/hub/` 查找仓库。它不是直接存放 `config.json` 的模型目录。两种加载方式都使用 `local_files_only=True`。保留原有缓存环境变量的默认设置且不覆盖已有值；若设置了 `HF_HUB_CACHE`，优先使用该路径。实际 hub 路径同时显式传给 tokenizer 和模型，避免 Hugging Face 导入顺序影响缓存查找。

MPS 不可用、内存不足或模型算子不兼容时，可尝试：

```bash
.venv/bin/python main.py --backend transformers \
  --model "/path/to/local model" --device cpu --max-new-tokens 128
```

模型加载失败会退出；生成失败或输出非法动作时，现有流程降级到规则建议。CPU 可能更慢且需要更多内存。Intel Mac 可使用无模型依赖的规则后端；本次没有验证 Intel Mac 的 PyTorch 安装和模型推理。

设备处理参考 [PyTorch MPS 文档](https://docs.pytorch.org/docs/stable/notes/mps.html)，离线加载及缓存参数参考 [Transformers 模型文档](https://huggingface.co/docs/transformers/main_classes/model)。

## 攻略入库与训练数据准备

JSONL 入库无需额外依赖，使用实际攻略文件和真实来源信息：

```bash
.venv/bin/python -m sts2_advisor.ingest "docs/我的攻略.md" \
  --title "攻略标题" --url "https://example.com/source" \
  --character "铁甲战士" --decision-type combat_next_action \
  --license "已确认的许可证"
```

Chroma 为可选功能：

```bash
uv sync --python 3.13 --locked --extra rag
.venv/bin/python -m sts2_advisor.ingest "docs/我的攻略.md" \
  --title "攻略标题" --url "https://example.com/source" \
  --license "已确认的许可证" --store both --chroma-path data/chroma
.venv/bin/python main.py --backend rule --rag chroma
```

Chroma 的 embedding 可能在首次使用时下载。本次默认验证 JSONL 链路；安装某个 extra 的 `uv sync` 会移除未选择的其他 extra，若需要模型与 Chroma 同时使用，传入 `--extra model --extra rag`。

macOS 可准备审核后的 SFT/DPO 数据，具体格式见 [训练说明](../training/README.md)。长时间 SFT/DPO 仍建议按项目原有流程在经过验证的训练设备上进行；本次 macOS 推理适配没有验证训练和 QLoRA。

## 测试

```bash
uv sync --python 3.13 --locked --extra dev --extra model
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pytest -q
```

覆盖设备选择、局面规范化、合法动作、中文及含空格路径、HTTP 错误、只读请求、会话录制、监听去重和 SIGINT 退出。模型运行测试在安装 torch/transformers 后启用，使用本地随机初始化的微型模型；没有 MPS 时跳过 GPU 项。这些测试不证明 MiniCPM 的兼容性或策略质量，也不能替代真实游戏契约测试。

测试会在 `127.0.0.1` 随机端口启动临时服务；受限沙箱需允许本机回环网络。
