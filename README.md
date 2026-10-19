# STS2 Advisor

基于 STS2MCP 的《杀戮尖塔 2》控制台建议器。默认只读取：

```text
GET http://localhost:15526/api/v1/singleplayer
curl -s http://localhost:15526/api/v1/singleplayer
```

添加 `--auto-act` 后会通过 POST 自动执行推荐：`.venv/bin/python main.py --watch --auto-act --backend transformers`。
动作发送与执行逻辑集中在 [`src/sts2_advisor/action.py`](src/sts2_advisor/action.py)。

## 当前能力

- 读取并规范化 STS2MCP 单人状态。
- 支持战斗、手牌选择、奖励、选牌、地图、事件、休息点、商店、宝箱等决策。
- 从状态生成严格的合法候选，模型只能选择候选 ID。
- 压缩重复牌堆，保留手牌索引、敌方目标和路线信息。
- 相同状态使用稳定 `decision_id`，持续监听时不会重复建议。
- 支持规则后端和本地 Transformers 模型后端。
- 支持 Chroma；未安装 Chroma 时自动使用本地 JSONL 知识库。
- 将局面、证据和建议写入 JSONL，但默认标记 `training_consent: false`。

## 快速开始

### macOS（zsh / bash）

需要 Python 3.11+；macOS 自带的 `/usr/bin/python3` 可能版本过低。已安装 [uv](https://docs.astral.sh/uv/) 后，在项目目录执行：

```bash
uv sync --python 3.13 --locked
.venv/bin/python main.py --backend rule
.venv/bin/python main.py --watch --backend rule
.venv/bin/python main.py --dump-context
```

先在游戏中安装并启动 STS2MCP，确认 `localhost:15526` 可访问。规则后端不需要 PyTorch、模型权重或 Chroma。

Apple Silicon 本地模型：

```bash
uv sync --python 3.13 --locked --extra model
.venv/bin/python main.py --watch --backend transformers \
  --model openbmb/MiniCPM5-2B --model-cache MiniCPM5_2B --device auto
```

`auto` 按 CUDA → Apple MPS → CPU 选择；可用 `--device mps` 或 `--device cpu` 指定。MPS 使用 float16，CPU 使用 float32。模型仍只从本地加载，需提前准备完整文件；`--model` 可直接指定本地模型目录，`--model-cache` 则指定 Hugging Face 缓存根目录（模型缓存位于其 `hub/` 子目录）。

完整安装、攻略入库、测试与故障排查见 [macOS 使用说明](docs/macos.md)。

Windows 与 macOS 共用同一代码；保留 Windows PowerShell 命令、CUDA 自动映射和现有缓存环境变量配置。已设置的 `HF_HOME`、`HF_HUB_CACHE`、`HF_XET_CACHE` 不会被覆盖，其中 `HF_HUB_CACHE` 优先于 `--model-cache` 的默认 hub 路径。

### Windows（PowerShell）

在项目目录运行一次只读建议：

```powershell
.venv\Scripts\python.exe main.py --backend rule
```

持续监听游戏：

```powershell
.venv\Scripts\python.exe main.py --watch --backend rule
```

打印发送给模型的压缩上下文：

```powershell
.venv\Scripts\python.exe main.py --dump-context
```

规则后端只用于验证工程链路，不代表高水平策略。

## 使用 MiniCPM

模型文件下载完整后运行：

```powershell
.venv\Scripts\python.exe main.py --watch --backend transformers `
  --model openbmb/MiniCPM5-2B `
  --model-cache MiniCPM5_2B
```

本地模型必须包含完整权重、配置和 tokenizer 文件。当前实现使用离线加载，不会在运行游戏时临时下载模型。模型返回非法动作、错误引用或无效 JSON 时，程序会自动降级到规则后端。

## 攻略入库

无需 Chroma，可以先把 Markdown/TXT 攻略切分进 JSONL：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m sts2_advisor.ingest docs\ironclad-guide.md `
  --title "铁甲战士攻略" `
  --url "https://example.com/source" `
  --character "铁甲战士" `
  --decision-type combat_next_action `
  --decision-type card_reward `
  --license "获得授权后的许可证"
```

不要省略或虚构来源和许可证。新内容默认标记为 `unreviewed`。

安装 `chromadb` 后可使用：

```powershell
.venv\Scripts\python.exe main.py --rag chroma
```

目前的入库命令输出 JSONL。向 Chroma 批量写入前，应先确定中文/英文 embedding 模型并用检索评测集验证；否则 `--rag auto` 会可靠地使用 JSONL 回退。

直接写入 Chroma：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m sts2_advisor.ingest docs\ironclad-guide.md `
  --store both `
  --decision-type combat_next_action `
  --chroma-path data\chroma
```

Chroma 默认 embedding 可能在首次入库时下载模型。正式使用前应固定 embedding 模型及其版本，并把版本写入数据 manifest。

## 测试

macOS：

```bash
uv sync --python 3.13 --locked --extra dev
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pytest -q
.venv/bin/python test_api.py
```

单元与集成测试使用本机临时 HTTP 服务，不需要游戏；`test_api.py` 需要真实游戏 API。安装 `model` 可选依赖后，测试还会用随机初始化的微型模型离线验证 CPU/MPS 加载与生成，不下载 MiniCPM 权重。

Windows：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

API 契约测试仍可运行：

```powershell
.venv\Scripts\python.exe test_api.py
```

## 已知边界

- 只支持单人建议，不自动操作。
- 规则后端仅做最低限度防御/合法性演示。
- MiniCPM5-2B 的策略质量必须用冻结评测集验证，不能仅凭示例判断。
- STS2MCP 使用 `DarkArcZ/STS2MCP` 的 `fix/v0.111` 分支时，游戏更新后仍应重新运行契约测试。
- Crystal Sphere、复杂 card/bundle/relic overlay 的字段可能随模组变化，需要继续用真实 fixture 扩充测试。
