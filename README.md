# STS2 Advisor

基于 STS2MCP 的《杀戮尖塔 2》LLM 决策助手。程序读取单人游戏状态，生成合法候选动作，结合 SQLite 游戏事实与攻略知识输出结构化建议。

默认只读；只有显式添加 `--auto-act` 才会向游戏发送动作。

源码按职责分为 `application`（流程）、`game`（游戏规则）、`decision`（决策与校验）、`integrations`（外部接入）、`recording`（记录统计）。入口负责组装实现，主流程通过接口使用模型、检索器和动作执行器。目录导航、依赖约束及“改什么去哪里找”见 [源码架构与维护指南](docs/arch.md)。

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。以下命令均在项目目录执行。

### macOS（zsh / bash）

macOS 自带的 `/usr/bin/python3` 可能版本过低；可让 uv 寻找或准备 Python 3.13。同时安装本地模型与远程 API 依赖：

```bash
uv sync --python 3.13 --locked --extra model --extra remote
.venv/bin/python --version
.venv/bin/python main.py --help
```

仅使用规则后端时执行 `uv sync --python 3.13 --locked` 即可；仅使用远程 API 时添加 `--extra remote`，无需安装 PyTorch 或准备本地模型权重。`uv sync` 会移除未选择的 extra，后续同步时需保留仍要使用的 `--extra` 参数。

### Windows（PowerShell）

同时安装本地模型与远程 API 依赖：

```powershell
uv sync --locked --extra model --extra remote
```

仅使用规则后端时，基础安装即可：

```powershell
uv sync --locked
```

## 运行

先在游戏中安装并启动 STS2MCP，确认 `localhost:15526` 可访问。

### macOS（zsh / bash）

交互选择本地或远程模型：

```bash
.venv/bin/python main.py --watch
```

也可显式指定：

```bash
# 远程 API，只输出建议；先按下文“远程 API”配置 .env
.venv/bin/python main.py --watch --backend openai

# 本地 Transformers 模型
.venv/bin/python main.py --watch --backend transformers \
  --model openbmb/MiniCPM5-2B --model-cache MiniCPM5_2B --device auto

# 规则后端，仅用于验证链路
.venv/bin/python main.py --backend rule
.venv/bin/python main.py --watch --backend rule

# 打印当前压缩决策上下文
.venv/bin/python main.py --dump-context
```

Apple Silicon 应使用原生 arm64 Python。`--device auto` 按 CUDA → Apple MPS → CPU 选择；可用 `--device mps` 或 `--device cpu` 指定。MPS 使用 float16，CPU 使用 float32。检查当前环境是否支持 MPS：

```bash
.venv/bin/python -c 'import platform, torch; print(platform.machine()); print("MPS:", torch.backends.mps.is_available())'
```

本地模型使用离线加载，需提前准备完整权重、配置和 tokenizer。`--model` 可直接指定本地模型目录（路径含空格时加引号）；`--model-cache MiniCPM5_2B` 对应的 Hugging Face 缓存位于 `MiniCPM5_2B/hub/`。已设置的 `HF_HOME`、`HF_HUB_CACHE`、`HF_XET_CACHE` 不会被覆盖，其中 `HF_HUB_CACHE` 优先于 `--model-cache` 的默认 hub 路径。

`.venv/bin/python main.py --watch --backend openai` 只会持续读取游戏状态并输出远程模型建议，不会自动控制人物动作。需要自动执行建议时，显式添加 `--auto-act`：

```bash
.venv/bin/python main.py --watch --backend openai --auto-act
```

启用后程序会直接向游戏发送动作。请先启动游戏和 STS2MCP，并按下文“远程 API”配置 `.env`；按 `Ctrl+C` 停止程序。

完整安装、攻略入库、测试与故障排查见 [macOS 使用说明](docs/macos.md)，包括 editable 安装后出现 `ModuleNotFoundError` 时的 `.pth` 隐藏标记排查。

### Windows（PowerShell）

启动游戏和 STS2MCP 后，交互选择本地或远程模型：

```powershell
.venv\Scripts\python.exe main.py --watch
```

也可显式指定：

```powershell
# 远程 API
.venv\Scripts\python.exe main.py --watch --backend openai

# 本地 Transformers 模型
.venv\Scripts\python.exe main.py --watch --backend transformers `
  --model openbmb/MiniCPM5-2B --model-cache MiniCPM5_2B --device auto

# 规则后端，仅用于验证链路
.venv\Scripts\python.exe main.py --watch --backend rule
```

打印当前压缩决策上下文：

```powershell
.venv\Scripts\python.exe main.py --dump-context
```

自动执行属于高风险模式，仅在明确需要时启用：

```powershell
.venv\Scripts\python.exe main.py --watch --backend openai --auto-act
```

## 远程 API

复制 `.env.example` 为 `.env`：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
OPENAI_MAX_TOKENS=1024
OPENAI_JSON_MODE=true
OPENAI_THINKING=disabled
```

远程后端使用官方 `openai` Python 库，也支持兼容 OpenAI Chat Completions 的服务。进程环境变量优先于 `.env`；`.env` 已被 Git 忽略。DeepSeek 默认使用 JSON Output 并关闭思考模式，避免为短决策生成冗长思维链；需要重新启用时设置 `OPENAI_THINKING=enabled`。

使用 OpenCode Go 套餐时设置 `OPENCODE_MODE=true`，并把 `OPENAI_BASE_URL` 指向 Go 入口：

```dotenv
OPENAI_BASE_URL=https://opencode.ai/inference/go/openai/v1
OPENAI_MODEL=kimi-k3
OPENCODE_MODE=true
```

开启后客户端会发送 Go 要求的 `User-Agent`（标识为 `spiresight-advisor/0.1`）和每个进程稳定复用的 `x-opencode-session`，避免被当作匿名 SDK 流量。该模式面向编码类流量设计，长时间非编码调用可能受用量或滥用策略限制。

运行日志写入 `logs/advisor-*.log`，包含远程 API 原始响应 JSON 以及抽取后的 `content`/`reasoning` 文本，启动时自动删除超过 3 天的日志。日志不记录 API 密钥，但可能包含游戏局面与模型回答，不应公开上传。可用 `--log-retention-days` 和 `--log-dir` 调整。

模型会为战斗、商店和奖励列表生成短计划：支持的战斗效果允许连续多步，通常以 `combat:end_turn` 收尾。后续状态若符合预期，程序重新核验剩余计划后复用；费用、效果和目标变化必须通过当前状态校验，抽牌、生成手牌、未支持效果、商品不可购买或界面跳转等情况会触发重新决策。

提示词按「静态前缀 → 动态尾部」组织。`stable.deck` 是卡牌效果图鉴，按（角色、幕、层）冻结；当前手牌费用和变化后的说明放在动态区覆盖图鉴，避免为了缓存而使用过期数值。候选提供动态费用、伤害和格挡，无法解析的值为 `null`。攻击牌的格挡也会计算。

本局牌组另存于 `player.deck_snapshot`：首次观测一场战斗时合并手牌、抽牌堆、弃牌堆和消耗堆，按名称、升级状态和数量汇总，跨界面提供给选牌、商店、事件、地图和休息点。已打出的能力牌不会因离开这些区域而被误删，后续新增的战斗牌另列；自动执行成功并观测到界面变化后，跟踪拿牌、购买及可识别的单张升级/移除，下场战斗重新校正。当前模组没有常驻完整牌组字段，因此历史快照会注明来源、可能过期和未跟踪变化，不能当成权威永久持牌清单。数据库中的全卡牌资料仍用于效果知识检索。

战斗计算器对支持的确定性效果枚举当前回合可行顺序，提供输出、格挡、剩余能量和预计掉血。模型计划与缓存复用都会经过逐步费用、手牌和目标检查；未支持的效果需要执行后重新读取状态。试算只是当前回合的启发式参考，不是完整游戏模拟。选牌输出还必须说明本局缺口、当前拿牌收益及与跳过的比较。模块边界和支持范围见 [架构说明](docs/arch.md)，固定日志回归见 [离线评估](evals/README.md)。

## 开发者模式

调试提示词或排查链路时添加 `--dev`。它按决策阶段输出环境、原始局面、检索证据、完整提示词、模型 reasoning 与输出、校验/修复结果、最终动作，并统计输入/输出/缓存命中的 token 消耗：

```bash
.venv/bin/python main.py --watch --backend openai --dev
```

详情写入 `logs/dev/dev-时间戳.log`，同时打印到终端；`--dev-quiet` 只写文件，`--dev-dir` 更换目录。开发者模式独立于轨迹记录，可与 `--no-record` 同时使用。完整说明见 [开发者模式](docs/developer-mode.md)。

## 原始轨迹记录

远程模型默认启用只读 function calling：`evaluate_combat_plan(decision_id, candidate_ids)`。模型可自行提出多套出牌顺序，工具返回逐步费用与支持范围内的结果，然后由模型返回最终建议。工具不执行游戏动作；抽牌、未知效果或不支持的局面会报告预测边界，不能将其当作完整回合结论。每个决策最多 3 轮、8 次工具调用，之后要求最终输出。调用会额外消耗模型 tokens；接口不支持时可设 `OPENAI_FUNCTION_CALLING=false`。工具结果作为 `model.tool_result` 写入轨迹。最终建议继续经过既有校验及执行前状态预检。

正常运行会自动保存到 `recordings/trajectories.sqlite3`，按会话、决策步骤和事件顺序记录原始游戏状态、攻略检索结果、模型实际输入输出、最终动作选择、动作请求和响应，以及后续轮询状态。短计划复用和模型失败回退也保留来源。

原有 `session-*.jsonl` 继续保存。可用 `--recording-dir` 更换目录，`--no-record` 关闭轨迹记录。按时间查看、导出 JSONL 和数据库字段说明见 [原始轨迹记录](docs/recording.md)。

## 数据与检索

- `data/sts2_facts.sqlite3`：卡牌、遗物、怪物等确定性 Wiki 事实。
- `data/knowledge.jsonl`：少量策略种子，也是未启用 Chroma 时的回退知识库。
- `data/chroma/`：后续存放需要语义检索的攻略和构筑知识。
- `data/manifests/`：记录攻略来源、许可证、版本、抓取时间和内容哈希。

`--rag auto` 会优先使用非空 Chroma，并合并 SQLite 事实；Chroma 不可用时使用 `knowledge.jsonl`。因此未来 Chroma 建好后，`knowledge.jsonl` 可以只保留为种子和应急回退，`manifests/` 应继续保留。

攻略入库示例：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m sts2_advisor.ingest docs\ironclad-guide.md `
  --store both --title "铁甲战士攻略" --url "https://example.com/source" `
  --character "铁甲战士" --decision-type combat_next_action `
  --chroma-path data\chroma --license "实际许可证"
```

## 测试

macOS（zsh / bash）：

```bash
uv sync --python 3.13 --locked --extra dev --extra model --extra remote
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pytest -q
.venv/bin/python test_api.py
```

Windows（PowerShell）：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe test_api.py
```

单元测试不需要游戏；`tools/test_api.py` 需要真实 STS2MCP。

## 边界

- 支持单人模式；多人模式不在范围内。
- 模型只能选择当前状态生成的合法候选，非法输出会被拒绝并回退规则后端。
- 规则后端只验证工程链路，不代表可靠策略。
- 游戏或模组更新后，应重新运行 API 契约测试。

## 许可证

本项目采用 [MIT License](LICENSE) 授权。
