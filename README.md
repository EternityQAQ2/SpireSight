# STS2 Advisor

基于 STS2MCP 的《杀戮尖塔 2》LLM 决策助手。程序读取单人游戏状态，生成合法候选动作，结合 SQLite 游戏事实与攻略知识输出结构化建议。

默认只读；只有显式添加 `--auto-act` 才会向游戏发送动作。

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。同时安装本地模型与远程 API 依赖：

```powershell
uv sync --locked --extra model --extra remote
```

仅使用规则后端时，基础安装即可：

```powershell
uv sync --locked
```

## 运行

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

运行日志写入 `logs/advisor-*.log`，包含远程 API 原始响应 JSON 以及抽取后的 `content`/`reasoning` 文本，启动时自动删除超过 3 天的日志。日志不记录 API 密钥，但可能包含游戏局面与模型回答，不应公开上传。可用 `--log-retention-days` 和 `--log-dir` 调整。

模型会为战斗、商店和奖励列表生成可验证的短计划。后续状态若只包含预期变化，程序复用计划而不再次请求模型；出现抽牌、卡牌变化、目标消失、商品不可购买或界面跳转时立即重新决策。固定规则与输出结构位于稳定的 system prompt 前缀中，以提高 DeepSeek 的上下文缓存命中并缩短动态请求。日志中的 `cache_hit_tokens`、`cache_miss_tokens` 和 `plan_cache_hit` 可用于评估效果。

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

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe test_api.py
```

单元测试不需要游戏；`test_api.py` 需要真实 STS2MCP。

## 边界

- 支持单人模式；多人模式不在范围内。
- 模型只能选择当前状态生成的合法候选，非法输出会被拒绝并回退规则后端。
- 规则后端只验证工程链路，不代表可靠策略。
- 游戏或模组更新后，应重新运行 API 契约测试。
