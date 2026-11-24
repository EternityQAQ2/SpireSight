# 开发者模式

`--dev` 为提示词调优和链路排查提供一份按阶段组织的可读记录：环境、局面、检索证据、完整提示词、模型 reasoning 与输出、校验/回退结果、最终动作，以及 token 消耗。

它与[原始轨迹记录](recording.md)互补：轨迹是面向程序的 SQLite 事件流，开发者模式是面向人的顺序文本。两者互不依赖，`--dev` 可以和 `--no-record` 一起使用。

## 使用

macOS（zsh / bash）：

```bash
# 远程 API，输出到终端并写入 logs/dev/
.venv/bin/python main.py --watch --backend openai --dev

# 只写文件，不刷屏
.venv/bin/python main.py --watch --backend openai --dev --dev-quiet

# 指定日志目录
.venv/bin/python main.py --watch --backend openai --dev --dev-dir /tmp/sts2-dev
```

Windows（PowerShell）：

```powershell
.venv\Scripts\python.exe main.py --watch --backend openai --dev
```

每次运行生成一个 `dev-YYYYmmdd-HHMMSS.log`。终端输出与文件内容一致；`--dev-quiet` 只关闭终端回显。文件写入失败不会中断决策流程。

## 输出结构

一次运行按以下顺序写入，同一数据只出现一次：

| 阶段 | 内容 |
| --- | --- |
| 环境信息 | 启动时间、Python/平台/架构/解释器/工作目录、依赖版本、游戏接口、后端、模型、设备、RAG、`top_k`、自动执行、轮询参数、日志与轨迹路径 |
| 步骤开始 | `decision_id`、`state_type`、`decision_type`、完整 `raw_state`、规范化 `context`（含全部候选动作） |
| 计划复用 / 确定性决策 | 分支原因、复用的建议；这两种分支不会请求模型，token 为 0 |
| 检索查询 / 排序 / 事实 / 注入 | 实际查询词或实体、词项得分或向量距离、SQLite 各表命中数、注入的基线规则；这些阶段不含证据正文，避免与下一行重复 |
| 检索证据 | `top_k`、检索器类型、每条证据的 `evidence_id`、来源、URL、metadata 和完整正文 |
| 提示词 | **完整 system prompt 与完整 user prompt**，按实际发送内容原样保存；远程模型重试时不会重复打印 |
| 模型请求 | 每次调用的后端、模型、尝试序号、输出上限、JSON/思考模式；失败请求也计入汇总 |
| 模型输出 | reasoning（服务返回时）与 content 全文、`finish_reason`、耗时、字符数、`call_id` |
| 建议生成 / 计划修复 / 校验失败 | 后端原始建议；短计划修复原因；异常类型与消息；规则后端回退建议 |
| 最终建议 | 控制台呈现文本、实际后端、只读或自动执行模式 |
| 动作请求 / 动作响应 | 自动执行时发送的完整动作参数和游戏响应 |
| Token | 每次模型调用的输入/输出/总计、缓存命中与未命中、reasoning token，以及步骤累计与会话累计 |
| 会话汇总 | 决策步骤数、模型请求数、输入/输出/总 token、缓存命中率、平均每次请求与平均每步 token、总耗时 |

## Token 统计口径

- 远程 API 使用服务返回的 `usage`：`prompt_tokens`、`completion_tokens`、`total_tokens`。
- 缓存字段同时兼容 OpenAI 的 `prompt_tokens_details.cached_tokens`、DeepSeek 的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`。
- 模型请求数包含失败或未返回 `usage` 的调用；未返回 `usage` 时明确标注「服务未返回 usage」。
- 重试会作为独立请求分别统计。
- 本地 Transformers 后端没有服务端 usage，使用 tokenizer 的实际输入长度和生成 token 数估算，记录中标注「估算」。

## 注意

- 日志包含完整游戏局面、攻略证据和模型原文，可能含敏感或版权内容，不应公开上传。
- 不记录 API 密钥；环境信息只包含模型名、接口地址和非敏感配置。
- `--dev` 会增加磁盘写入和终端输出，不建议在长时间无人值守运行时开启终端回显。
