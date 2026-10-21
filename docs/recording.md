# 原始轨迹记录

轨迹记录负责保留程序实际读到、检索到、生成和发送的数据，供以后回看、回放分析或自行转换。没有奖励计算、样本筛选、训练标注或 SFT/GRPO 数据转换。

## 使用

正常运行助手就会记录，无需额外开关。例如 macOS：

```bash
.venv/bin/python main.py --watch --backend openai
```

自动执行时继续使用原有的 `--auto-act`。记录过程不增加游戏请求、不重发动作。用 `--recording-dir /path/to/directory` 更换保存目录；`--no-record` 同时关闭 SQLite 轨迹和旧版会话 JSONL。运行日志仍由原有日志功能管理。

```text
recordings/
├── trajectories.sqlite3         # 多次运行共用的轨迹库
└── session-时间戳.jsonl           # 原有决策摘要，保留兼容性
```

SQLite 数据使用 JSON 文本保存原有字段和值；同一事件中的数组顺序保留，不将原始数据展开成固定的游戏字段表。模型原始文本和校验后的建议分别保存。

## 三张表

| 表 | 字段 | 含义 |
| --- | --- | --- |
| `sessions` | `session_id`, `started_at`, `ended_at`, `status`, `metadata_json` | 一次程序运行，包含 Python、系统、架构和启动配置 |
| `steps` | `step_id`, `session_id`, `step_index`, `decision_id`, `started_at`, `ended_at`, `status` | 一次决策发生；同一状态重复决策会得到不同 `step_id` |
| `events` | `event_id`, `session_id`, `sequence`, `step_id`, `timestamp`, `elapsed_ms`, `event_type`, `payload_json` | 按发生顺序保存原始过程数据 |

`session_id`、`step_id` 为 UUID。`decision_id` 保留现有上下文哈希，用于识别相同决策状态，不能代替步骤主键。`step_index` 和 `sequence` 分别是会话内的步骤序号、事件序号。时间戳使用 UTC；`elapsed_ms` 使用单调时钟记录相对于记录器创建时的时间。

环境轮询发生在步骤之间，因此 `events.step_id` 可以为空；其 `environment.observation` 载荷包含 `after_step_id`，指出这是哪次决策之后的观测。这个关系只代表时间顺序，不表示状态变化一定由该动作造成。游戏提供的 run、seed、楼层、回合等信息保留在原始状态中，不自行推断对局身份。

每次写入独立提交，不会在等待模型或游戏响应时持有写事务。正常完成、错误退出和 Ctrl+C 分别标记为 `completed`、`error`、`interrupted`。进程被强制杀死时，尚未结束的会话或步骤可能仍为 `running`，已经提交的事件保留。

## 事件内容

| 事件 | 原始内容 |
| --- | --- |
| `environment.request` / `environment.response` | 环境 GET 的关联 ID、端点、HTTP 状态和响应文本；解析失败的响应也保留 |
| `environment.observation` | 每次主循环读取的完整游戏状态及 `after_step_id`，包括没有候选动作的状态 |
| `environment.error` / `environment.transport_error` | 读取失败和网络异常 |
| `decision.input` | 完整 `raw_state`、规范化 `context`、所有候选动作及参数 |
| `retrieval.request` | 使用的检索器、上下文、`top_k` |
| `retrieval.query` | JSONL 词项、Chroma 查询文本，或 SQLite 实体 ID/名称等实际查询条件 |
| `retrieval.ranked` | JSONL 返回的证据快照及原生词项匹配分数 |
| `retrieval.raw_result` | Chroma 返回结果，包含其实际返回的距离等字段 |
| `retrieval.rows` | SQLite 命中的原始事实/交互记录 |
| `retrieval.result` | 最终合并后交给后端的证据快照，保留文本、顺序、来源和 metadata |
| `model.request` | `call_id`、后端、实际消息和生成参数；远程 API 不包含认证参数 |
| `model.input` | 本地模型渲染后的提示词、输入 token/attention mask、generation config |
| `model.raw_response` / `model.response` | 远程 SDK 返回的结构化原始对象，以及提取后的文本、服务返回的 reasoning、usage、结束原因；本地模型记录生成文本及 token IDs |
| `model.parsed` | 本地模型输出解析后的 JSON |
| `model.error` | 单次调用异常及耗时 |
| `decision.proposal` | 主后端给出的结构化建议，保存于校验之前 |
| `decision.validation_error` / `decision.fallback` | 解析、生成或校验错误，以及使用规则回退的原因 |
| `decision.output` | 最终建议、实际后端来源和控制台呈现文本 |
| `plan.created` / `plan.reused` / `plan.invalidated` | 短计划内容、来源步骤、复用后的建议与证据、失效事件 |
| `action.precheck` | 执行动作前重新读取的完整原始状态 |
| `action.request` | `action_id` 和实际 POST 动作参数 |
| `action.response` | 对应 HTTP 状态、原始响应文本及解析后的响应 |
| `action.error` | 请求失败或响应无法解析；连接中断、超时等保留 `outcome=unknown` |
| `action.skipped` | 只读模式、重复状态或状态过期等跳过原因 |
| `decision.error` / `session.end` | 步骤异常及会话结束状态 |

同一次远程调用的请求与响应通过 `call_id` 关联；同一次动作的请求与响应通过 `action_id` 关联。应用层扩大 token 上限后的再次调用具有新的 `call_id`；SDK 内部的 HTTP 重试不单独采集。

短计划复用会关联最初生成计划的 `origin_step_id`，保存当前重新绑定的候选和证据，不产生虚构的模型调用或检索记录。规则后端也不产生模型生成记录。

`action.response` 表示收到了响应，是否被游戏接受需查看其中的业务 `status`，不能仅按 HTTP 200 判断。动作响应成功也不能代替后续游戏状态观测。单次运行或退出之后不会额外请求状态；监听模式下，后续正常轮询自然记录变化。

记录器不读取 `.env`，也不采集请求认证头。原始游戏内容、提示词和模型回答完整保留。SQLite 写入失败会在日志中警告并停用本次数据库记录，原有建议和动作流程继续；旧 JSONL 仍按原有机制尝试写入。

## 按时间查看和导出

macOS 在项目目录列出会话：

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.record
```

将指定会话按事件顺序输出到终端：

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.record --session "会话 UUID"
```

导出为 JSONL：

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.record \
  --db recordings/trajectories.sqlite3 \
  --session "会话 UUID" --output replay.jsonl
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m sts2_advisor.record
.venv\Scripts\python.exe -m sts2_advisor.record `
  --session "会话 UUID" --output replay.jsonl
```

导出首行 `record_type=session` 包含会话信息和步骤清单，后续每行 `record_type=event` 包含一个完整事件。导出是查看和离线回放的数据入口，不会向游戏重新发送动作。需要稳定的完整文件时，在会话结束后导出。

程序中直接读取：

```python
from pathlib import Path
from sts2_advisor.record import iter_events

for event in iter_events(Path("recordings/trajectories.sqlite3"), session_id):
    print(event["sequence"], event["step_index"], event["event_type"])
    payload = event["payload"]
```

也可使用 SQLite 工具查询：

```sql
SELECT sequence, step_id, event_type, payload_json
FROM events
WHERE session_id = ?
ORDER BY sequence;
```

旧版 `session-*.jsonl` 新增 `session_id`、`step_id`，可以关联到数据库中的具体决策。现有字段继续保留；旧摘要不包含全部过程事件，以 SQLite 轨迹为准。
