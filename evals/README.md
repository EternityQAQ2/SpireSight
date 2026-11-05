# 固定局面回归

`fixtures/log_cases.jsonl` 冻结了 2026-11-03 最近一局中的三个真实局面：三能量五费连招、缺少持牌清单的选牌、包含全身撞击和铁斩波的战斗。每行保留来源会话、步骤与事件序号，运行时不读取会变化的轨迹数据库。

不调用模型或游戏的基线分析：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe evals/replay.py --output tmp/decision-replay.json
```

报告区分候选合法性、逐步资源错误、必须重新观测的边界、选牌理由字段，以及修复后计划。历史输出存在错误是预期基线，不会使这条分析命令失败。

比较模型或思考模式时，为同一份 fixtures 收集每个 `case_id` 的原始输出，分别保存为 JSONL：

```json
{"case_id":"overbudget","advice":{"candidate_id":"card:0:NIBBIT_0","reason":"...","confidence":0.7,"plan_candidate_ids":["card:0:NIBBIT_0","card:1:NIBBIT_0","card:2","combat:end_turn"]}}
```

每行必须包含 `candidate_id`、`reason`、`confidence`，`decision_id` 由固定状态重新生成；选牌输出还需 `selection_analysis`，包含 `gap`、`contribution`、`skip_comparison`。每份预测需覆盖全部局面，重复、未知 case ID 和缺失案例不予忽略。

```powershell
.venv\Scripts\python.exe evals/replay.py --predictions tmp/model-a.jsonl --output tmp/model-a-report.json
.venv\Scripts\python.exe evals/replay.py --predictions tmp/model-b.jsonl --output tmp/model-b-report.json
```

预测检查在覆盖不足、非法候选、资源错误、计划跨越观测边界或缺少选牌解释时返回非零。比较时固定提示词、牌组快照、检索证据、输出预算和 fixtures，只改变模型或思考配置；另外记录耗时、tokens、模型版本。收集预测属于单独的模型实验，本工具不加载凭据、不请求 API。

这三个案例检验机械正确性，不评估长期最优，也不足以证明胜率提升。新效果应先用实际状态变化验证数值语义，再添加独立用例并进入支持白名单。随后用固定种子、多局实测比较过第一幕比例、每战掉血、药水使用与通关率。
