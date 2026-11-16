# 固定局面回归

20个日志快照及来源审计在 `fixtures/prompt_cases.jsonl` 等文件中。多轮复用后，原heldout标签不再代表独立未见测试。实验结论和证据索引集中在 [维护笔记](../docs_agents/EXPERIMENTS.md)，当前配置与后续方案见 [当前状态](../docs_agents/PROJECT_STATE.md)。

| 入口（python -m） | 用途 | 默认行为 / 付费预算 |
|---|---|---|
| evals.compare_prompts | 当前提示与精简提示对照 | 离线；--run才付费，32请求/16万token |
| evals.compare_retrieval | Git基线与当前真实检索对照 | 本地SQLite/Chroma/seed，不调用决策API |
| evals.replay_retrieval_decisions | 相同提示、不同冻结证据的8组重放 | 离线；--run才付费，32请求/16万token |
| evals.compare_decision_options | 工具、摘要与Pro四组对照 | 离线；--run才付费，64请求/36万token |
| evals.compare_attack_delivery | 真实应用流程，工具往返 vs 首轮攻击结果 | 离线；--prepare冻结本地检索，--run为6题×2配置×2次，最多48请求/20万token |
| evals.compare_decision_quality | 旧输入Flash / 修正输入Flash / 修正输入Pro | 离线；--run为8题×3配置×2次，最多96请求/50万token；预算不足则在网络前取消 |
| evals.replay_end_turn | 扫描本地录制的结束回合计划，区分未知、致死投影、实际是否触发重决策 | 始终离线，不导入模型/游戏客户端，不花token；--output须新路径 |
| evals.oracle_diagnostic | Flash自由规划 / 两计划 / 两计划＋核实后果 | 默认离线；--run为3题×3组×2次，最多18单请求/15万token，不挂工具或校验重试 |
| evals.compare_player_outcome | 只比较工具回包新增玩家即时HP/格挡，Git b827915为旧基线 | 默认离线；--prepare冻结三路证据；--run为3快照×2组×2次，最多48请求/20万token；普通Flash、原function calling |
| evals.medium_player_outcome | 旧计算器+旧描述 vs 新计算器+新描述，中规模只读回放 | 默认离线；32快照×2组×2次=128决策；--run才付费，硬上限512请求/200万token；不因错误补跑 |

用户于2026-11-16要求以后不再测试Pro。含Pro的旧入口仅保留复现资料，后续不得沿用其默认组合发起Pro付费测试。

oracle诊断是受限比较实验，不是线上性能回放：它绕过AdvisorService，不因校验错误重决策；事后分别审查自由规划的选择、菜单选择、动作/计划一致性和理由。评分不传给模型；第二次交换菜单顺序。参考结果是助手核对，非独立人类金标；荆棘题原输入已包含turn_search，其菜单也非全局最优。修改题目后必须重新核对后果，不能将硬编码参考表当生产模拟器或从菜单6/6外推胜率。

付费脚本必须用新输出路径，不覆盖原始结果。历史实验使用1600输出上限和top_k=6；线上战斗目前top_k最多4，另加不占名额的固定seed，本机输出配置1024。`compare_attack_delivery`复用CLI配置工厂和AdvisorService的检索、校验、一次重决策与只读交付，保留原始提案及校验后前缀；恢复已录制的观测上下文，不重新猜测历史牌组。这是隔离局面的回放，不检验连续计划复用或平均楼层。其他旧实验不等于完整线上回放。

新6题固定在 `fixtures/attack_delivery_cases.jsonl`，人工复核在 `fixtures/attack_delivery_review.jsonl`。题目与旧20题无重叠，覆盖5次录制、6场战斗；两组轮换先后，每组每题重复2次。每次独立后端实例、SDK重试关闭；输入证据冻结，仍走真实应用检索接口。原始请求及全部回复保存在本机 `tmp/attack-delivery-20261110-api.json`，复核结果不能把返回成功当成正确。

质量对照另用 `fixtures/decision_quality_cases.jsonl`（6个回归、2个新增录制局面，非独立留出集），复核 `decision_quality_review.jsonl`。`compare_decision_quality`从Git c1dd6b6加载4个旧输入模块，不能用历史坏答案充当现测基线。默认普通模式、1024输出、真实应用top_k；评分分别保留动作合理/明确差解/待判断、理由错误、原计划错误、空回复与未发送。全部API尝试共用预算，最后剩余预算不足保守预留时也不得强行发送。2026-11-10实际47次付费决策、80请求、485999 token；第48项未发，比较使用三组共同15对。

```powershell
$env:PYTHONPATH='src;tests;.'
.venv/Scripts/python.exe -m evals.compare_attack_delivery
# 明确预算并使用新的输出路径才进行付费复测：
.venv/Scripts/python.exe -m evals.compare_attack_delivery --run --token-budget 200000 --output tmp/attack-delivery-new.json
```

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
