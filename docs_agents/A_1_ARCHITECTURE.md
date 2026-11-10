# 当前架构与维护导航

2026-11-10核实。此文替换旧的只读MVP架构草案；当前已经有显式开启的自动执行、短计划和function calling。既有用户要求优先于旧草案。

## 职责与边界

| 范围 | 主要文件与职责 |
|---|---|
| 启动组装 | cli.py：参数、环境配置、具体实现组装；create_remote_backend供线上/回放共用；contracts.py：跨模块数据和协议 |
| 应用编排 | application/advisor.py：观察、去重、唯一候选、检索、模型、校验、短计划、执行 |
| 游戏状态 | game/state.py、deck.py、candidates.py：规范化、持牌观测、合法候选 |
| 单张算术 | game/combat/calculator.py：指定攻击直接伤害，纯函数，无数据库/模型/游戏I/O |
| 已支持回合效果 | game/combat/effects.py、simulation.py、search.py：有限效果与搜索，不能当通用模拟器 |
| 决策输入 | decision/context.py、metrics.py、prompts.py：图鉴缓存、当前数值、提示词和skill加载；可选首轮攻击结果复用原calculator，不实现第二套算术 |
| 决策输出 | decision/parsing.py、validation.py、plan_check.py、plans.py：解析、校验、资源/目标与复用 |
| 工具协议 | decision/tools.py：schema和局面绑定分派，不实现第二套伤害规则 |
| 游戏接入 | integrations/sts2/client.py、actions.py：REST、动作预检和去重 |
| 模型接入 | integrations/models/remote.py、local.py、rules.py：SDK传输和响应处理，不放战术规则 |
| 知识接入 | integrations/knowledge/factory.py、sqlite.py、chroma.py、lexical.py、seeds.py |
| 检索相关性 | integrations/knowledge/relevance.py：纯查询组织和排序；common.py保留共用接口 |
| 记录统计 | recording/session.py、trajectory.py、stats.py、logging.py |
| 离线工具 | evals/、tools/、training/：与在线执行分开，实验不隐式操作游戏 |

所有src路径相对于 `src/sts2_advisor`。

- game依赖标准库、同包和contracts，不读取数据库，不调用模型或执行动作。
- application/decision不直接导入具体integrations；CLI通过协议注入实现。
- recording不回调业务；内部依赖无环，可选SDK延迟加载。约束由tests/test_architecture.py检查。
- StateBuilder/DeckTracker按会话隔离，CodexCache按后端实例隔离。不要引入全局可变游戏状态或缓存污染其他会话。
- 小功能使用现有模块边界，不为这个单进程项目先造路由/服务框架。

## 线上流程注意事项

1. 相同状态去重；唯一合法动作已有确定处理，不调用检索/模型。
2. 已有计划先检查是否可安全复用；未知、抽牌、随机、关键HP/盾/状态偏离时停止。
3. 战斗检索top_k为min(配置,4)，其他局面用配置值；seed不占名额。
4. 模型提案检查合法性；无效计划最多重决策一次。战斗不自动改为规则战术。
5. 默认只读，只有auto-act执行。动作前重新读取；watch的失败状态暂停避免重复付费。

policy.py/survival.py等旧方法的存在不意味着在线使用，检查advisor.py实际导入与调用。不要通过改测试恢复已取消的战术接管。

首轮攻击结果模式通过OPENAI_PRECOMPUTE_ATTACKS控制，默认false（实测降本但质量门槛失败）。开启只影响战斗输入/工具注册，不改变应用战术责任。`evals.compare_attack_delivery`注入录制观察、冻结证据及无执行器，调用真实AdvisorService.run_once；复用配置工厂，避免复制另一套在线准备流程。

## 重要事实来源

- 三路知识与当前模型配置见 [PROJECT_STATE](PROJECT_STATE.md)。当前手牌描述/费用优先，静态Wiki不能覆盖当前值。
- API契约：`../docs/SLS2MCP_api.md`。字段缺失以实际观测和完整度标记处理，不发明接口。
- 记录使用与导出：`../docs/recording.md`；统计固定输出 `../docs/run-stats.md`。
- 历史实验和失败方向见 [EXPERIMENTS](EXPERIMENTS.md)，修改前先读，避免重复花费。

## 验证

Windows项目虚拟环境可运行 `PYTHONPATH=src;tests;.` 加 `python -m unittest discover -s tests -q`。部分模型测试会生成本地微型随机模型，避免据此声称真实模型策略能力。环境未安装pytest时用现有unittest，不为文档修改安装新依赖。

文档清理仅验证保护文件哈希、Markdown链接和Git差异，不需要重跑模型或支付API。全量测试可能刷新run-stats生成时间；只能在确认无其他差异时恢复测试生成的那一行，不能覆盖用户战绩。
