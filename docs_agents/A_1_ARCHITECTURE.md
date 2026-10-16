# 杀戮尖塔 2 智能建议系统：软件架构设计

> 状态：架构草案 v0.1  
> 日期：2026-10-16  
> 目标项目目录：`H:\Visualstudiotable\SlayTheSpire_Instructs`  
> 当前范围：读取游戏状态，在标准输出中给出建议；不自动操作游戏，不做前端。

## 1. 执行摘要

本项目应拆成三个相互独立、可分别迭代的能力：

1. **游戏状态接入**：通过 STS2MCP 读取当前局面，并转换成稳定的内部状态格式。
2. **知识增强推理**：从 Chroma 检索攻略、卡牌、遗物和事件知识，与当前局面一起交给模型。
3. **模型后训练**：先用 SFT 学习“如何分析和按固定格式回答”，有足够偏好对后再用 DPO 改进决策风格和取舍。

核心结论：**Chroma/RAG 不是微调**。RAG 是推理时的外部知识层；SFT/DPO 才会改变模型参数。二者组合是合理的，但应分别建设、分别评测。

推荐的落地顺序是：

`只读接入 -> 基础模型 + RAG 基线 -> 评测集 -> 数据采集 -> SFT -> DPO -> 持续回归评测`

不要一开始同时做 SFT、DPO 和 RAG。没有基线和评测集时，无法判断任何一个步骤是否真的改善了建议质量。

## 2. 需求边界

### 2.1 当前目标

- 从 STS2MCP 读取单人游戏状态。
- 识别当前是否处于需要玩家决策的状态。
- 对战斗、选牌、路线、事件、商店、休息点、宝箱和 Boss 遗物等场景给出建议。
- 建议输出到标准输出，供玩家阅读。
- 输出应包含推荐动作、简短理由、备选动作、置信度和引用的攻略来源。
- 支持更换基础模型、Embedding 模型和推理后端。
- 保存匿名化局面、检索结果、建议和玩家最终选择，用于后续训练和评测。

### 2.2 当前不做

- 不通过 STS2MCP 的 POST 接口自动执行动作。
- 不做网页、桌面悬浮窗或游戏内 UI。
- 不保证多人模式正确性；STS2MCP 将多人能力标记为 beta。
- 不要求模型从截图理解游戏；优先使用结构化状态。
- 不把模型的长篇思维过程作为训练目标或用户输出。

### 2.3 建议粒度

每次只建议**当前局面的下一步动作**。尤其在战斗中，打出一张牌后游戏状态会变化，系统应重新读取状态并重新推理，而不是一次生成整回合的不可校验动作序列。

## 3. 对现有想法的评审

| 想法 | 评审 | 建议调整 |
|---|---|---|
| SFT + Chroma/RAG 微调模型 | 方向合理，但概念混在一起 | 把 RAG 定义为推理时知识层，把 SFT/DPO 定义为离线训练层，单独评测 |
| 用攻略做 SFT/DPO 数据 | 攻略适合 RAG，但不能直接成为高质量训练样本 | 攻略先入知识库；训练集必须转成“完整局面 -> 合法建议”，并由规则或专家复核 |
| 一开始同时做 SFT 和 DPO | 风险高，难定位收益 | 先建评测集和 SFT 基线；有稳定的 chosen/rejected 对后再做 DPO |
| 使用 1B/2B 小模型 | 能做低成本 MVP，但复杂长程规划可能不足 | 用强模型或人工生成/审核标签，小模型做学生模型；用真实评测决定是否升级到 7B/8B 级别 |
| 只靠攻略提升胜率 | 不足 | 攻略描述一般原则，实际决策还需要完整牌组、遗物、血量、路线、敌人意图及版本信息 |
| 用 MCP 作为唯一通信方式 | 可用，但会增加部署耦合 | 建立 `GameStateProvider` 抽象；本项目默认直连本机 REST GET，保留 MCP 适配器 |
| 输出自然语言建议 | 对玩家友好，但难验证 | 模型先输出严格结构化结果，程序验证后再渲染成控制台文本 |

## 4. 总体架构

```text
Slay the Spire 2
        |
        v
STS2MCP Mod (localhost:15526, no auth)
        |
        | GET only
        v
+----------------------- Advice Runtime ------------------------+
| GameStateProvider -> StateNormalizer -> DecisionGate          |
|                                  |                            |
|                                  +-> FeatureBuilder           |
|                                  +-> RetrievalQueryBuilder    |
|                                           |                   |
|                                           v                   |
|                                  Chroma Knowledge Base        |
|                                           |                   |
|                                           v                   |
|                                Prompt/Context Assembler        |
|                                           |                   |
|                                           v                   |
|                                  Policy Model Backend         |
|                                           |                   |
|                                           v                   |
|                         Schema + Legality + Citation Validator|
|                                           |                   |
|                         ConsoleRenderer <-+-> Session Recorder|
+---------------------------------------------------------------+
                                                |
                                                v
                                   Training / Evaluation Store
                                                |
                    +---------------------------+----------------+
                    |                                            |
                    v                                            v
             SFT dataset builder                         DPO pair builder
                    |                                            |
                    +-----------------> Trainer <----------------+
                                          |
                                          v
                              Versioned adapter/checkpoint
```

在线建议进程与离线训练进程必须分开。游戏过程中不训练模型，只记录数据；训练完成后通过版本化配置切换新模型。

## 5. 在线建议链路

### 5.1 `GameStateProvider`

职责：读取 STS2MCP 状态，对上层隐藏 REST/MCP 差异。

建议接口：

```python
class GameStateProvider(Protocol):
    async def get_state(self) -> RawGameState: ...
    async def get_profile(self) -> RawProfile: ...
    async def search_wiki(self, query: str, item_type: str) -> list[WikiItem]: ...
```

实现：

- `Sts2RestStateProvider`：默认实现，访问 `GET /api/v1/singleplayer`。
- `Sts2McpStateProvider`：未来需要接入已有 MCP Host 时使用。
- `ReplayStateProvider`：从本地样本重放状态，用于测试和评测。

当前进程**不实现通用 POST 方法**，从接口层防止模型误操作游戏。STS2MCP 的 HTTP 服务无鉴权且能控制游戏，只应绑定并访问 `localhost`。

### 5.2 `StateNormalizer`

STS2MCP 的原始响应随 `state_type` 变化。Normalizer 将其转换成版本化的内部结构：

```json
{
  "schema_version": "1.0",
  "game_version": "0.103.2",
  "adapter_version": "sts2mcp-0.3.4",
  "decision_id": "sha256:...",
  "decision_type": "combat_next_action",
  "run": {"run_id": "...", "act": 1, "floor": 7, "ascension": 3},
  "player": {},
  "encounter": {},
  "legal_actions": [],
  "visible_options": [],
  "history_summary": []
}
```

重要规则：

- 保留 `entity_id`、卡牌索引、奖励索引等执行所需标识，但不给模型虚构标识的机会。
- `legal_actions` 必须由程序根据状态生成，不由模型猜测。
- `decision_id` 由规范化状态哈希生成，用于去重。
- 保存游戏版本、STS2MCP 版本和数据模式版本，防止游戏更新后旧数据静默污染新模型。

### 5.3 `DecisionGate`

职责：避免轮询时重复输出或在动画/过渡态误触发。

- 每 300-500 ms 轮询一次本机状态，具体值可配置。
- 只在 `decision_id` 改变且状态连续两次稳定时触发。
- 对 `unknown`、`overlay`、主菜单等不可建议状态保持静默或输出一次状态提示。
- `monster/elite/boss` 每次动作后重新推理。
- 对完全相同的状态使用短期缓存，避免重复推理成本。

### 5.4 `FeatureBuilder`

生成模型真正需要的紧凑特征，减少 token 和噪声：

- 战斗：敌人意图、格挡/生命、能量、手牌、抽/弃/消耗堆摘要、能力和遗物效果。
- 选牌：当前牌组曲线、核心协同、已有重复牌、缺失能力、即将面对的 Boss。
- 路线：可达节点、剩余血量、药水、金币、精英收益风险、Boss 信息。
- 商店：库存、价格、金币、删除卡价值、当前构筑缺口。
- 事件/休息点：选项、代价、当前资源和后续路线。

原始状态仍写入记录，但发送给模型的是裁剪后的 `DecisionContext`。

### 5.5 Chroma 知识库

建议 collection：

| Collection | 内容 | 更新频率 |
|---|---|---|
| `strategy_chunks` | 攻略原则、角色构筑、路线与战斗技巧 | 攻略更新时 |
| `game_entities` | 卡牌、遗物、药水、敌人、事件的版本化说明 | 游戏/模组更新时 |
| `expert_cases` | 已审核的相似局面与建议 | 持续追加 |
| `patch_notes` | 版本改动和数据失效范围 | 每个版本 |

每个 chunk 至少带以下 metadata：

```json
{
  "source_id": "stable-id",
  "source_title": "...",
  "source_url": "...",
  "license": "...",
  "game_version_min": "0.103.2",
  "game_version_max": null,
  "character": "ironclad",
  "act": 1,
  "decision_types": ["card_reward"],
  "entity_ids": ["..."],
  "language": "zh-CN",
  "review_status": "human_verified"
}
```

检索流程：

1. 从当前局面生成结构化查询，而不是直接向量化整份 JSON。
2. 先按游戏版本、角色和决策类型做 metadata filter。
3. 向量召回 10-20 条。
4. 使用轻量 reranker 或规则重排到 4-8 条。
5. 设定上下文 token 预算，去除重复和版本冲突内容。
6. 把 `source_id` 一同传给模型，输出时只能引用已提供的来源。

Embedding 模型必须支持中文和英文攻略；具体模型在做小型检索评测后决定，不应仅凭排行榜选型。

### 5.6 `PolicyModelBackend`

用统一接口支持：

- 本地 Transformers 推理，用于开发验证。
- vLLM 等独立推理服务，用于稳定运行和批量评测。
- 可选远程强模型，用作教师、标签生成器或基线。

建议将 LoRA/QLoRA adapter 与基础模型分开版本化。模型输入只包含：系统规则、规范化局面、合法候选、RAG 证据；模型输出严格 JSON。

### 5.7 输出协议与验证

模型输出：

```json
{
  "decision_id": "sha256:...",
  "recommended_action": {
    "type": "play_card",
    "candidate_id": "hand:2",
    "target_id": "JAW_WORM_0"
  },
  "reason": "先降低本回合预计伤害，同时保留能量完成击杀。",
  "alternatives": [
    {"candidate_id": "hand:0", "condition": "若优先保留药水"}
  ],
  "confidence": 0.78,
  "evidence_ids": ["guide-ironclad-001#chunk-12"],
  "warnings": []
}
```

验证器检查：

- JSON schema 合法。
- `decision_id` 与当前状态一致，拒绝过期结果。
- `candidate_id` 和 `target_id` 位于程序生成的候选集合中。
- 引用的 `evidence_ids` 确实来自本次检索。
- 理由长度、备选数量和输出语言符合配置。
- 失败时最多修复生成一次；再次失败则输出保守的错误信息，不猜动作。

控制台渲染示例：

```text
[A1 F7 | 战斗 | 置信度 78%]
建议：打出「防御」-> 自身
理由：先覆盖敌方本回合伤害，同时保留 1 点能量完成后续攻击。
备选：若准备使用格挡药水，可先打出「打击」-> JAW_WORM_0
依据：Ironclad 入门攻略 #12
```

## 6. 数据与训练架构

### 6.1 攻略不能直接等同于训练数据

攻略通常缺少完整输入局面，也未提供明确的 rejected 答案。直接把攻略段落做 SFT，会让模型更会“复述攻略”，不一定更会在具体局面选择动作。

攻略的优先用途应是：

1. 作为 RAG 知识源。
2. 生成候选训练场景时提供原则。
3. 由教师模型或人工把原则应用到完整游戏状态，形成可验证样本。

### 6.2 数据来源优先级

1. 专家对完整局面的标注，质量最高。
2. 玩家真实游玩轨迹，加上选择原因或事后复盘。
3. 强模型对完整局面生成候选，由规则校验和人工抽检。
4. 同一 seed 的分支/重放结果，用后续收益形成偏好对。
5. 攻略文本，仅作为知识与标注依据，不直接冒充局面标签。

必须记录来源许可。公开视频、社区攻略和网站文字可能受版权或站点条款约束；未经确认不要批量抓取并再分发原文。

### 6.3 统一样本格式

建议原始数据使用 Parquet 或 JSONL，并保留以下层次：

```text
raw_state -> normalized_state -> retrieved_evidence -> annotation
          -> model_output -> player_action -> short/long-term outcome
```

关键字段：

- `sample_id`、`run_id`、`seed`、`decision_id`
- 游戏、模组、状态 schema 和攻略版本
- 决策类型、角色、难度、楼层
- 完整合法候选集合
- 推荐选择、理由、来源、标注者和审核状态
- 最终玩家选择和结果
- 是否允许进入训练集

### 6.4 SFT 数据

SFT 学习目标：遵循结构化输出、使用证据、从合法候选中选择、给出简洁可读理由。

```json
{
  "messages": [
    {"role": "system", "content": "你是只读的 STS2 决策建议器……"},
    {"role": "user", "content": "<DecisionContext JSON>\n<Evidence>"},
    {"role": "assistant", "content": "<ValidatedAdvice JSON>"}
  ],
  "metadata": {
    "sample_id": "...",
    "run_id": "...",
    "decision_type": "card_reward",
    "quality": "human_verified"
  }
}
```

训练/验证/测试应按 `run_id` 或 seed 分组切分，不能随机按单条决策切分，否则同一局的相邻状态会泄漏到不同集合，评测结果会虚高。

### 6.5 DPO 数据

DPO 应使用同一完整 prompt 下的优劣回答对：

```json
{
  "prompt": "<same system + DecisionContext + Evidence>",
  "chosen": "<better ValidatedAdvice JSON>",
  "rejected": "<worse but plausible ValidatedAdvice JSON>",
  "preference_reason": "chosen 规避致死伤害；rejected 忽略敌人意图",
  "label_source": "expert_pairwise",
  "agreement": 0.92
}
```

不要把“非法 JSON/明显胡说”作为大多数 rejected，它们太容易区分，无法教会模型处理真正困难的策略取舍。高价值偏好对应该是两个都合法、都看似合理，但长期收益不同的答案。

只有满足以下条件后才进入 DPO：

- SFT 模型已稳定输出合法格式。
- 有至少一套冻结的测试集。
- 偏好标签有明确规则，且抽样检查标注者一致性。
- 能证明 DPO 没有降低合法率、引用正确率和关键场景表现。

### 6.6 数据质量门禁

- 自动检查状态 schema、动作合法性、索引和目标有效性。
- 冲突攻略按版本和审核等级处理。
- 去重近似状态，避免高频简单局面支配训练。
- 按角色、章节、决策类型、难度分层统计覆盖率。
- 人工审核高损失、高分歧和低置信度样本。
- 训练数据、评测数据、原始攻略分别建版本清单和哈希。

## 7. 评测体系

没有评测集，不应开始正式微调。至少建设四层评测：

### 7.1 机械正确性

- JSON schema 成功率。
- 合法候选选择率，目标应为 100%。
- 过期状态拒绝率。
- 引用存在率和版本匹配率。
- 无依据实体/规则幻觉率。

### 7.2 决策质量

- Top-1/Top-2 专家选择一致率。
- 专家成对偏好胜率。
- 致死风险、资源浪费、明显反协同等严重错误率。
- 按 `decision_type` 分层报告，不能只给总平均值。

### 7.3 长期效果

- 固定 seed 和固定版本上的通关率、到达楼层、剩余生命、精英收益。
- 有条件时用状态分支或重放估计 regret。
- 同时报告均值和置信区间，避免少量随机 run 得出错误结论。

### 7.4 系统指标

- 决策延迟 p50/p95。
- Prompt、检索证据和输出 token 数。
- 显存/内存占用。
- 状态轮询错误率、重复输出率、模型修复重试率。

初期验收建议：合法动作率 100%，结构化输出成功率 >= 99%，重复输出率 < 1%，常规决策 p95 延迟目标 <= 3 秒。策略质量指标先测基线，再设置提升目标。

## 8. 推荐工程结构

```text
SlayTheSpire_Instructs/
├─ pyproject.toml
├─ README.md
├─ configs/
│  ├─ app.yaml
│  ├─ models.yaml
│  └─ retrieval.yaml
├─ src/sts2_advisor/
│  ├─ cli.py
│  ├─ domain/
│  │  ├─ state.py
│  │  ├─ advice.py
│  │  └─ candidates.py
│  ├─ adapters/
│  │  ├─ sts2_rest.py
│  │  ├─ sts2_mcp.py
│  │  └─ replay.py
│  ├─ pipeline/
│  │  ├─ normalizer.py
│  │  ├─ decision_gate.py
│  │  ├─ features.py
│  │  ├─ retrieval.py
│  │  ├─ inference.py
│  │  └─ validation.py
│  ├─ knowledge/
│  │  ├─ ingest.py
│  │  ├─ chunking.py
│  │  └─ store.py
│  ├─ recording/
│  │  └─ session_recorder.py
│  └─ rendering/
│     └─ console.py
├─ training/
│  ├─ build_sft.py
│  ├─ build_dpo.py
│  ├─ train_sft.py
│  └─ train_dpo.py
├─ evals/
│  ├─ datasets/
│  ├─ graders/
│  └─ run_eval.py
├─ data/
│  ├─ manifests/
│  ├─ raw/
│  ├─ processed/
│  └─ chroma/
├─ tests/
│  ├─ fixtures/
│  ├─ unit/
│  ├─ contract/
│  └─ integration/
└─ artifacts/
   ├─ adapters/
   └─ reports/
```

`../data`、模型缓存和 `artifacts/` 默认不提交 Git；只提交小型 fixtures、manifest、配置和可复现脚本。

## 9. 技术选型建议

- 语言：Python。
- Python 版本：优先 3.11 或 3.12，而不是当前项目声明的 `>=3.13`；机器学习生态、量化和训练扩展通常在较保守版本上兼容更稳。最终以实际依赖锁定测试为准。
- 环境管理：继续使用 uv。
- 状态访问：`httpx` 异步客户端，严格超时和重试。
- 数据模型：Pydantic，版本化 JSON Schema。
- RAG：Chroma 持久化库 + 可替换 embedding/reranker 接口。
- 本地训练：Transformers + PEFT + TRL，优先 LoRA/QLoRA，不先做全参数微调。
- 推理：开发期 Transformers，稳定运行后按硬件评估 vLLM/llama.cpp 等后端。
- 数据：原始日志 JSONL，规模增大后转 Parquet；SQLite 保存运行索引和反馈关系。
- 测试：pytest；保存 STS2MCP 响应 fixture 做契约测试。

现有 `../main.py` 直接加载 `MiniCPM5-2B` 可作为模型冒烟测试，但不应继续承载业务逻辑。1B/2B 两个模型都应先跑同一套冻结评测，再决定保留哪个；不能用参数量直接推断胜负。

## 10. 安全、稳定性与版本漂移

- STS2MCP 默认 `localhost:15526` 且无认证，并具备完整游戏控制能力。不要把端口暴露到局域网或公网。
- 建议器只使用 GET；执行动作应是另一个独立进程、独立配置和显式开关。
- 对连接超时、游戏未启动、动画过渡、`unknown` 和 `overlay` 状态做明确处理。
- 保留最近一次有效状态，但绝不能基于旧状态给出可执行标识。
- STS2 仍可能频繁更新。每条知识、样本、评测和模型都必须关联游戏版本。
- STS2MCP README 当前写明测试版本为 `v0.103.2`，升级游戏或模组后必须先跑契约测试。
- 将第三方攻略视为不可信数据：清洗其中的提示注入式文字，绝不让攻略内容覆盖系统规则。

## 11. 分阶段实施计划

### 阶段 0：基线与契约（约 2-4 天）

- 固化 30-100 个覆盖不同 `state_type` 的响应 fixture。
- 定义 `NormalizedGameState`、候选动作和 `Advice` schema。
- 创建首版冻结评测集。
- 对 MiniCPM5-1B/2B 与一个强教师模型跑零样本基线。

交付条件：能离线重放状态，并生成可评分结果。

### 阶段 1：只读控制台 MVP（约 3-7 天）

- REST GET 接入、状态规范化、DecisionGate。
- Chroma 入库与版本过滤。
- 模型结构化输出、合法性验证、控制台渲染。
- 会话记录和错误处理。

交付条件：真实游戏中只在新决策点输出一次合法建议，不执行动作。

### 阶段 2：数据闭环（约 1-2 周）

- 记录玩家最终选择和可选反馈。
- 建立标注工具所需的数据格式；前端仍可不做，先用 CLI/文件审核。
- 建设分层评测报告和数据质量报告。
- 将攻略转换为带来源、版本和许可证的知识块。

交付条件：每条训练候选可追溯到局面、证据、标签来源和审核状态。

### 阶段 3：SFT（取决于数据量与硬件）

- 先用高质量小数据 LoRA/QLoRA。
- 与“基础模型 + 相同 RAG”做严格对照。
- 只有通过机械正确性和冻结评测才发布 adapter。

交付条件：相对基线在主要决策类型上有稳定提升，且无严重回归。

### 阶段 4：DPO

- 收集困难、合法、可争议的偏好对。
- 检查标注一致性，控制各类场景比例。
- 小规模训练并进行 SFT vs SFT+DPO 消融实验。

交付条件：专家偏好胜率提升，长期指标不退化。

## 12. 需要项目负责人审核的决策

以下问题会实质影响实现，不建议在缺少信息时替你决定：

1. **硬件预算**：GPU 型号和显存决定可训练模型大小、量化方式及推理延迟。
2. **目标角色和范围**：首版是否只支持一个角色、单人模式和低进阶。建议先单角色、单人。
3. **建议目标**：追求最高胜率，还是教学可解释性；两者训练标签可能不同。
4. **输出语言**：仅中文，还是中文建议 + 英文攻略检索。建议内部双语检索、中文输出。
5. **数据许可**：哪些攻略允许抓取、存储、改编和用于训练。
6. **反馈方式**：是否记录玩家采纳/拒绝，以及是否允许把这些记录用于训练。
7. **模型策略**：完全离线，还是允许强云模型做教师和标注辅助。
8. **游戏版本策略**：版本更新后是冻结旧模型，还是维护多版本知识库和评测集。

默认推荐：先做 Ironclad（或你最熟悉的一个角色）、单人、低进阶、中文输出、完全只读；在 200-500 个高质量评测局面上完成基线后，再决定是否投入 SFT。

## 13. 当前项目的立即改进项

1. 把目标路径统一为实际存在的 `H:\Visualstudiotable\SlayTheSpire_Instructs`。
2. 将 `../pyproject.toml` 的 Python 版本改为经过训练依赖验证的 3.11/3.12，并显式声明依赖。
3. 将 Hugging Face 缓存目录和模型名称移入配置或环境变量，不硬编码在 `../main.py`。
4. 把 `../main.py` 缩减为 CLI 入口，业务逻辑放入 `src/sts2_advisor/`。
5. 立即补充 `../.gitignore`，排除模型缓存、Chroma 数据、训练输出、会话日志和密钥文件。
6. 在下载更多模型或采集大量攻略前，先完成状态 schema、10-20 条真实 fixture 和首版评测脚本。

## 14. 参考资料

- [STS2MCP 项目与安装说明](https://github.com/Gennadiyev/STS2MCP)
- [STS2MCP API Quick Reference](https://github.com/Gennadiyev/STS2MCP/blob/main/docs/raw-simplified.md)
- [OpenAI 官方模型优化流程](https://developers.openai.com/api/docs/guides/model-optimization)：强调先建立 eval，再进行提示与微调迭代；其中也区分了 SFT 与 DPO 的用途。该资料用于方法论参考，本架构不要求使用 OpenAI 模型或 API。

外部项目状态与接口信息核对日期：2026-10-16。
