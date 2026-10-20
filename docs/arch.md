https://sts2.huijiwiki.com/wiki/%E9%A6%96%E9%A1%B5
https://github.com/DarkArcZ/STS2MCP 优化 v0.111版本
https://github.com/Gennadiyev/STS2MCP
https://huggingface.co/openbmb/MiniCPM5-2B


```C++
src/sts2_advisor/
  api.py          # 只读访问 STS2MCP /singleplayer
  normalize.py    # 原始游戏状态 -> 紧凑决策上下文、合法候选动作
  retrieval.py    # Chroma / JSONL 知识检索
  model.py        # RuleAdvisor、MiniCPM Transformers 推理
  validation.py   # 拒绝模型虚构的动作、目标、证据
  render.py       # 控制台建议输出
  service.py      # 轮询、状态去重、降级处理、会话记录
  cli.py          # 命令行入口

data/
  knowledge.jsonl # 当前的本地知识库
  chroma/         # Chroma 持久化目录
  manifests/      # 攻略来源、许可证、版本记录

training/
  build_sft.py    # 审核过的会话记录 -> SFT 数据
  build_dpo.py    # 偏好标注 -> DPO 数据
  train_sft.py    # LoRA SFT 入口
  train_dpo.py    # DPO 入口

evals/
  run_eval.py     # 检查模型输出是否合法

tests/
  fixtures/       # 固定游戏状态样本
  test_normalize.py
```
### 训练
每条训练样本必须有：
```json
{
  "state": "完整 STS2MCP 原始状态",
  "context": "压缩后的决策上下文",
  "candidates": "程序生成的合法动作",
  "evidence": "本次检索到的攻略片段",
  "approved_advice": {
    "candidate_id": "card:2",
    "reason": "敌人本回合攻击12，现有生命满但防御可避免无意义掉血。",
    "confidence": 0.85
  },
  "reviewer": "你的名字或标注者",
  "game_version": "v0.111",
  "decision_type": "combat_next_action"
}
```
推荐过程：
1. 用 --backend rule --watch 或当前模型运行，自动记录局面。```.venv\\Scripts\\python.exe main.py --watch --backend rule```
2. 人工审核每条建议，修正推荐动作和理由。
3. 把审核结果写到 data/annotations.jsonl。
4. 使用 training/build_sft.py 构建 train/validation/test。
5. 在 AutoDL 用 QLoRA 做 SFT。
6. 先比较“基础模型 + 相同 RAG”与“SFT 模型 + 相同 RAG”。
7. 有足够困难的成对偏好数据后，再做 DPO。

数据覆盖应优先保证不同决策类型，而不是只采集大量战斗：
- 战斗：不同敌人意图、斩杀、药水、抽牌、状态。
- 选牌：拿牌、跳过、构筑协同、Boss 对策。
- 地图：精英、篝火、商店、问号的路线取舍。
- 商店：买卡、买遗物、删牌、存钱。
- 休息点：休息、升级、其他特殊选项。
- 事件与 Boss 遗物。

### Chroma
macOS 的安装及入库命令见 [macOS 使用说明](macos.md)。下列为 Windows PowerShell 示例；先进入当前项目目录。

要启用 Chroma：
```POWERSHELL
uv sync --extra rag
```
然后把攻略入库：
```POWERSHELL   
$env:PYTHONPATH="src"

.venv\Scripts\python.exe -m sts2_advisor.ingest 攻略文件.md `
  --title "铁甲战士攻略" `
  --url "攻略原始链接" `
  --character "铁甲战士" `
  --decision-type combat_next_action `
  --decision-type card_reward `
  --license "确认后的许可证" `
  --store both
```
之后运行：
```aiignore
.venv\Scripts\python.exe main.py --watch --backend rule --rag chroma
```

### 远程api优化路径
核心逻辑位于 [planning.py](H:/Visualstudiotable/SlayTheSpire_Instructs/src/sts2_advisor/planning.py:20)：

- 模型通过 `plan_candidate_ids` 返回最多 8 步短计划。
- 战斗中允许连续规划出牌和结束回合。
- 商店中允许按当前金币规划多个购买。
- 奖励界面允许连续领取确定性奖励。
- 地图、事件、选牌等不确定场景仍然一步一问。

每次状态变化都会验证：

- 手牌是否只移除了计划中的牌
- 是否出现额外抽牌、弃牌、升级或费用变化
- 下一张牌是否仍可打出、目标是否仍存在
- 商店商品是否按计划消失
- 金币减少量是否等于已购买商品总价
- 奖励领取后是否进入选牌等新界面

验证通过时输出：

```text
复用上一轮短计划，本次未请求模型。

计划：痛击 -> 打击 -> 结束回合
```

验证失败则废弃计划，重新请求模型。手牌、商店和奖励索引变化会自动按“动作类型 + 名称 + 目标”重新匹配。即使你连续手动执行多步、监听器没有观察到中间状态，也能识别已完成的计划前缀。

缓存方面：

- 固定规则和 JSON 输出结构全部移入稳定的 system prompt 前缀。
- 删除了每次动态重复发送的 JSON Schema。
- 短计划直接减少 API 请求次数。
- 日志新增 `cache_hit_tokens`、`cache_miss_tokens` 和 `plan_cache_hit`。

首次使用新提示词通常是冷缓存，后续请求才会开始命中。DeepSeek 后台的历史比例是累计值，建议以新日志中的单次请求数据判断效果。

直接运行：

```powershell
.venv\Scripts\python.exe main.py --watch --backend openai
```

加上 `--auto-act` 后，也是一项一项执行，每一步执行前都会重新读取并验证状态，不会把整串动作无检查地一次性发给游戏。

完整测试结果：61 项通过，2 项因当前机器没有 Apple MPS 而跳过。未调用真实 DeepSeek API，因此没有消耗额度。

