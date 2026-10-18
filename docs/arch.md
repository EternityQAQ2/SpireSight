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
要启用 Chroma：
```POWERSHELL
cd H:\Visualstudiotable\SlayTheSpire_Instructs
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
