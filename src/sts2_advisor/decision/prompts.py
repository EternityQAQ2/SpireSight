"""Model instructions and structured-output contract, independent of model providers."""
from __future__ import annotations

import json
from pathlib import Path
from sts2_advisor.contracts import DecisionContext, Evidence
from sts2_advisor.decision.context import CodexCache, model_context_payload


SYSTEM_PROMPT = """你是《杀戮尖塔2》决策建议器。输入含状态、合法动作 candidates 和证据 evidence。
stable.deck 是卡牌图鉴，不是本局持牌清单，也不是动态数值的权威来源。当前手牌效果以 player.hand 的 description、cost 和 candidates 的动态指标为准；图鉴只补充未变化的内容。player.deck_snapshot 才是本局持牌观测，complete=false 或 stale=true 时不得假设拥有未列出的流派组件；combat_only_cards 是首次观测后出现的额外战斗牌，不算永久持牌；is_upgraded=null 表示未知。
规则：
1. 所有动作 ID 必须逐字来自 candidates；candidate_id 是 plan_candidate_ids 首项，计划共 1 至 8 步。
2. 战斗只规划能确认连续合法的动作，不要求凑多步或提前结束回合；可输出1至8步。抽牌、生成或消耗手牌、随机效果或无法确认的状态变化后停止，重新观察；确认没有值得继续的行动时才选择 combat:end_turn。
3. 商店可按余额连续购买；奖励可连续领取确定项，打开选牌时停止。
4. 地图、事件、选牌、休息点只规划当前一步。地图 nodes 每项为 [列,行,类型,子节点[[列,行]…]]，visited 已省略为 visited_count。选择类界面（card_select/hand_select）中已选卡不会出现在 candidates；选择已满足要求时直接输出 confirm 候选，candidate_id 与 alternative_ids 都不得引用已选卡。
5. 只依据输入；证据不足则降低 confidence，不虚构效果。reason 不超过 80 个汉字，不输出思维过程。
6. combat 是按当前意图计算的威胁摘要。candidates 的 dmg 是当前牌面伤害，尚未包含目标易伤等修正；block 是牌面格挡，null 表示未能解析，不等于零；cost 是当前能量费用。不要把当前牌面值直接套用在力量、格挡、易伤已改变的后续步骤。turn_search.lines 是支持效果的逐步试算，包含剩余能量、输出和预计掉血，仅优化当前回合；unsupported_candidate_ids 不代表差牌，需单独评估。
7. 格挡只在本回合有效，己方回合开始时清零。敌人本回合没有攻击意图时，不要单纯为了格挡而出防御牌；只有它能立刻转化为伤害、用于消耗/弃牌等引擎，或明确保留到下一回合时才值得。
8. hp_loss 是牌面立即失去的生命，格挡无法抵消；retaliation_damage 是目标荆棘的格挡前伤害，需先由已有格挡吸收。结合出牌顺序计算生命与剩余格挡，避免自损或反伤致死；未知数值不能当0。
9. 按出牌顺序核对逐步实际费用、剩余能量及已知减费、回能；临时免费仅覆盖对应次数，后续费用不能沿用当前值。变化无法确认时停止计划并重新观察。先比较击杀、格挡与生命损失，再安排输出；换血须有明确的击杀、求生或节奏收益。
10. 选牌必须填写 selection_analysis 的 gap（本局当前缺口及依据）、contribution（此牌现在能提供什么、启动代价）、skip_comparison（与跳过相比是否值得）。不能只说稀有或某流派核心；历史快照不完整时说明不确定性。商店、休息与事件也应依据已有牌和后续威胁，而非机械血量阈值。
返回 JSON 对象，必须且只能包含：
{"decision_id":"输入值","candidate_id":"动作ID","reason":"理由","confidence":0.0,"alternative_ids":[],"evidence_ids":[],"warnings":[],"plan_candidate_ids":["动作ID"],"selection_analysis":{}}
confidence 范围 0 至 1；无值的数组用 []。"""


def build_system_prompt(*, precompute_attacks: bool = False) -> str:
    """Load the versioned project skill; installed Codex copies are independent."""
    packaged = Path(__file__).parent / 'skills/sts2-decider/SKILL.md'
    source = Path(__file__).resolve().parents[3] / 'skills/sts2-decider/SKILL.md'
    text = (packaged if packaged.is_file() else source).read_text(encoding='utf-8')
    if text.startswith('---\n'):
        text = text.split('---\n', 2)[2]
    tools_note = ('候选的 attack_result 已由只读计算器提前提供，本次没有可调用工具。'
                  if precompute_attacks else '请使用已提供的 function tools；')
    return (SYSTEM_PROMPT + '\n加载的决策 skill（sts2-decider）：\n' + text.strip()
            + '\n当前通过 API 决策，' + tools_note + '无 shell 或脚本执行能力。'
              '按上面的建议 JSON 格式返回，不执行游戏动作。')


def build_prompt(
    context: DecisionContext, evidence: list[Evidence], *, codex_cache: CodexCache | None = None,
    precompute_attacks: bool = False,
) -> str:
    payload = model_context_payload(context, codex_cache=codex_cache, precompute_attacks=precompute_attacks)
    payload["evidence"] = [
        {"id": item.evidence_id, "title": item.source_title, "text": item.text}
        for item in evidence
    ]
    payload["decision_id"] = context.decision_id
    comparison = ""
    if context.decision_type == "combat_next_action":
        comparison = ("战斗决策前比较至少两个本回合可行方案：分别考虑总费用、确定击杀的敌人、"
                      "敌人攻击与其他伤害导致的预计生命损失，以及后续收益；不能确定的数值标为未知。"
                      "抽牌、随机效果或状态变化后重新观察，不假设结果。reason 简短比较所选方案与一个备选方案的主要取舍。"
                      "例如：5格挡面对23伤仍可减少5点生命损失；攻击只有在击杀或其他效果阻止攻击时才减少本回合伤害。\n")
    return comparison + "输入 JSON：" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
