"""Simple deterministic fallback backend."""
from __future__ import annotations

from sts2_advisor.contracts import Advice, DecisionContext, Evidence
from sts2_advisor.game.combat.facts import incoming_damage


class RuleAdvisor:
    """Deterministic fallback used for pipeline validation and model failures."""

    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice:
        candidate = choose_rule_candidate(context)
        alternatives = [item.candidate_id for item in context.candidates if item != candidate][:2]
        return Advice(
            decision_id=context.decision_id,
            candidate_id=candidate.candidate_id,
            reason=rule_reason(context, candidate.label),
            confidence=0.45,
            alternative_ids=alternatives,
            evidence_ids=[item.evidence_id for item in evidence[:2]],
            warnings=["规则后端仅用于验证流程，不代表高水平策略。"],
            plan_candidate_ids=[candidate.candidate_id],
            selection_analysis={
                "gap": "规则回退无法可靠评估当前牌组缺口",
                "contribution": "按现有合法候选回退，收益尚未评估",
                "skip_comparison": "未完成与跳过的策略比较，请降低对此建议的信任",
            } if context.decision_type == "card_reward" else {},
        )


def choose_rule_candidate(context: DecisionContext):
    # 选择类界面出现 confirm 候选说明当前选择已可提交；回退必须直接确认，
    # 否则会盲目再选一张卡，把多选界面一路推向全选（附魔事件曾因此卡死）。
    confirm = next(
        (
            candidate
            for candidate in context.candidates
            if candidate.action_type in {"confirm_selection", "combat_confirm_selection"}
        ),
        None,
    )
    if confirm is not None:
        return confirm
    if context.decision_type == "combat_next_action":
        player = context.player
        incoming = incoming_damage(context.situation)
        uncovered = max(0, incoming - int(player.get("block", 0)))
        if uncovered:
            defense = next(
                (candidate for candidate in context.candidates if "防御" in candidate.label or "格挡" in candidate.label),
                None,
            )
            if defense:
                return defense
        attack = next((candidate for candidate in context.candidates if candidate.action_type == "play_card"), None)
        if attack:
            return attack
    non_exit = next(
        (
            candidate
            for candidate in context.candidates
            if candidate.action_type not in {"proceed", "end_turn", "skip_card_reward", "skip_relic_selection"}
        ),
        None,
    )
    return non_exit or context.candidates[0]


def rule_reason(context: DecisionContext, label: str) -> str:
    if context.decision_type == "combat_next_action":
        damage = incoming_damage(context.situation)
        return f"当前敌方预计造成 {damage} 点伤害；规则后端选择：{label}。"
    return f"规则后端从当前合法候选中选择：{label}。"
