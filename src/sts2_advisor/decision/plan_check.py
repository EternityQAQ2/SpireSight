"""Certify sequential resources, card consumption, targets and observation boundaries."""
from __future__ import annotations

from sts2_advisor.contracts import DecisionContext
from sts2_advisor.contracts import PlanCheck
from sts2_advisor.game.combat.effects import energy_cost, energy_gain, known_effect
from sts2_advisor.game.combat.simulation import initial_state, play


def check_plan(context: DecisionContext, ids: list[str]) -> PlanCheck:
    """Walk resource use even without effect support. Stop after unknown effects.

    A prefix crossing draw/random/selection cannot be certified just by adding
    static costs; return only up to that action and require a fresh observation.
    """
    if context.decision_type != "combat_next_action":
        return PlanCheck(ids)
    candidates = {c.candidate_id: c for c in context.candidates if c.legal}
    cards = {c.get("index"): c for c in context.player.get("hand", [])}
    energy = int(context.player.get("energy") or 0)
    stars = int(context.player.get("stars") or 0)
    used: set[int] = set()
    prefix: list[str] = []
    state = initial_state(context)
    for pos, ident in enumerate(ids):
        candidate = candidates.get(ident)
        if candidate is None:
            return PlanCheck(prefix, "计划包含不可用动作")
        if candidate.action_type == "end_turn":
            if pos != len(ids) - 1:
                return PlanCheck(prefix, "结束回合后仍有计划动作")
            prefix.append(ident)
            break
        if candidate.action_type != "play_card":
            return PlanCheck(prefix + [ident], boundary="非出牌效果需要重新观测")
        index = candidate.params.get("card_index")
        card = cards.get(index)
        if card is None or index in used:
            return PlanCheck(prefix, "同一张手牌被重复使用或已经消耗")
        cost = energy_cost(card, energy)
        star_cost = card.get("star_cost")
        if star_cost is not None and not str(star_cost).isdigit():
            return PlanCheck(prefix + [ident], boundary="辉星费用未知")
        if cost is None:
            return PlanCheck(prefix + [ident], boundary="能量费用未知")
        if cost > energy or int(star_cost or 0) > stars:
            return PlanCheck(prefix, "计划逐步费用超过可用能量或辉星")
        energy -= cost
        stars -= int(star_cost or 0)
        used.add(index)
        if state is not None:
            if candidate.params.get("target") and not any(
                e.entity_id == candidate.params["target"] and e.hp > 0 for e in state.enemies
            ):
                return PlanCheck(prefix, "计划目标已死亡或不存在")
        prefix.append(ident)
        effect = known_effect(card)
        if effect is None or cost == 0 and str(card.get("cost", "")).upper() == "X":
            return PlanCheck(prefix, boundary="未支持或不确定效果，执行后重新观测")
        if state is not None:
            state = play(state, candidate, card)
            if state is None:
                return PlanCheck(prefix[:-1], "无法验证目标或资源变化")
            if state.boundary:
                return PlanCheck(prefix, boundary="抽牌或遗物触发后重新观测")
            if state.hp <= 0:
                return PlanCheck(prefix[:-1], "计划导致反伤致死")
            if not any(e.hp > 0 for e in state.enemies):
                return PlanCheck(prefix, boundary="战斗结束，后续动作无效" if pos < len(ids) - 1 else None)
        elif len(ids) > 1:
            return PlanCheck(prefix, boundary="当前机制无法精确试算，执行一步后重新观测")
        energy += energy_gain(card)
    return PlanCheck(prefix)
