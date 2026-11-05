"""Policy for repairing model plans; orchestration does not implement combat rules."""
from __future__ import annotations

from dataclasses import replace

from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.combat.search import rank_turns
from sts2_advisor.contracts import Advice, DecisionContext


def repair_plan_if_current_action_is_legal(advice: Advice, context: DecisionContext) -> tuple[Advice, str | None]:
    candidates = {c.candidate_id for c in context.candidates if c.legal}
    if advice.candidate_id not in candidates:
        return advice, None
    plan = advice.plan_candidate_ids or [advice.candidate_id]
    if plan[0] != advice.candidate_id:
        reason = "plan_first_step_mismatch"
    elif set(plan) - candidates:
        reason = "illegal_plan_steps:" + ",".join(sorted(set(plan) - candidates))
    else:
        checked = check_plan(context, plan)
        if not checked.error:
            # Validation truncates at an observation boundary and reports why.
            return advice, None
        reason = checked.error
        search = rank_turns(context)
        # A supported alternative must cover the whole available action set;
        # never silently substitute a heuristic when a potion/unknown card exists.
        if (search.get("available") and search.get("lines")
                and not search["unsupported_candidate_ids"]
                and not any(c.action_type not in {"play_card", "end_turn"} for c in context.candidates)):
            line = search["lines"][0]
            return replace(advice, candidate_id=line["candidate_ids"][0],
                           plan_candidate_ids=line["candidate_ids"], confidence=min(advice.confidence, 0.6),
                           reason=f"原计划无法执行；采用本回合试算：预计掉血{line['hp_loss']}、输出{line['damage']}。",
                           warnings=[reason, "仅按当前回合掉血排序，未证明长期最优", *advice.warnings]), reason
    return replace(advice, plan_candidate_ids=[advice.candidate_id],
                   warnings=[*advice.warnings, "模型后续计划无效，已仅保留当前动作。"]), reason
