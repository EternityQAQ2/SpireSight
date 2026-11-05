"""Override only a proven fatal first action when a certified surviving line exists."""
from dataclasses import replace

from sts2_advisor.contracts import Advice, DecisionContext
from sts2_advisor.game.combat.search import rank_turns


def avoid_proven_death(advice: Advice, context: DecisionContext) -> Advice:
    if context.decision_type != "combat_next_action":
        return advice
    forced = rank_turns(context, first_candidate=advice.candidate_id)
    if (not forced.get("available") or not forced.get("lines") or forced["truncated"]
            or forced["unsupported_candidate_ids"] or forced["has_observation_paths"]
            or any(c.action_type not in {"play_card", "end_turn"} for c in context.candidates)
            or forced["lines"][0]["hp_after"] > 0):
        return advice
    alternatives = rank_turns(context)
    safe = next((r for r in alternatives.get("lines", [])
                 if r["hp_after"] > 0 and not r["requires_observation"]), None)
    if safe is None:
        return advice
    return replace(advice, candidate_id=safe["candidate_ids"][0], plan_candidate_ids=safe["candidate_ids"],
                   reason=f"当前动作的已覆盖后续均致死；改用可活方案，预计剩余生命{safe['hp_after']}。",
                   confidence=min(advice.confidence, 0.6),
                   warnings=["生存检查替换明确致死动作；只保证已支持的当前回合", *advice.warnings])
