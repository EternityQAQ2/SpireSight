"""Read-only model tools, bound to a single observed decision. No SDK or game I/O."""
from __future__ import annotations

from dataclasses import asdict
import json

from sts2_advisor.contracts import DecisionContext
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.combat.simulation import initial_state, play, outcome, unsupported_mechanics
from sts2_advisor.game.combat.calculator import calculate


TOOL_SCHEMAS = [{"type": "function", "function": {
    "name": "calculate_attack",
    "description": "核对当前一张攻击牌对指定目标的直接伤害；不执行动作，不预测其他效果。手牌伤害已包含玩家力量等修正。",
    "parameters": {"type": "object", "properties": {
        "decision_id": {"type": "string", "description": "当前输入的 decision_id"},
        "candidate_id": {"type": "string", "description": "当前攻击候选 ID"},
        "expected_hp_after": {"type": "integer", "minimum": 0, "description": "预计目标攻击后剩余生命"},
    }, "required": ["decision_id", "candidate_id", "expected_hp_after"], "additionalProperties": False},
}}]


class DecisionTools:
    def __init__(self, context: DecisionContext):
        self.context = context

    def call(self, name: str, arguments: str) -> dict:
        try:
            if name not in {"evaluate_combat_plan", "calculate_attack"}:
                raise ValueError("未知工具")
            if len(arguments) > 8192:
                raise ValueError("参数过长")
            args = json.loads(arguments)
            if name == "calculate_attack":
                if self.context.decision_type != "combat_next_action":
                    raise ValueError("仅支持战斗局面")
                if not isinstance(args, dict) or set(args) != {"decision_id", "candidate_id", "expected_hp_after"}:
                    raise ValueError("参数必须包含 decision_id、candidate_id 和 expected_hp_after")
                if args["decision_id"] != self.context.decision_id:
                    raise ValueError("过期 decision_id")
                if not isinstance(args["candidate_id"], str) or type(args["expected_hp_after"]) is not int or args["expected_hp_after"] < 0:
                    raise ValueError("攻击参数类型错误")
                return calculate(self.context, args["candidate_id"], args["expected_hp_after"])
            if not isinstance(args, dict) or set(args) != {"decision_id", "candidate_ids"}:
                raise ValueError("需要 decision_id 和 candidate_ids")
            if args["decision_id"] != self.context.decision_id:
                raise ValueError("过期 decision_id")
            ids = args["candidate_ids"]
            if not isinstance(ids, list) or not 1 <= len(ids) <= 8 or not all(isinstance(i, str) for i in ids):
                raise ValueError("candidate_ids 应为 1 至 8 个动作 ID")
            return self.evaluate(ids)
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}

    def evaluate(self, ids: list[str]) -> dict:
        ctx = self.context
        if ctx.decision_type != "combat_next_action":
            return {"ok": False, "error": "仅支持战斗局面"}
        check = check_plan(ctx, ids)
        result = {"ok": check.error is None, "decision_id": ctx.decision_id,
                  "check": asdict(check), "projection": None, "exact": False}
        state = initial_state(ctx)
        if check.error or state is None:
            result["limitation"] = check.error or "局面包含尚未支持的机制，未预测数值"
            if state is None:
                result["unsupported_mechanics"] = unsupported_mechanics(ctx)
            return result
        before = state
        candidates = {c.candidate_id: c for c in ctx.candidates}
        cards = {c["index"]: c for c in ctx.player.get("hand", [])}
        trace = []
        for ident in check.candidate_ids:
            candidate = candidates[ident]
            if candidate.action_type == "end_turn":
                break
            if candidate.action_type != "play_card":
                result["limitation"] = "尚未支持该动作效果"
                return result
            nxt = play(state, candidate, cards[candidate.params["card_index"]])
            if nxt is None:
                result["limitation"] = "尚未支持该卡牌效果"
                return result
            state = nxt
            trace.append({"candidate_id": ident, "energy_left": state.energy,
                          "block": state.block, "hp": state.hp})
            if state.boundary:
                result.update(trace=trace, limitation="抽牌或未知结果之后需重新观测；未预测回合最终结果")
                return result
        result.update(exact=True, trace=trace, projection=outcome(state, before).to_dict(),
                      scope="假设此序列之后结束回合，仅覆盖已支持机制，不代表长期最优")
        return result
