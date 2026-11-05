"""Bounded turn search; ranking is a short-term heuristic."""
from __future__ import annotations

from sts2_advisor.contracts import DecisionContext
from sts2_advisor.game.combat.state import CombatState
from sts2_advisor.game.combat.effects import known_effect
from sts2_advisor.game.combat.simulation import initial_state, play, outcome


def rank_turns(context: DecisionContext, *, max_nodes: int = 2000, limit: int = 5, first_candidate: str | None = None) -> dict:
    """Enumerate supported deterministic sequences; expose coverage and truncation.

    Ranking is a one-turn heuristic (survival, HP loss, kills, damage), not an
    optimal game policy. Unsupported cards remain available to the model.
    """
    initial = initial_state(context)
    if context.decision_type != "combat_next_action" or initial is None:
        return {"available": False, "reason": "当前状态、遗物或机制尚未支持精确试算"}
    cards = {c.get("index"): c for c in context.player.get("hand", [])}
    actions = [(c, cards[c.params.get("card_index")]) for c in context.candidates
               if c.legal and c.action_type == "play_card" and c.params.get("card_index") in cards]
    supported = [(c, card) for c, card in actions if known_effect(card) and not card.get("star_cost")]
    unsupported = [c.candidate_id for c, card in actions if not known_effect(card) or card.get("star_cost")]
    rows: list[dict] = []
    nodes = 0
    end = next((c.candidate_id for c in context.candidates if c.action_type == "end_turn"), None)

    def visit(state: CombatState, path: list[str]) -> None:
        nonlocal nodes
        if nodes >= max_nodes:
            return
        nodes += 1
        result = outcome(state, initial).to_dict()
        plan = path + ([end] if end and result["enemies_remaining"] and not state.boundary else [])
        if plan and (first_candidate is None or path or end == first_candidate):
            rows.append({"candidate_ids": plan, "requires_observation": state.boundary, **result})
        if state.boundary or len(path) >= 7 or result["enemies_remaining"] == 0 or state.hp <= 0:
            return
        for c, card in supported:
            if not path and first_candidate is not None and c.candidate_id != first_candidate:
                continue
            nxt = play(state, c, card)
            if nxt is not None:
                visit(nxt, path + [c.candidate_id])

    visit(initial, [])
    rows.sort(key=lambda r: (r["hp_after"] <= 0, r["hp_loss"], -r["kills"], -r["damage"], len(r["candidate_ids"])))
    unique: list[dict] = []
    seen: set[tuple] = set()
    for row in rows:
        signature = tuple(row[k] for k in ("hp_after", "damage", "block", "kills", "energy_left"))
        if signature not in seen:
            unique.append(row)
            seen.add(signature)
        if len(unique) == limit:
            break
    return {"available": True, "scope": "仅当前回合；掉血优先的启发式排序，不代表长期最优",
            "unsupported_candidate_ids": unsupported, "nodes": nodes,
            "truncated": nodes >= max_nodes, "has_observation_paths": any(r["requires_observation"] for r in rows), "lines": unique}
