"""Offline coverage and one-action prediction error for a recorded session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_advisor.game.state import build_decision_context
from sts2_advisor.game.combat.simulation import initial_state, play


def evaluate(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    counts = {"combat_steps": 0, "supported_states": 0, "played_cards": 0,
              "compared_actions": 0, "enemy_hp_mismatches": 0, "energy_mismatches": 0,
              "block_mismatches": 0}
    mismatches = []
    for step, (before_row, after_row) in enumerate(zip(rows, rows[1:])):
        raw = before_row["raw_state"]
        if raw.get("state_type") not in {"monster", "elite", "boss"}:
            continue
        counts["combat_steps"] += 1
        context = build_decision_context(raw)
        if context is None:
            continue
        state = initial_state(context)
        if state is None:
            continue
        counts["supported_states"] += 1
        advice = before_row.get("advice") or {}
        candidate = next((c for c in context.candidates if c.candidate_id == advice.get("candidate_id")), None)
        if candidate is None or candidate.action_type != "play_card":
            continue
        counts["played_cards"] += 1
        card = next((c for c in context.player.get("hand", []) if c.get("index") == candidate.params.get("card_index")), None)
        predicted = play(state, candidate, card) if card else None
        observed = after_row["raw_state"]
        if predicted is None or observed.get("state_type") not in {"monster", "elite", "boss"}:
            continue
        if observed.get("run", {}).get("floor") != raw.get("run", {}).get("floor"):
            continue
        if observed.get("battle", {}).get("round") != raw.get("battle", {}).get("round"):
            continue
        enemy_hp = {e["entity_id"]: e["hp"] for e in observed.get("battle", {}).get("enemies", [])}
        counts["compared_actions"] += 1
        # The game reindexes entity_id after a kill, so compare the surviving HP multiset.
        hp_mismatch = sorted(e.hp for e in predicted.enemies if e.hp > 0) != sorted(enemy_hp.values())
        counts["enemy_hp_mismatches"] += int(hp_mismatch)
        if hp_mismatch:
            mismatches.append({"step": step, "card": card.get("id"),
                               "predicted": {e.entity_id: e.hp for e in predicted.enemies},
                               "observed": enemy_hp})
        counts["energy_mismatches"] += int(predicted.energy != observed.get("player", {}).get("energy"))
        counts["block_mismatches"] += int(predicted.block != observed.get("player", {}).get("block"))
    counts["enemy_hp_mismatch_details"] = mismatches
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.session), ensure_ascii=False, indent=2))
