from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
import unittest

from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.combat.search import rank_turns
from sts2_advisor.game.deck import DeckTracker, inventory
from sts2_advisor.decision.policy import repair_plan_if_current_action_is_legal
from sts2_advisor.contracts import Advice, Candidate
from sts2_advisor.game.state import build_decision_context
from sts2_advisor.decision.context import CodexCache, model_context_payload
from sts2_advisor.decision.validation import AdviceValidationError, validate_advice


ROOT = Path(__file__).resolve().parents[1]
CASES = {r["case_id"]: r for r in
         (json.loads(line) for line in (ROOT / "evals/fixtures/log_cases.jsonl").read_text(encoding="utf-8").splitlines())}


def combat():
    raw = copy.deepcopy(CASES["overbudget"]["raw_state"])
    context = build_decision_context(raw)
    assert context is not None
    return raw, context


class CombatEngineTests(unittest.TestCase):
    def test_simulation_does_not_mutate_observation_or_previous_state(self):
        from sts2_advisor.game.combat.simulation import initial_state, play
        from sts2_advisor.game.combat.state import CombatState
        raw, context = combat()
        before = copy.deepcopy(context.to_dict())
        state = initial_state(context)
        self.assertIsInstance(state, CombatState)
        previous = copy.deepcopy(state)
        candidate = context.candidates[0]
        card = next(c for c in context.player["hand"] if c["index"] == candidate.params["card_index"])
        after = play(state, candidate, card)
        self.assertEqual(state, previous)
        self.assertEqual(context.to_dict(), before)
        self.assertEqual(after.energy, previous.energy - 1)

    def test_codex_caches_are_isolated_between_backend_sessions(self):
        _, context = combat()
        first_cache, second_cache = CodexCache(), CodexCache()
        original = model_context_payload(context, codex_cache=first_cache)
        changed_player = copy.deepcopy(context.player)
        changed_player["hand"][0]["description"] = "造成99点伤害。"
        changed = model_context_payload(replace(context, player=changed_player), codex_cache=second_cache)
        first_strike = next(c for c in original["stable"]["deck"] if c["k"] == "打击")
        second_strike = next(c for c in changed["stable"]["deck"] if c["k"] == "打击")
        self.assertEqual(first_strike["description"], "造成6点伤害。")
        self.assertEqual(second_strike["description"], "造成99点伤害。")
    def test_frozen_replay_reports_original_errors_and_missing_predictions(self):
        from evals.replay import evaluate
        result = evaluate(list(CASES.values()))
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["plan_errors"], 1)
        self.assertEqual(result["selection_explanation_errors"], 1)
        self.assertTrue(all(r["repaired_plan_error"] is None for r in result["cases"]))
        missing = evaluate(list(CASES.values()), {})
        self.assertEqual(missing["coverage"], 0)
        self.assertTrue(all(r["missing_prediction"] for r in missing["cases"]))

    def test_logged_five_energy_plan_is_rejected_and_repaired(self):
        _, context = combat()
        historical = CASES["overbudget"]["historical_advice"]
        advice = Advice(**historical)
        self.assertIn("费用", check_plan(context, advice.plan_candidate_ids).error)
        with self.assertRaisesRegex(AdviceValidationError, "费用"):
            validate_advice(replace(advice, evidence_ids=[]), context, [])
        repaired, reason = repair_plan_if_current_action_is_legal(advice, context)
        self.assertIsNotNone(reason)
        self.assertIsNone(check_plan(context, repaired.plan_candidate_ids).error)
        search = rank_turns(context)
        self.assertEqual(search["lines"][0]["hp_loss"], 7)
        self.assertEqual(search["lines"][0]["damage"], 12)
        self.assertEqual(search["lines"][0]["energy_left"], 0)

    def test_same_card_cannot_be_played_on_two_targets(self):
        raw, _ = combat()
        extra = copy.deepcopy(raw["battle"]["enemies"][0])
        extra["entity_id"] = "SECOND"
        raw["battle"]["enemies"].append(extra)
        context = build_decision_context(raw)
        ids = [c.candidate_id for c in context.candidates if c.params.get("card_index") == 0]
        self.assertIn("重复", check_plan(context, ids).error)

    def test_kill_removes_remaining_attack_and_stops_dead_target_plan(self):
        raw, _ = combat()
        raw["battle"]["enemies"][0]["hp"] = 6
        context = build_decision_context(raw)
        ids = [c.candidate_id for c in context.candidates if c.params.get("card_index") in (0, 1)]
        checked = check_plan(context, ids)
        self.assertEqual(len(checked.candidate_ids), 1)
        self.assertIn("战斗结束", checked.boundary)
        line = rank_turns(context)["lines"][0]
        self.assertEqual(line["incoming"], 0)
        self.assertEqual(line["hp_loss"], 0)

    def test_unknown_draw_effect_is_observation_boundary(self):
        raw, _ = combat()
        raw["player"]["hand"][0].update(id="SHRUG_IT_OFF", name="耸肩无视", type="Skill",
                                          description="获得8点格挡。抽1张牌。", target_type="Self")
        context = build_decision_context(raw)
        checked = check_plan(context, ["card:0", "combat:end_turn"])
        self.assertEqual(checked.candidate_ids, ["card:0"])
        self.assertIsNotNone(checked.boundary)

    def test_unknown_relic_disables_forecast_and_multi_step_certification(self):
        raw, _ = combat()
        raw["player"]["relics"].append({"id": "UNKNOWN"})
        context = build_decision_context(raw)
        self.assertFalse(rank_turns(context)["available"])
        checked = check_plan(context, [context.candidates[0].candidate_id, "combat:end_turn"])
        self.assertEqual(len(checked.candidate_ids), 1)
        self.assertIsNotNone(checked.boundary)

    def test_search_has_node_budget(self):
        _, context = combat()
        result = rank_turns(context, max_nodes=2)
        self.assertEqual(result["nodes"], 2)
        self.assertTrue(result["truncated"])

    def test_unknown_attack_does_not_claim_known_zero_damage(self):
        raw, _ = combat()
        raw["battle"]["enemies"][0]["intents"] = [{"type": "Attack", "label": "?", "description": "未知攻击"}]
        payload = model_context_payload(build_decision_context(raw))
        self.assertFalse(payload["combat"]["incoming_damage_known"])
        self.assertFalse(payload["turn_search"]["available"])

    def test_thorns_reduce_block_before_enemy_attack(self):
        raw, _ = combat()
        raw["player"]["hp"] = 3
        raw["battle"]["enemies"][0]["status"] = [{"id": "THORNS_POWER", "amount": 4}]
        context = build_decision_context(raw)
        attack = context.candidates[0].candidate_id
        self.assertIn("反伤致死", check_plan(context, [attack]).error)
        checked = check_plan(context, ["card:2", attack, "combat:end_turn"])
        self.assertIsNone(checked.error)


    def test_bash_vulnerability_and_artifact_use_step_order(self):
        raw, _ = combat()
        bash = {"index": 0, "id": "BASH", "name": "痛击", "cost": "2", "type": "Attack",
                "description": "造成8点伤害。给予2层易伤。", "target_type": "AnyEnemy", "can_play": True}
        strike = copy.deepcopy(raw["player"]["hand"][0]); strike["index"] = 1
        raw["player"]["hand"] = [bash, strike]
        result = rank_turns(build_decision_context(raw))
        self.assertEqual(max(r["damage"] for r in result["lines"]), 17)
        raw["battle"]["enemies"][0]["status"] = [{"id": "ARTIFACT_POWER", "amount": 1}]
        result = rank_turns(build_decision_context(raw))
        self.assertEqual(max(r["damage"] for r in result["lines"]), 14)

    def test_attack_block_and_dynamic_description_override_frozen_codex(self):
        cache = CodexCache()
        raw = copy.deepcopy(CASES["body_slam_dynamic"]["raw_state"])
        context = build_decision_context(raw)
        first = model_context_payload(context, codex_cache=cache)
        iron = next(c for c in first["candidates"] if c["id"].startswith("card:3:"))
        self.assertEqual(iron["block"], 5)
        raw["player"]["block"] = 9
        raw["player"]["hand"][0]["description"] = "造成你当前格挡值的伤害。 （造成9点伤害）"
        second = model_context_payload(build_decision_context(raw), codex_cache=cache)
        self.assertEqual(first["stable"], second["stable"])
        self.assertIn("9点", second["player"]["hand"][0]["description"])
        self.assertEqual(second["candidates"][0]["dmg"], 9)


class DeckStateTests(unittest.TestCase):
    def test_hand_and_all_piles_merge_by_name_and_upgrade_not_missing_id(self):
        rows = inventory([
            {"id": "STRIKE_IRONCLAD", "name": "打击", "is_upgraded": False},
            {"name": "打击"}, {"name": "打击+"},
            {"id": "STRIKE_IRONCLAD", "name": "打击+", "is_upgraded": True},
        ])
        self.assertEqual([(r["is_upgraded"], r["count"]) for r in rows], [(False, 2), (True, 2)])

    def test_inventory_reaches_noncombat_and_does_not_invent_empty_deck(self):
        raw, _ = combat()
        tracker = DeckTracker()
        tracker.observe(raw)
        raw = {"state_type": "rest_site", "run": raw["run"],
               "player": {"character": raw["player"]["character"]},
               "rest_site": {"options": [{"index": 0, "name": "休息"}]}}
        snapshot = tracker.observe(raw)
        context = build_decision_context(raw, deck_snapshot=snapshot)
        payload = model_context_payload(context)
        self.assertTrue(payload["player"]["deck_snapshot"]["cards"])
        self.assertTrue(snapshot["stale"])
        self.assertFalse(snapshot["complete"])
        raw["state_type"] = "menu"
        self.assertEqual(tracker.observe(raw)["cards"], [])

    def test_played_power_is_retained_and_generated_card_is_separate(self):
        raw, _ = combat()
        raw["player"]["hand"].append({"name": "燃烧", "id": "INFLAME", "index": 5,
                                        "type": "Power", "cost": "1", "description": "获得2点力量。"})
        tracker = DeckTracker(); first = tracker.observe(raw)
        # Repeat polling and auto-act precheck must produce an identical digest.
        self.assertEqual(first, tracker.observe(copy.deepcopy(raw)))
        raw["player"]["hand"].pop()
        raw["player"]["hand"].append({"name": "临时牌", "cost": "0"})
        snapshot = tracker.observe(raw)
        self.assertTrue(any(c.get("name") == "燃烧" for c in snapshot["cards"]))
        self.assertFalse(any(c.get("name") == "临时牌" for c in snapshot["cards"]))
        self.assertEqual(snapshot["combat_only_cards"][0]["name"], "临时牌")

    def test_confirmed_upgrade_changes_name_count_without_guessing_effect(self):
        raw, _ = combat()
        tracker = DeckTracker(); tracker.observe(raw)
        raw = {"state_type": "card_select", "run": raw["run"],
               "player": {"character": raw["player"]["character"]},
               "card_select": {"screen_type": "upgrade", "cards": [{"index": 0, "name": "打击"}]}}
        tracker.observe(raw)
        tracker.accepted_action(raw, Candidate("card_select:0", "select_card", "打击", {"index": 0}))
        raw["card_select"]["preview_showing"] = True
        tracker.accepted_action(raw, Candidate("card_select:confirm", "confirm_selection", "确认"))
        raw["state_type"] = "rest_site"
        snapshot = tracker.observe(raw)
        upgraded = next(c for c in snapshot["cards"] if c.get("name") == "打击+")
        self.assertTrue(upgraded["is_upgraded"])
        self.assertNotIn("description", upgraded)

    def test_accepted_reward_commits_once_after_state_change(self):
        raw, _ = combat()
        tracker = DeckTracker(); tracker.observe(raw)
        raw = {"state_type": "card_reward", "run": raw["run"],
               "player": {"character": raw["player"]["character"]},
               "card_reward": {"cards": [{"index": 1, "id": "INFLAME", "name": "燃烧"}]}}
        tracker.observe(raw)
        tracker.accepted_action(raw, Candidate("card_reward:1", "select_card_reward", "燃烧", {"card_index": 1}))
        self.assertFalse(any(c.get("name") == "燃烧" for c in tracker.observe(raw)["cards"]))
        raw["state_type"] = "rewards"
        tracker.observe(raw)
        self.assertEqual(next(c for c in tracker.observe(raw)["cards"] if c.get("name") == "燃烧")["count"], 1)

    def test_reward_requires_gap_and_skip_comparison(self):
        context = build_decision_context(CASES["missing_inventory"]["raw_state"])
        advice = Advice(**CASES["missing_inventory"]["historical_advice"])
        with self.assertRaisesRegex(AdviceValidationError, "选牌必须"):
            validate_advice(replace(advice, decision_id=context.decision_id, evidence_ids=[]), context, [])


if __name__ == "__main__":
    unittest.main()
