from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from sts2_advisor.domain import Advice
from sts2_advisor.model import RuleAdvisor
from sts2_advisor.normalize import build_decision_context
from sts2_advisor.validation import AdviceValidationError, validate_advice


FIXTURE = Path(__file__).parent / "fixtures" / "combat_state.json"


class NormalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.context = build_decision_context(self.raw)
        assert self.context is not None

    def test_generates_legal_combat_candidates(self) -> None:
        ids = {candidate.candidate_id for candidate in self.context.candidates}
        self.assertEqual(
            ids,
            {"card:0:NIBBIT_0", "card:1:NIBBIT_0", "card:2", "combat:end_turn"},
        )

    def test_compacts_duplicate_pile_cards(self) -> None:
        self.assertEqual(self.context.player["draw_pile"][0]["count"], 2)

    def test_decision_id_is_stable(self) -> None:
        again = build_decision_context(self.raw)
        assert again is not None
        self.assertEqual(self.context.decision_id, again.decision_id)

    def test_rule_backend_returns_valid_candidate(self) -> None:
        advice = RuleAdvisor().advise(self.context, [])
        validated = validate_advice(advice, self.context, [])
        self.assertIn(validated.candidate_id, {candidate.candidate_id for candidate in self.context.candidates})

    def test_rejects_invented_candidate(self) -> None:
        advice = Advice(self.context.decision_id, "card:999", "非法动作", 1.0)
        with self.assertRaises(AdviceValidationError):
            validate_advice(advice, self.context, [])

    def test_illegal_alternatives_are_dropped_with_warning(self) -> None:
        advice = Advice(
            self.context.decision_id, "card:2", "测试建议", 0.8,
            alternative_ids=["card:1", "card:not-real"],
        )
        validated = validate_advice(advice, self.context, [])
        self.assertEqual(validated.candidate_id, "card:2")
        self.assertNotIn("card:not-real", validated.alternative_ids)
        self.assertTrue(any("非法备选" in warning for warning in validated.warnings))

    def test_rule_backend_confirms_completed_selection(self) -> None:
        context = build_decision_context(
            {
                "state_type": "card_select",
                "run": {"act": 3, "floor": 40},
                "player": {},
                "card_select": {
                    "cards": [
                        {"index": 0, "name": "剑柄打击"},
                        {"index": 1, "name": "恶魔之焰"},
                    ],
                    "selected_cards": [{"index": 1, "name": "恶魔之焰"}],
                    "selected_count": 1,
                    "preview_showing": False,
                    "can_confirm": True,
                },
            }
        )
        assert context is not None
        advice = RuleAdvisor().advise(context, [])
        self.assertEqual(advice.candidate_id, "card_select:confirm")

    def test_removes_primary_and_same_label_alternatives(self) -> None:
        candidates = list(self.context.candidates)
        candidates[1] = replace(candidates[1], label=candidates[0].label)
        context = replace(self.context, candidates=candidates)
        advice = Advice(
            context.decision_id,
            "card:0:NIBBIT_0",
            "测试建议",
            1.0,
            alternative_ids=["card:0:NIBBIT_0", "card:1:NIBBIT_0", "card:2"],
        )
        validated = validate_advice(advice, context, [])
        self.assertEqual(validated.alternative_ids, ["card:2"])

    def test_open_shop_always_exposes_proceed_to_close_inventory(self) -> None:
        context = build_decision_context(
            {
                "state_type": "shop",
                "run": {"act": 1, "floor": 9},
                "player": {"gold": 33},
                "shop": {
                    "items": [],
                    "is_open": True,
                    "can_proceed": False,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["shop:proceed"])
        self.assertEqual(context.candidates[0].action_type, "proceed")

    def test_closed_shop_exposes_proceed_to_leave_room(self) -> None:
        context = build_decision_context(
            {
                "state_type": "shop",
                "run": {"act": 1, "floor": 9},
                "player": {"gold": 33},
                "shop": {
                    "items": [],
                    "is_open": False,
                    "can_proceed": True,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["shop:proceed"])
        self.assertEqual(context.candidates[0].action_type, "proceed")

    def test_rewards_expose_only_highest_index_for_deterministic_claim(self) -> None:
        context = build_decision_context(
            {
                "state_type": "rewards",
                "run": {"act": 2, "floor": 33},
                "player": {},
                "rewards": {
                    "items": [
                        {"index": 0, "type": "card", "description": "Add a card"},
                        {"index": 1, "type": "gold", "description": "25 Gold"},
                        {"index": 2, "type": "potion", "description": "Potion"},
                    ],
                    "can_proceed": True,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["reward:2"])
        self.assertEqual(context.candidates[0].action_type, "claim_reward")

    def test_empty_rewards_expose_only_proceed(self) -> None:
        context = build_decision_context(
            {
                "state_type": "rewards",
                "run": {"act": 2, "floor": 33},
                "player": {},
                "rewards": {"items": [], "can_proceed": True},
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["reward:proceed"])

    def test_card_reward_requires_choosing_a_card(self) -> None:
        context = build_decision_context(
            {
                "state_type": "card_reward",
                "run": {"act": 2, "floor": 33},
                "player": {},
                "card_reward": {
                    "cards": [
                        {"index": 0, "name": "Demon Fire"},
                        {"index": 1, "name": "Feed the Fire"},
                        {"index": 2, "name": "Unload"},
                    ],
                    "can_skip": True,
                },
            }
        )
        assert context is not None
        self.assertEqual(
            [candidate.candidate_id for candidate in context.candidates],
            ["card_reward:0", "card_reward:1", "card_reward:2"],
        )
        self.assertNotIn("card_reward:skip", [candidate.candidate_id for candidate in context.candidates])

    def test_full_potion_slots_skip_potion_reward_and_proceed(self) -> None:
        context = build_decision_context(
            {
                "state_type": "rewards",
                "run": {"act": 3, "floor": 36},
                "player": {
                    "max_potion_slots": 2,
                    "potions": [
                        {"slot": 0, "name": "Fire Potion"},
                        {"slot": 1, "name": "Weak Potion"},
                    ],
                },
                "rewards": {
                    "items": [
                        {
                            "index": 1,
                            "type": "potion",
                            "description": "Energy Potion",
                            "potion_name": "Energy Potion",
                        }
                    ],
                    "can_proceed": True,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["reward:proceed"])
        self.assertEqual(context.candidates[0].action_type, "proceed")

    def test_full_potion_slots_skip_potion_but_claim_other_rewards(self) -> None:
        context = build_decision_context(
            {
                "state_type": "rewards",
                "run": {"act": 3, "floor": 36},
                "player": {
                    "max_potion_slots": 2,
                    "potions": [
                        {"slot": 0, "name": "Fire Potion"},
                        {"slot": 1, "name": "Weak Potion"},
                    ],
                },
                "rewards": {
                    "items": [
                        {"index": 0, "type": "gold", "description": "25 Gold"},
                        {"index": 1, "type": "potion", "description": "Energy Potion"},
                    ],
                    "can_proceed": True,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["reward:0"])
        self.assertEqual(context.candidates[0].action_type, "claim_reward")

    def test_potion_reward_is_claimed_directly_when_slot_is_available(self) -> None:
        context = build_decision_context(
            {
                "state_type": "rewards",
                "run": {"act": 3, "floor": 36},
                "player": {
                    "max_potion_slots": 2,
                    "potions": [{"slot": 0, "name": "Fire Potion"}],
                },
                "rewards": {
                    "items": [{"index": 1, "type": "potion", "description": "Energy Potion"}],
                    "can_proceed": False,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["reward:1"])
        self.assertEqual(context.candidates[0].action_type, "claim_reward")

    def test_ready_upgrade_selection_only_exposes_confirm(self) -> None:
        context = build_decision_context(
            {
                "state_type": "hand_select",
                "run": {"act": 1, "floor": 2},
                "player": {},
                "hand_select": {
                    "mode": "upgrade_select",
                    "cards": [
                        {"index": 0, "name": "Strike"},
                        {"index": 1, "name": "Defend"},
                    ],
                    "selected_cards": [{"index": 0, "name": "Strike"}],
                    "can_confirm": True,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["hand_select:confirm"])
        self.assertEqual(context.candidates[0].action_type, "combat_confirm_selection")

    def test_ready_exhaust_selection_only_exposes_confirm(self) -> None:
        context = build_decision_context(
            {
                "state_type": "hand_select",
                "run": {"act": 1, "floor": 7},
                "player": {},
                "hand_select": {
                    "mode": "simple_select",
                    "prompt": "Choose a card to Exhaust.",
                    "cards": [
                        {"index": 0, "name": "Defend"},
                        {"index": 1, "name": "Headbutt"},
                        {"index": 2, "name": "Second Wind"},
                    ],
                    "selected_cards": [{"index": 0, "name": "Defend"}],
                    "can_confirm": True,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["hand_select:confirm"])
        self.assertEqual(context.candidates[0].action_type, "combat_confirm_selection")

    def test_incomplete_multi_hand_selection_still_exposes_cards(self) -> None:
        context = build_decision_context(
            {
                "state_type": "hand_select",
                "run": {"act": 1, "floor": 7},
                "player": {},
                "hand_select": {
                    "mode": "simple_select",
                    "prompt": "Choose 2 cards to Discard.",
                    "cards": [
                        {"index": 0, "name": "Defend"},
                        {"index": 1, "name": "Strike"},
                    ],
                    "selected_cards": [{"index": 0, "name": "Defend"}],
                    "can_confirm": False,
                },
            }
        )
        assert context is not None
        self.assertEqual(
            [candidate.candidate_id for candidate in context.candidates],
            ["hand_select:0", "hand_select:1"],
        )


    def test_card_multi_select_excludes_already_selected_cards(self) -> None:
        context = build_decision_context(
            {
                "state_type": "card_select",
                "run": {"act": 2, "floor": 21},
                "player": {},
                "card_select": {
                    "screen_type": "select",
                    "cards": [
                        {"index": 0, "name": "Shrug It Off"},
                        {"index": 1, "name": "Thunderclap"},
                        {"index": 2, "name": "Taunt"},
                    ],
                    "selected_cards": [{"index": 0, "name": "Shrug It Off"}],
                    "selected_count": 1,
                    "preview_showing": False,
                    "can_confirm": False,
                },
            }
        )
        assert context is not None
        self.assertEqual(
            [candidate.candidate_id for candidate in context.candidates],
            ["card_select:1", "card_select:2"],
        )

    def test_card_select_preview_only_exposes_confirm(self) -> None:
        context = build_decision_context(
            {
                "state_type": "card_select",
                "run": {"act": 2, "floor": 21},
                "player": {},
                "card_select": {
                    "cards": [{"index": 0, "name": "Shrug It Off"}],
                    "selected_cards": [{"index": 0, "name": "Shrug It Off"}],
                    "selected_count": 1,
                    "preview_showing": True,
                    "can_confirm": True,
                },
            }
        )
        assert context is not None
        self.assertEqual([candidate.candidate_id for candidate in context.candidates], ["card_select:confirm"])


if __name__ == "__main__":
    unittest.main()

