from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from sts2_advisor.domain import Advice, Candidate, DecisionContext
from sts2_advisor.normalize import build_decision_context
from sts2_advisor.planning import ActivePlan
from sts2_advisor.service import AdvisorService
from sts2_advisor.validation import AdviceValidationError, validate_advice


FIXTURE = Path(__file__).parent / "fixtures" / "combat_state.json"


class PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.context = build_decision_context(self.raw)
        assert self.context is not None
        self.playable = [item for item in self.context.candidates if item.action_type == "play_card"]
        self.first, self.second = self.playable[:2]

    def make_plan(self) -> ActivePlan:
        advice = Advice(
            self.context.decision_id,
            self.first.candidate_id,
            "先后打出两张牌",
            0.8,
            plan_candidate_ids=[self.first.candidate_id, self.second.candidate_id],
        )
        validated = validate_advice(advice, self.context, [])
        plan = ActivePlan.from_advice(validated, self.context, [])
        assert plan is not None
        return plan

    def state_after_first_card(self) -> dict:
        return self.state_after_cards([self.first])

    def state_after_cards(self, played_candidates: list[Candidate]) -> dict:
        raw = copy.deepcopy(self.raw)
        played_indices = {item.params["card_index"] for item in played_candidates}
        raw["player"]["hand"] = [
            card for card in raw["player"]["hand"] if card["index"] not in played_indices
        ]
        for index, card in enumerate(raw["player"]["hand"]):
            card["index"] = index
        raw["player"]["energy"] -= len(played_candidates)
        return raw

    def test_reuses_plan_and_rebinds_shifted_card_index(self) -> None:
        current = build_decision_context(self.state_after_first_card())
        assert current is not None
        advice = self.make_plan().advance(current)
        self.assertIsNotNone(advice)
        assert advice is not None
        candidate = next(item for item in current.candidates if item.candidate_id == advice.candidate_id)
        self.assertEqual(candidate.label, self.second.label)
        self.assertEqual(advice.plan_candidate_ids, [advice.candidate_id])

    def test_invalidates_plan_when_play_causes_an_extra_draw(self) -> None:
        raw = self.state_after_first_card()
        extra = copy.deepcopy(self.raw["player"]["draw_pile"][0])
        extra.update({"id": "DRAWN_CARD", "index": len(raw["player"]["hand"]), "can_play": True})
        raw["player"]["hand"].append(extra)
        current = build_decision_context(raw)
        assert current is not None
        self.assertIsNone(self.make_plan().advance(current))

    def test_recognizes_multiple_manually_completed_plan_steps(self) -> None:
        third = self.playable[2]
        advice = Advice(
            self.context.decision_id,
            self.first.candidate_id,
            "连续执行三步",
            0.8,
            plan_candidate_ids=[
                self.first.candidate_id, self.second.candidate_id, third.candidate_id,
            ],
        )
        validated = validate_advice(advice, self.context, [])
        plan = ActivePlan.from_advice(validated, self.context, [])
        assert plan is not None
        current = build_decision_context(self.state_after_cards([self.first, self.second]))
        assert current is not None
        cached = plan.advance(current)
        self.assertIsNotNone(cached)
        assert cached is not None
        candidate = next(item for item in current.candidates if item.candidate_id == cached.candidate_id)
        self.assertEqual(candidate.label, third.label)

    def test_rejects_plan_whose_first_step_differs_from_advice(self) -> None:
        advice = Advice(
            self.context.decision_id,
            self.first.candidate_id,
            "无效计划",
            0.8,
            plan_candidate_ids=[self.second.candidate_id, self.first.candidate_id],
        )
        with self.assertRaisesRegex(AdviceValidationError, "计划第一步"):
            validate_advice(advice, self.context, [])

    def test_service_uses_second_step_without_model_or_retrieval_call(self) -> None:
        states = [self.raw, self.state_after_first_card()]

        class Client:
            def get_singleplayer_state(self):
                return states.pop(0)

        class Retriever:
            calls = 0

            def retrieve(self, context, top_k=6):
                self.calls += 1
                return []

        first_id = self.first.candidate_id
        second_id = self.second.candidate_id

        class Backend:
            calls = 0

            def advise(self, context, evidence):
                self.calls += 1
                return Advice(
                    context.decision_id,
                    first_id,
                    "短计划测试",
                    0.8,
                    plan_candidate_ids=[first_id, second_id],
                )

        class Recorder:
            def record(self, *args):
                pass

        retriever = Retriever()
        backend = Backend()
        output: list[str] = []
        service = AdvisorService(
            Client(), retriever, backend, Recorder(),
            backend_name="test", output=output.append,
        )
        self.assertTrue(service.run_once())
        self.assertTrue(service.run_once())
        self.assertEqual(backend.calls, 1)
        self.assertEqual(retriever.calls, 1)
        self.assertTrue(any("未请求模型" in line for line in output))

    def test_shop_plan_requires_expected_purchase_and_gold_change(self) -> None:
        first = Candidate("shop:0", "shop_purchase", "卡牌甲", {"index": 0})
        second = Candidate("shop:1", "shop_purchase", "卡牌乙", {"index": 1})
        previous = DecisionContext(
            "1.0", "before", "shop", "shop", {}, {"gold": 100},
            {"shop": {"items": [{"index": 0, "price": 30}, {"index": 1, "price": 40}]}},
            [first, second],
        )
        advice = Advice(
            "before", "shop:0", "依次购买", 0.8,
            plan_candidate_ids=["shop:0", "shop:1"],
        )
        plan = ActivePlan.from_advice(advice, previous, [])
        assert plan is not None
        current = DecisionContext(
            "1.0", "after", "shop", "shop", {}, {"gold": 70},
            {"shop": {"items": [{"index": 0, "price": 40}]}},
            [Candidate("shop:0", "shop_purchase", "卡牌乙", {"index": 0})],
        )
        cached = plan.advance(current)
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(cached.candidate_id, "shop:0")

        bad_plan = ActivePlan.from_advice(advice, previous, [])
        assert bad_plan is not None
        unexpected_gold = DecisionContext(
            "1.0", "bad", "shop", "shop", {}, {"gold": 80},
            current.situation, current.candidates,
        )
        self.assertIsNone(bad_plan.advance(unexpected_gold))


if __name__ == "__main__":
    unittest.main()
