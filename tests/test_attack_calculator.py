import copy
import json
import unittest
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sts2_advisor.decision.tools import DecisionTools
from sts2_advisor.game.state import build_decision_context
from sts2_advisor.integrations.models.remote import OpenAIAdvisor


class AttackCalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((Path(__file__).parent / "fixtures/calculator_log_cases.json").read_text(encoding="utf-8"))

    def call(self, ctx, candidate, expected):
        return DecisionTools(ctx).call("calculate_attack", json.dumps({"decision_id": ctx.decision_id,
            "candidate_id": candidate, "expected_hp_after": expected}))

    def test_group_log_rejects_false_kill(self):
        ctx = build_decision_context(self.rows["group"])
        result = self.call(ctx, "card:2:WRIGGLER_3", 0)
        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["target_hp_after"], 6)
        self.assertFalse(result["matches"])
        actual = self.rows["group_after"]["battle"]["enemies"][-1]["hp"]
        self.assertEqual(result["target_hp_after"], actual)

    def test_elite_slow_and_vulnerable_match_observed_damage(self):
        ctx = build_decision_context(self.rows["elite"])
        result = self.call(ctx, "card:0:BYGONE_EFFIGY_0", 33)
        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["damage_per_hit"], 16)
        self.assertEqual(result["target_hp_after"], 33)
        self.assertTrue(result["matches"])

    def test_slow_value_missing_remains_unknown(self):
        raw = copy.deepcopy(self.rows["elite"])
        raw["battle"]["enemies"][0]["status"][0]["amount"] = None
        ctx = build_decision_context(raw)
        self.assertEqual(self.call(ctx, "card:0:BYGONE_EFFIGY_0", 33)["status"], "unknown")

    def test_four_hits_and_block(self):
        raw = copy.deepcopy(self.rows["group"])
        raw["player"]["hand"][0].update(description="造成3点伤害4次。", type="Attack")
        raw["battle"]["enemies"][0].update(hp=18, block=5)
        ctx = build_decision_context(raw)
        result = self.call(ctx, "card:0:WRIGGLER_0", 11)
        self.assertEqual(result["target_hp_after"], 11)
        self.assertEqual(result["target_block_after"], 0)
        self.assertTrue(result["matches"])

    def test_unverified_modifiers_are_not_accepted(self):
        ctx = build_decision_context(self.rows["group"])
        result = DecisionTools(ctx).call("calculate_attack", json.dumps({"decision_id": ctx.decision_id,
            "candidate_id": "card:2:WRIGGLER_3", "expected_hp_after": 0, "modifiers": []}))
        self.assertFalse(result["ok"])

    def test_vulnerable_modifiers_apply_per_hit(self):
        raw = copy.deepcopy(self.rows["group"])
        raw["battle"]["enemies"][0]["status"] = [{"id": "VULNERABLE_POWER", "amount": 2}]
        ctx = build_decision_context(raw)
        result = self.call(ctx, "card:1:WRIGGLER_0", 0)
        self.assertEqual(result["damage_per_hit"], 10)
        self.assertEqual(result["target_hp_after"], 0)
        self.assertEqual(result["hits"], 2)

    def test_automatic_check_retries_false_kill_once(self):
        ctx = build_decision_context(self.rows["group"])
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            payload = {"candidate_id": "card:2:WRIGGLER_3", "reason": "12伤可秒杀18血", "confidence": 0.8}
            if len(requests) > 1:
                payload = {"candidate_id": "card:0:WRIGGLER_0", "reason": "调整目标", "confidence": 0.5}
            return NS(choices=[NS(message=NS(content=json.dumps(payload)), finish_reason="stop")])
        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = NS(completions=NS(create=create))
        with patch.dict(sys.modules, {"openai": NS(OpenAI=FakeOpenAI)}):
            advice = OpenAIAdvisor("test", api_key="test").advise(ctx, [])
        self.assertEqual(len(requests), 2)
        self.assertIn("target_hp_after", requests[1]["messages"][-1]["content"])
        self.assertEqual(advice.candidate_id, "card:0:WRIGGLER_0")


if __name__ == "__main__":
    unittest.main()
