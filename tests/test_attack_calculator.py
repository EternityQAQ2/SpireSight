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

    def test_calculator_no_longer_requires_a_model_prediction(self):
        ctx = build_decision_context(self.rows['group'])
        result = DecisionTools(ctx).call('calculate_attack', json.dumps({
            'decision_id': ctx.decision_id, 'candidate_id': 'card:2:WRIGGLER_3'}))
        self.assertTrue(result['ok'])
        self.assertEqual(result['target_hp_after'], 6)
        self.assertNotIn('matches', result)
        from sts2_advisor.decision.tools import TOOL_SCHEMAS
        params = TOOL_SCHEMAS[0]['function']['parameters']
        self.assertEqual(set(params['required']), {'decision_id', 'candidate_id'})
        self.assertNotIn('expected_hp_after', params['properties'])

    def test_recorded_zero_cost_kill_ignores_unrelated_relics_and_displayed_modifiers(self):
        from evals.compare_prompts import restore
        rows = [json.loads(s) for s in Path('evals/fixtures/prompt_cases.jsonl').read_text(encoding='utf-8').splitlines()]
        ctx, _ = restore(next(r for r in rows if r['case_id'] == 'zero-cost-kill'))
        result = self.call(ctx, 'card:2:SHRINKER_BEETLE_0', 0)
        self.assertEqual(result['status'], 'calculated')
        self.assertEqual(result['damage_per_hit'], 10)
        self.assertEqual(result['target_hp_after'], 0)
        self.assertEqual(result['target_block_after'], 0)

    def test_truly_unknown_player_or_relic_effect_still_returns_unknown(self):
        for source in ['status', 'relics']:
            raw = copy.deepcopy(self.rows['group'])
            raw['player'].setdefault(source, []).append({'id': 'UNVERIFIED_EFFECT', 'amount': 1})
            ctx = build_decision_context(raw)
            self.assertEqual(self.call(ctx, 'card:2:WRIGGLER_3', 6)['status'], 'unknown')

    def test_slippery_applies_only_to_hits_that_reach_hp(self):
        raw = copy.deepcopy(self.rows['group'])
        raw['player']['hand'][0]['description'] = '造成3点伤害4次。'
        raw['battle']['enemies'][0].update(hp=18, block=4,
            status=[{'id': 'SLIPPERY_POWER', 'amount': 1}, {'id': 'MINION_POWER', 'amount': 1}])
        ctx = build_decision_context(raw)
        result = self.call(ctx, 'card:0:WRIGGLER_0', 11)
        self.assertEqual(result['status'], 'calculated')
        self.assertEqual(result['target_hp_after'], 11)
        self.assertEqual(result['target_block_after'], 0)
        self.assertEqual(result['slippery_remaining'], 0)

    def test_missing_target_modifier_is_unknown_instead_of_zero(self):
        raw = copy.deepcopy(self.rows['group'])
        for amount in [None, '?']:
            raw['battle']['enemies'][0]['status'] = [{'id': 'VULNERABLE_POWER', 'amount': amount}]
            ctx = build_decision_context(raw)
            self.assertEqual(self.call(ctx, 'card:2:WRIGGLER_0', 7)['status'], 'unknown')

    def test_unknown_target_hp_or_block_is_not_zero(self):
        for field in ['hp', 'block']:
            ctx = build_decision_context(self.rows['group'])
            situation = copy.deepcopy(ctx.situation)
            situation['battle']['enemies'][0][field] = None
            ctx = replace(ctx, situation=situation)
            self.assertEqual(self.call(ctx, 'card:2:WRIGGLER_0', 0)['status'], 'unknown')

    def test_model_reason_is_not_automatically_checked_or_rewritten(self):
        ctx = build_decision_context(self.rows["group"])
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return NS(choices=[NS(message=NS(content=json.dumps({"candidate_id": "card:2:WRIGGLER_3",
                "reason": "剥夺配合打击可击杀目标", "confidence": 0.8})), finish_reason="stop")])
        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = NS(completions=NS(create=create))
        with patch.dict(sys.modules, {"openai": NS(OpenAI=FakeOpenAI)}):
            advice = OpenAIAdvisor("test", api_key="test").advise(ctx, [])
        self.assertEqual(len(requests), 1)
        self.assertEqual(advice.reason, "剥夺配合打击可击杀目标")
        self.assertEqual(advice.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
