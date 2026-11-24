import unittest
from pathlib import Path
import json

from sts2_advisor.decision.prompts import build_prompt, build_system_prompt
from sts2_advisor.game.state import build_decision_context


class CombatComparisonPromptTests(unittest.TestCase):
    def test_plans_are_optional_and_resources_follow_sequential_actual_costs(self):
        prompt = build_system_prompt()
        self.assertNotIn('应规划 2 至 8 步', prompt)
        self.assertNotIn('计划中所有出牌的费用总和不得超过当前能量', prompt)
        self.assertIn('不要求凑多步', prompt)
        self.assertIn('逐步实际费用', prompt)
        self.assertIn('减费', prompt)
        self.assertIn('回能', prompt)

    def test_system_loads_versioned_skill(self):
        prompt = build_system_prompt()
        self.assertIn('# STS2 Decider', prompt)
        self.assertIn('candidate_id', prompt)
        self.assertIn('Do not predict HP before calling', prompt)
        self.assertNotIn('description: Analyze', prompt)

    def test_combat_requests_comparison(self):
        rows = json.loads((Path(__file__).parent / 'fixtures/calculator_log_cases.json').read_text(encoding='utf-8'))
        prompt = build_prompt(build_decision_context(rows['group']), [])
        self.assertIn('比较至少两个', prompt)
        self.assertIn('不能确定', prompt)
