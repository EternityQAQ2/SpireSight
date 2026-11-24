import unittest

from evals.compare_decision_options import transform_request, load_legacy
from evals.compare_prompts import restore
import json
from pathlib import Path


class DecisionOptionsTests(unittest.TestCase):
    def test_experiment_transform_does_not_mutate_messages_or_game_state(self):
        request = {'messages': [{'role': 'system', 'content': 'original'}, {'role': 'user', 'content': 'state'}],
                   'model': 'test', 'max_tokens': 1600}
        changed = transform_request(request, compare=True)
        self.assertEqual(request['messages'][0]['content'], 'original')
        self.assertEqual(changed['messages'][1], request['messages'][1])
        self.assertIn('combat_comparison', changed['messages'][0]['content'])
        self.assertEqual(transform_request(request, old_system='frozen')['messages'][0]['content'], 'frozen')

    def test_legacy_and_new_tools_reproduce_old_unknown_and_new_result(self):
        rows = [json.loads(s) for s in Path('evals/fixtures/prompt_cases.jsonl').read_text(encoding='utf-8').splitlines()]
        context, _ = restore(next(r for r in rows if r['case_id'] == 'zero-cost-kill'))
        arguments = json.dumps({'decision_id': context.decision_id,
                                'candidate_id': 'card:2:SHRINKER_BEETLE_0', 'expected_hp_after': 0})
        legacy = load_legacy()
        from sts2_advisor.decision.tools import DecisionTools
        self.assertEqual(legacy.DecisionTools(context).call('calculate_attack', arguments)['status'], 'unknown')
        result = DecisionTools(context).call('calculate_attack', arguments)
        self.assertEqual(result['status'], 'calculated')
        self.assertEqual(result['target_hp_after'], 0)
