from __future__ import annotations

import json
from pathlib import Path
import unittest

from evals.compare_prompts import restore
from sts2_advisor.decision.context import model_context_payload


class DecisionQualityEvalTests(unittest.TestCase):
    def test_old_input_is_loaded_from_git_without_new_semantics(self):
        from evals.compare_decision_quality import load_baseline
        old = load_baseline()
        row = next(json.loads(s) for s in Path('evals/fixtures/attack_delivery_cases.jsonl').read_text(
            encoding='utf-8').splitlines() if json.loads(s)['case_id'] == 'thorn-tadpoles')
        context, evidence = restore(row)
        before = old.model_context_payload(context)
        after = model_context_payload(context)
        self.assertNotIn('card_metrics_scope', before)
        self.assertNotIn('entity_id', before['combat']['enemies'][0])
        self.assertNotIn('target_id', before['candidates'][0])
        self.assertIn('self_damage', before['candidates'][0])
        self.assertIn('hp_loss', after['candidates'][0])
        self.assertNotIn('self_damage', after['candidates'][0])
        self.assertNotIn('hp_loss 是牌面', old.build_system_prompt())

    def test_quality_summary_does_not_treat_a_wrong_reason_as_a_wrong_action(self):
        from evals.compare_decision_quality import summarize_reviews
        results = [{'case_id': 'one', 'repeat': 1, 'profile': 'new', 'advice': {'candidate_id': 'x'}},
                   {'case_id': 'two', 'repeat': 1, 'profile': 'new', 'advice': {'candidate_id': 'y'}},
                   {'case_id': 'three', 'repeat': 1, 'profile': 'new', 'advice': None}]
        reviews = [{'case_id': 'one', 'repeat': 1, 'profile': 'new',
                    'action_grade': 'acceptable', 'reason_errors': ['incorrect HP'], 'raw_plan_errors': []},
                   {'case_id': 'two', 'repeat': 1, 'profile': 'new',
                    'action_grade': 'unresolved', 'reason_errors': [], 'raw_plan_errors': ['bad suffix']}]
        summary = summarize_reviews(results, reviews)['new']
        self.assertEqual(summary['acceptable_actions'], 1)
        self.assertEqual(summary['explicit_action_mistakes'], 0)
        self.assertEqual(summary['unresolved_actions'], 1)
        self.assertEqual(summary['unreviewed'], 1)
        self.assertEqual(summary['answers_with_reason_errors'], 1)
        self.assertEqual(summary['answers_with_raw_plan_errors'], 1)
        self.assertNotIn('accuracy', summary)

    def test_budget_cancel_and_empty_reply_are_not_action_mistakes(self):
        from evals.compare_decision_quality import summarize_reviews
        results = [{'case_id': 'empty', 'repeat': 1, 'profile': 'new', 'advice': None},
                   {'case_id': 'budget', 'repeat': 1, 'profile': 'new', 'advice': None}]
        reviews = [{'case_id': 'empty', 'repeat': 1, 'profile': 'new',
                    'action_grade': 'no_response', 'reason_errors': [], 'raw_plan_errors': []},
                   {'case_id': 'budget', 'repeat': 1, 'profile': 'new',
                    'action_grade': 'not_sent', 'reason_errors': [], 'raw_plan_errors': []}]
        summary = summarize_reviews(results, reviews)['new']
        self.assertEqual(summary['no_responses'], 1)
        self.assertEqual(summary['not_sent'], 1)
        self.assertEqual(summary['explicit_action_mistakes'], 0)


if __name__ == '__main__':
    unittest.main()
