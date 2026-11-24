import copy
import unittest

from evals.medium_player_outcome import schedule, validate_limits, validate_suite
from evals.compare_prompts import digest
import json


class MediumOutcomeEvalTests(unittest.TestCase):
    def fixtures(self):
        manifest = {'cases': [{'case_id': f'm{i:02}', 'source': f's{i}.jsonl',
                               'floor': 1, 'decision_id': str(i)} for i in range(32)]}
        rows = [{'case_id': c['case_id'], 'source': c['source'],
                 'context': {'decision_id': c['decision_id'], 'run': {'floor': 1}}}
                for c in manifest['cases']]
        for row in rows:
            row['context_sha256'] = digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True))
        return manifest, rows

    def test_hard_budget_ceiling_and_positive_limits(self):
        validate_limits(2000000, 512)
        for tokens, requests in ((2000001, 512), (0, 512), (2000000, 513), (2000000, 0)):
            with self.assertRaises(ValueError):
                validate_limits(tokens, requests)

    def test_counterbalanced_128_decisions_and_pairs(self):
        _, rows = self.fixtures()
        trials = list(schedule(rows))
        self.assertEqual(len(trials), 128)
        for case in rows:
            first = [t[2] for t in trials if t[1]['case_id'] == case['case_id']]
            self.assertEqual(sorted(first), ['new', 'new', 'old', 'old'])
            self.assertEqual(first[:2], first[2:][::-1])

    def test_duplicate_encounters_or_changed_fixture_cannot_start_paid_run(self):
        manifest, rows = self.fixtures()
        validate_suite(manifest, rows)
        changed = copy.deepcopy(rows)
        changed[0]['context']['decision_id'] = 'different'
        with self.assertRaises(ValueError):
            validate_suite(manifest, changed)
        changed = copy.deepcopy(rows)
        changed[0]['context']['player'] = {'hp': 999}
        with self.assertRaises(ValueError):
            validate_suite(manifest, changed)
        duplicate = copy.deepcopy(manifest)
        duplicate['cases'][1]['source'] = duplicate['cases'][0]['source']
        with self.assertRaises(ValueError):
            validate_suite(duplicate, rows)
