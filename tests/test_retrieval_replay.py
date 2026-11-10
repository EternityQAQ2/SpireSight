import unittest

from evals.replay_retrieval_decisions import paired_rows


class RetrievalReplayTests(unittest.TestCase):
    def test_pairs_keep_context_identical_and_do_not_include_rubrics(self):
        fixtures = [{'case_id': 'x', 'split': 'heldout', 'context': {'decision_id': 'same'}, 'rubric': 'secret'}]
        retrieval = {'cases': [{'case_id': 'x', 'before': {'evidence': [{'evidence_id': 'old'}]},
                               'after': {'evidence': [{'evidence_id': 'new'}]}}]}
        pairs = paired_rows(fixtures, retrieval, ['x'])
        self.assertEqual(pairs[0]['context'], pairs[1]['context'])
        self.assertNotIn('rubric', pairs[0])
        self.assertEqual(pairs[0]['retrieval_version'], 'before')
        self.assertEqual(pairs[1]['evidence'], [{'evidence_id': 'new'}])

    def test_missing_or_duplicate_cases_fail_before_api_calls(self):
        with self.assertRaises(ValueError):
            paired_rows([], {'cases': []}, ['missing'])
        fixtures = [{'case_id': 'x'}] * 2
        with self.assertRaises(ValueError):
            paired_rows(fixtures, {'cases': []}, [])
