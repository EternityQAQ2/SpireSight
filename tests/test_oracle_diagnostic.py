import json
from pathlib import Path
import unittest


class OracleDiagnosticTests(unittest.TestCase):
    def test_menu_and_verified_groups_have_same_choices_but_only_one_has_outcomes(self):
        from evals.oracle_diagnostic import make_messages, make_cases
        case = make_cases()[0]
        plain = make_messages(case, 'plans_only', 1)[1]['content']
        oracle = make_messages(case, 'verified', 1)[1]['content']
        self.assertNotIn('verified_outcome', plain)
        self.assertIn('verified_outcome', oracle)
        self.assertNotIn('preferred_plan', oracle)
        self.assertNotIn(case['review_rubric'], oracle)

    def test_order_is_counterbalanced_without_changing_plan_identities(self):
        from evals.oracle_diagnostic import presented_plans, make_cases
        case = make_cases()[0]
        one, two = presented_plans(case, 1), presented_plans(case, 2)
        self.assertEqual(one[0]['candidate_ids'], two[1]['candidate_ids'])
        self.assertEqual(one[1]['candidate_ids'], two[0]['candidate_ids'])

    def test_grading_matches_whole_plan_and_does_not_grade_freeform_as_menu_failure(self):
        from evals.oracle_diagnostic import grade, make_cases
        case = make_cases()[2]
        self.assertEqual(grade(case, 'state_only', {'plan_candidate_ids':['x']})['selection'], 'manual_review')
        a = case['plans'][0]['candidate_ids']
        self.assertEqual(grade(case, 'verified', {'plan_candidate_ids':a})['selection'], 'preferred')
        self.assertEqual(grade(case, 'verified', {'plan_candidate_ids':a[:1]})['selection'], 'other_plan')


if __name__ == '__main__':
    unittest.main()
