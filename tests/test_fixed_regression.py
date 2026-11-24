import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evals.replay_fixed_regression import PreExampleCollection, run, validate_budget


class FixedRegressionTests(unittest.TestCase):
    def test_paid_budget_bounds_checked_before_any_request(self):
        validate_budget(500000, 128)
        for tokens, requests in [(0, 128), (500001, 128), (500000, 129), (500000, 0)]:
            with self.assertRaises(ValueError): validate_budget(tokens, requests)

    def test_prior_output_is_not_overwritten_or_followed_by_api_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            output.write_text('existing', encoding='utf-8')
            args = SimpleNamespace(token_budget=500000, request_budget=128, output=output)
            with patch('evals.replay_fixed_regression.load_environment') as setup:
                with self.assertRaises(ValueError): run(args)
                setup.assert_not_called()
            self.assertEqual(output.read_text(encoding='utf-8'), 'existing')

    def test_baseline_collection_rejects_new_example_leak_without_mutating_store(self):
        collection = Mock()
        collection.query.return_value = {'metadatas': [[{'kind': 'verified_decision_example'}]]}
        with self.assertRaises(ValueError):
            PreExampleCollection(collection).query(query_texts=['state'], n_results=12)
        collection.upsert.assert_not_called()
        collection.delete.assert_not_called()
