from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from sts2_advisor.domain import Advice
from sts2_advisor.model import RuleAdvisor
from sts2_advisor.normalize import build_decision_context
from sts2_advisor.validation import AdviceValidationError, validate_advice


FIXTURE = Path(__file__).parent / "fixtures" / "combat_state.json"


class NormalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.context = build_decision_context(self.raw)
        assert self.context is not None

    def test_generates_legal_combat_candidates(self) -> None:
        ids = {candidate.candidate_id for candidate in self.context.candidates}
        self.assertEqual(
            ids,
            {"card:0:NIBBIT_0", "card:1:NIBBIT_0", "card:2", "combat:end_turn"},
        )

    def test_compacts_duplicate_pile_cards(self) -> None:
        self.assertEqual(self.context.player["draw_pile"][0]["count"], 2)

    def test_decision_id_is_stable(self) -> None:
        again = build_decision_context(self.raw)
        assert again is not None
        self.assertEqual(self.context.decision_id, again.decision_id)

    def test_rule_backend_returns_valid_candidate(self) -> None:
        advice = RuleAdvisor().advise(self.context, [])
        validated = validate_advice(advice, self.context, [])
        self.assertIn(validated.candidate_id, {candidate.candidate_id for candidate in self.context.candidates})

    def test_rejects_invented_candidate(self) -> None:
        advice = Advice(self.context.decision_id, "card:999", "非法动作", 1.0)
        with self.assertRaises(AdviceValidationError):
            validate_advice(advice, self.context, [])

    def test_removes_primary_and_same_label_alternatives(self) -> None:
        candidates = list(self.context.candidates)
        candidates[1] = replace(candidates[1], label=candidates[0].label)
        context = replace(self.context, candidates=candidates)
        advice = Advice(
            context.decision_id,
            "card:0:NIBBIT_0",
            "测试建议",
            1.0,
            alternative_ids=["card:0:NIBBIT_0", "card:1:NIBBIT_0", "card:2"],
        )
        validated = validate_advice(advice, context, [])
        self.assertEqual(validated.alternative_ids, ["card:2"])


if __name__ == "__main__":
    unittest.main()

