from __future__ import annotations

import json
from pathlib import Path
import unittest

from sts2_advisor.contracts import Advice, Evidence
from sts2_advisor.game.state import build_decision_context


class AlignedReplayTests(unittest.TestCase):
    def test_replay_uses_service_top_k_and_keeps_rejected_proposal_before_one_retry(self):
        from evals.compare_attack_delivery import replay_decision

        raw = json.loads((Path(__file__).parent / 'fixtures/combat_state.json').read_text(encoding='utf-8'))
        context = build_decision_context(raw)
        evidence = Evidence('test', 'facts', 'source')

        class Retriever:
            calls = []

            def retrieve(self, ctx, top_k=6):
                self.calls.append(top_k)
                return [evidence]

        class Backend:
            feedbacks = []

            def advise(self, ctx, ev):
                return Advice(ctx.decision_id, 'invalid', 'invalid first proposal', 0.8,
                              plan_candidate_ids=['invalid'])

            def advise_with_feedback(self, ctx, ev, feedback):
                self.feedbacks.append(feedback)
                return Advice(ctx.decision_id, 'card:0:NIBBIT_0', 'valid second proposal', 0.8,
                              plan_candidate_ids=['card:0:NIBBIT_0'])

        backend, retriever = Backend(), Retriever()
        result = replay_decision({'raw_state': raw, 'context': context.to_dict(),
                                  'evidence': []}, backend, retriever=retriever)
        self.assertIsNone(result['error'])
        self.assertEqual(retriever.calls, [4])
        self.assertEqual(len(backend.feedbacks), 1)
        self.assertEqual([p['candidate_id'] for p in result['proposals']],
                         ['invalid', 'card:0:NIBBIT_0'])
        self.assertEqual(result['advice']['candidate_id'], 'card:0:NIBBIT_0')
        self.assertEqual(result['evidence'], [evidence.to_dict()])
        self.assertTrue(any(e['type'] == 'action.skipped' for e in result['application_events']))

    def test_replay_stops_after_second_invalid_proposal(self):
        from evals.compare_attack_delivery import replay_decision

        raw = json.loads((Path(__file__).parent / 'fixtures/combat_state.json').read_text(encoding='utf-8'))
        context = build_decision_context(raw)

        class InvalidBackend:
            calls = 0

            def advise(self, ctx, ev):
                self.calls += 1
                return Advice(ctx.decision_id, 'invalid', 'invalid', 0.8)

        backend = InvalidBackend()
        result = replay_decision({'raw_state': raw, 'context': context.to_dict(), 'evidence': []}, backend)
        self.assertEqual(backend.calls, 2)
        self.assertIsNone(result['advice'])
        self.assertIsNotNone(result['error'])
        self.assertEqual(len(result['proposals']), 2)


if __name__ == '__main__':
    unittest.main()
