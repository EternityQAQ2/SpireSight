import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from sts2_advisor.contracts import Advice, Evidence
from sts2_advisor.game.state import build_decision_context


def scene():
    return build_decision_context(json.loads(Path('tests/fixtures/combat_state.json').read_text(encoding='utf-8')))


class DecisionExampleTests(unittest.TestCase):
    def test_paid_comparison_has_one_shared_bounded_budget(self):
        from evals.compare_decision_improvements import validate_budget
        validate_budget(200000, 64)
        for tokens, requests in ((200001, 64), (0, 64), (200000, 65), (200000, 0)):
            with self.assertRaises(ValueError): validate_budget(tokens, requests)

    def test_json_fallback_upserts_reviewed_examples_and_filters_mechanisms(self):
        import tempfile
        from sts2_advisor.integrations.knowledge.ingest import main
        from sts2_advisor.integrations.knowledge.lexical import JsonKnowledgeRetriever
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'knowledge.jsonl'
            for _ in range(2):
                main(['data/decision_examples.jsonl', '--output', str(path)])
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 4)
            self.assertEqual(JsonKnowledgeRetriever(path).retrieve(scene(), 4), [])
            ctx = scene(); ctx.player['block'] = 5
            rows = JsonKnowledgeRetriever(path).retrieve(ctx, 4)
            # Example-only fallback has no guide slot, so must not evict facts.
            self.assertEqual(rows, [])

    def test_slippery_example_arithmetic_matches_verified_calculator(self):
        from sts2_advisor.game.combat.calculator import calculate
        raw = json.loads(Path('tests/fixtures/combat_state.json').read_text(encoding='utf-8'))
        raw['battle']['enemies'][0].update(hp=10, block=0, status=[{'id': 'SLIPPERY_POWER', 'amount': 2}])
        raw['player']['hand'][0].update(description='造成3点伤害4次。')
        result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')
        self.assertEqual(result['target_hp_after'], 2)
        self.assertEqual(result['slippery_remaining'], 0)

    def test_only_matching_reviewed_example_replaces_one_guide(self):
        from sts2_advisor.integrations.knowledge.examples import select_examples
        ctx = scene()
        guides = [Evidence('g1', '攻略一', '攻略'), Evidence('g2', '攻略二', '攻略')]
        def example(ident, mechanism, reviewed=True):
            return Evidence(ident, '局面→方案→算术→适用条件', '核实案例', metadata={
                'kind': 'verified_decision_example', 'mechanism': mechanism,
                'review_status': 'verified_arithmetic' if reviewed else 'unreviewed'})
        cases = [example('wrong', 'slippery'), example('unchecked', 'current_block', False),
                 example('block1', 'current_block'), example('block2', 'current_block')]
        self.assertEqual(select_examples(ctx, guides + cases, 2), guides)
        ctx.player['block'] = 4
        result = select_examples(ctx, guides + cases, 2)
        self.assertEqual([e.evidence_id for e in result], ['block1', 'g1'])
        self.assertEqual(select_examples(ctx, guides + cases, 0), [])
        # An example may replace a guide, never displace a SQLite fact when no
        # applicable guide was retrieved at all.
        self.assertEqual(select_examples(ctx, cases, 2), [])

    def test_chroma_queries_matching_mechanism_and_never_leaks_other_cases(self):
        from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
        ctx = scene(); ctx.player['block'] = 4
        retriever = ChromaRetriever.__new__(ChromaRetriever)
        retriever.collection = Mock()
        retriever.collection.count.return_value = 4
        retriever.collection.query.side_effect = [
            {'ids': [['g']], 'documents': [['原攻略']], 'metadatas': [[{}]]},
            {'ids': [['e']], 'documents': [['示例']], 'metadatas': [[{
                'kind': 'verified_decision_example', 'mechanism': 'current_block',
                'review_status': 'verified_arithmetic'}]]}]
        result = retriever.retrieve(ctx, 1)
        self.assertEqual([e.evidence_id for e in result], ['e'])
        self.assertIn('where', retriever.collection.query.call_args.kwargs)

    def test_no_applicable_guide_triggers_bounded_local_overfetch(self):
        from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
        retriever = ChromaRetriever.__new__(ChromaRetriever)
        retriever.collection = Mock()
        retriever.collection.count.return_value = 1000
        retriever.collection.query.side_effect = [
            {'ids': [['wrong']], 'documents': [['其他流派']],
             'metadatas': [[{'decision_types': 'card_reward'}]]},
            {'ids': [['right']], 'documents': [['当前通用规则']],
             'metadatas': [[{'decision_types': 'combat_next_action'}]]}]
        result = retriever.retrieve(scene(), 4)
        self.assertEqual([e.evidence_id for e in result], ['right'])
        self.assertEqual([call.kwargs['n_results'] for call in retriever.collection.query.call_args_list], [12, 48])

    def test_curated_examples_are_bounded_reviewed_and_have_source_encounters(self):
        from sts2_advisor.integrations.knowledge.examples import load_examples
        records = load_examples(Path('data/decision_examples.jsonl'))
        self.assertGreaterEqual(len(records), 3)
        for row in records:
            self.assertLessEqual(len(row['text'].encode('utf-8')), 900)
            self.assertEqual(row['metadata']['review_status'], 'verified_arithmetic')
            self.assertTrue(row['metadata']['source_encounter'])
            self.assertNotIn('card:', row['text'])


class RepairFeedbackTests(unittest.TestCase):
    def test_feedback_distinguishes_simulated_resources_from_unchanged_snapshot(self):
        from sts2_advisor.decision.feedback import build_repair_feedback
        ctx = scene(); ctx.player['energy'] = 2
        ids = ['card:0:NIBBIT_0', 'card:1:NIBBIT_0']
        proposal = Advice(ctx.decision_id, ids[0], '原计划', .8, plan_candidate_ids=ids)
        result = json.loads(build_repair_feedback(ctx, proposal, '计划费用不足'))
        self.assertFalse(result['actions_executed'])
        self.assertEqual(result['observed_energy'], 2)
        self.assertEqual(result['resource_trace'][0]['energy_after'], 1)
        self.assertEqual(result['resource_trace'][1]['status'], 'insufficient_energy')
        self.assertEqual(result['previous_plan'], ids)
        self.assertIn(['card:1:NIBBIT_0'], [r['candidate_ids'] for r in result['repair_options']])
        self.assertEqual(ctx.player['energy'], 2)

    def test_unknown_effect_stops_trace_and_cannot_certify_suffix(self):
        from sts2_advisor.decision.feedback import build_repair_feedback
        ctx = scene()
        ctx.player['hand'][0].update(id='UNKNOWN', description='随机获得能量。')
        ids = ['card:0:NIBBIT_0', 'card:1:NIBBIT_0', 'card:2']
        proposal = Advice(ctx.decision_id, ids[0], '原计划', .8, plan_candidate_ids=ids)
        result = json.loads(build_repair_feedback(ctx, proposal, '无效计划'))
        self.assertEqual(len(result['resource_trace']), 1)
        self.assertEqual(result['resource_trace'][0]['status'], 'observe_after_action')
        self.assertIsNone(result['resource_trace'][0]['energy_after'])
        self.assertTrue(all(len(r['candidate_ids']) <= 1 for r in result['repair_options']))

    def test_unknown_relics_produce_no_resource_claim_or_repair_certificate(self):
        from sts2_advisor.decision.feedback import build_repair_feedback
        ctx = scene(); ctx.player['relics'].append({'id': 'UNKNOWN_ENERGY_TRIGGER'})
        ids = ['card:0:NIBBIT_0', 'card:1:NIBBIT_0']
        proposal = Advice(ctx.decision_id, ids[0], '原计划', .8, plan_candidate_ids=ids)
        result = json.loads(build_repair_feedback(ctx, proposal, '无效计划'))
        self.assertEqual(result['resource_trace'], [])
        self.assertEqual(result['repair_options'], [])

    def test_service_passes_full_feedback_once_without_executing(self):
        from evals.compare_attack_delivery import replay_decision
        ctx = scene(); ctx.player['energy'] = 2
        raw = json.loads(Path('tests/fixtures/combat_state.json').read_text(encoding='utf-8'))
        ids = ['card:0:NIBBIT_0', 'card:1:NIBBIT_0']
        backend = Mock()
        backend.advise.return_value = Advice(ctx.decision_id, ids[0], '原计划', .8, plan_candidate_ids=ids)
        backend.advise_with_feedback.return_value = backend.advise.return_value
        result = replay_decision({'context': ctx.to_dict(), 'raw_state': raw, 'evidence': []}, backend)
        self.assertIsNotNone(result['error'])
        self.assertEqual(backend.advise_with_feedback.call_count, 1)
        self.assertFalse(json.loads(backend.advise_with_feedback.call_args.args[2])['actions_executed'])
        self.assertIsNone(result['advice'])
