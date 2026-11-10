import unittest
from pathlib import Path
import json

from evals.compare_prompts import restore
from sts2_advisor.integrations.knowledge.relevance import rank_facts
from sts2_advisor.contracts import Evidence


class RelevanceTests(unittest.TestCase):
    def test_mixed_pickup_and_ongoing_relic_is_not_demoted_as_pickup_only(self):
        rows = [json.loads(s) for s in Path('evals/fixtures/prompt_cases.jsonl').read_text(encoding='utf-8').splitlines()]
        context, _ = restore(rows[0])
        facts = [Evidence('pickup', '原文', '来源', metadata={'table': 'relics', 'description': '拾起时，获得金币。'}),
                 Evidence('ongoing', '原文', '来源', metadata={'table': 'relics', 'description': '拾起时获得金币。每回合获得1点能量。'})]
        ranked = rank_facts(context, facts)
        self.assertEqual([e.evidence_id for e in ranked], ['ongoing', 'pickup'])
        self.assertEqual(ranked[0].text, '原文')
        self.assertNotIn('priority', facts[0].metadata)

    def test_chroma_query_changes_but_existing_metadata_filter_remains(self):
        from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
        rows = [json.loads(s) for s in Path('evals/fixtures/prompt_cases.jsonl').read_text(encoding='utf-8').splitlines()]
        context, _ = restore(rows[0])
        retriever = ChromaRetriever.__new__(ChromaRetriever)
        class Collection:
            def count(self): return 2
            def query(self, **kwargs):
                self.query_text = kwargs['query_texts'][0]
                return {'ids': [['wrong', 'right']], 'documents': [['错误场景', '正确场景']],
                        'metadatas': [[{'decision_types': 'card_reward'}, {'decision_types': 'combat_next_action'}]]}
        retriever.collection = Collection()
        result = retriever.retrieve(context, 2)
        self.assertEqual([e.evidence_id for e in result], ['right'])
        self.assertIn('当前手牌', retriever.collection.query_text)
