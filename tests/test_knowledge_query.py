import unittest

from sts2_advisor.contracts import Candidate, DecisionContext
from sts2_advisor.integrations.knowledge.common import build_query


def context(kind='combat_next_action', *, player=None, situation=None, candidates=None):
    return DecisionContext('1.0', 'test', kind, 'monster' if kind == 'combat_next_action' else kind,
                           {'act': 1}, {'character': '铁甲战士', **(player or {})},
                           situation or {}, candidates or [])


class KnowledgeQueryTests(unittest.TestCase):
    def test_combat_query_uses_current_hand_enemies_and_status_not_piles(self):
        ctx = context(player={
            'hand': [{'name': '双重打击', 'description': '造成7点伤害两次。', 'cost': '1'}],
            'status': [{'name': '虚弱', 'amount': 2}],
            'draw_pile': [{'name': '不属于当前手牌'}],
        }, situation={'battle': {'enemies': [{
            'name': '扭动虫', 'hp': 21, 'status': [{'name': '易伤', 'amount': 1}],
            'intents': [{'type': 'Attack', 'description': '重复意图原文' * 200}],
        }]}})
        query = build_query(ctx)
        for term in ['双重打击', '扭动虫', '易伤', '虚弱']:
            self.assertIn(term, query)
        self.assertNotIn('不属于当前手牌', query)
        self.assertNotIn('重复意图原文', query)
        self.assertLessEqual(len(query), 800)

    def test_reward_query_includes_candidates_and_observed_deck_with_provenance(self):
        ctx = context('card_reward', player={
            'deck_snapshot': {'cards': [{'name': '燃烧', 'is_upgraded': True, 'count': 2}],
                              'complete': False, 'stale': True},
            'draw_pile': [{'name': '战斗临时牌'}],
        }, situation={'card_reward': {'cards': [{'name': '恶魔之焰', 'description': '消耗所有手牌。'}]}},
            candidates=[Candidate('card_reward:0', 'choose_card', '恶魔之焰')])
        query = build_query(ctx)
        for term in ['恶魔之焰', '燃烧+', '不完整', '过时']:
            self.assertIn(term, query)
        self.assertNotIn('战斗临时牌', query)

    def test_duplicate_cards_do_not_repeat_query_and_keep_upgrade(self):
        card = {'name': '打击', 'description': '造成9点伤害。', 'is_upgraded': True, 'cost': '0'}
        ctx = context(player={'hand': [card] * 20})
        query = build_query(ctx)
        self.assertEqual(query.count('打击+'), 1)
        self.assertIn('0费', query)
        self.assertEqual(query, build_query(ctx))

    def test_other_screens_keep_options_without_serializing_map_graph(self):
        ctx = context('map', situation={'map': {'nodes': [[1, 2, 'monster', []]] * 500}},
                      candidates=[Candidate('map:1', 'choose_map_node', '前往精英')])
        query = build_query(ctx)
        self.assertIn('前往精英', query)
        self.assertNotIn('nodes', query)
        self.assertLessEqual(len(query), 800)

    def test_empty_state_and_unknown_cost_do_not_invent_deck_or_values(self):
        query = build_query(context(player={'hand': [{'name': '未知牌', 'cost': None}]}))
        self.assertIn('未知牌', query)
        self.assertNotIn('0费', query)
        self.assertNotIn('完整牌组', query)
