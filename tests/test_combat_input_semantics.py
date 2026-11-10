from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from sts2_advisor import cli
from sts2_advisor.decision.context import model_context_payload
from sts2_advisor.game.state import build_decision_context


FIXTURE = Path(__file__).parent / 'fixtures/combat_state.json'


class CombatInputSemanticsTests(unittest.TestCase):
    def raw(self):
        raw = json.loads(FIXTURE.read_text(encoding='utf-8'))
        first = raw['battle']['enemies'][0]
        first.update(name='同名敌人', status=[{'id': 'THORNS_POWER', 'name': '荆棘', 'amount': 2}])
        other = copy.deepcopy(first)
        other.update(entity_id='OTHER_1', hp=4, status=[])
        raw['battle']['enemies'].append(other)
        return raw

    def test_same_named_enemies_are_bound_by_id_in_summary_and_actions(self):
        payload = model_context_payload(build_decision_context(self.raw()))
        self.assertEqual([e['entity_id'] for e in payload['combat']['enemies']], ['NIBBIT_0', 'OTHER_1'])
        actions = {c['id']: c for c in payload['candidates']}
        for ident in ('NIBBIT_0', 'OTHER_1'):
            self.assertEqual(actions['card:0:' + ident]['target_id'], ident)
            self.assertIn(ident, actions['card:0:' + ident]['label'])
        self.assertEqual(actions['card:0:NIBBIT_0']['retaliation_damage'], 2)
        self.assertEqual(actions['card:0:OTHER_1']['retaliation_damage'], 0)

    def test_direct_life_cost_and_retaliation_are_distinct(self):
        raw = self.raw()
        raw['player']['hand'][0].update(id='HEMOKINESIS', name='御血术',
                                       description='失去2点生命。 造成15点伤害。')
        payload = model_context_payload(build_decision_context(raw))
        actions = {c['id']: c for c in payload['candidates']}
        self.assertEqual(actions['card:0:NIBBIT_0']['hp_loss'], 2)
        self.assertEqual(actions['card:0:NIBBIT_0']['retaliation_damage'], 2)
        self.assertEqual(actions['card:0:OTHER_1']['hp_loss'], 2)
        self.assertEqual(actions['card:0:OTHER_1']['retaliation_damage'], 0)
        self.assertIn('无视格挡', payload['card_metrics_scope'])

    def test_unknown_life_cost_or_thorns_are_not_fabricated_zero(self):
        raw = self.raw()
        raw['player']['hand'][0]['description'] = '失去X点生命。造成6点伤害。'
        raw['battle']['enemies'][0]['status'][0]['amount'] = None
        payload = model_context_payload(build_decision_context(raw))
        attack = next(c for c in payload['candidates'] if c['id'] == 'card:0:NIBBIT_0')
        self.assertIsNone(attack['hp_loss'])
        self.assertIsNone(attack['retaliation_damage'])
        self.assertIsNone(payload['combat']['enemies'][0]['thorns'])
        self.assertFalse(payload['turn_search']['available'])
        self.assertNotIn('self_damage', attack)

    def test_x_attack_does_not_fabricate_one_hit_retaliation(self):
        raw = self.raw()
        raw['player']['hand'][0].update(id='WHIRLWIND', name='旋风斩', cost='X',
                                       target_type='AllEnemies',
                                       description='对所有敌人造成5点伤害X次。')
        payload = model_context_payload(build_decision_context(raw))
        attack = next(c for c in payload['candidates'] if c['id'] == 'card:0')
        self.assertIsNone(attack['retaliation_damage'])

    def test_missing_status_list_is_unknown_rather_than_no_thorns(self):
        raw = self.raw()
        raw['battle']['enemies'][0]['status'] = None
        payload = model_context_payload(build_decision_context(raw))
        self.assertIsNone(payload['combat']['enemies'][0]['thorns'])
        self.assertIsNone(payload['candidates'][0]['retaliation_damage'])

    def test_free_attack_marks_costs_current_only_and_requires_reobservation(self):
        raw = self.raw()
        raw['player']['status'] = [{'id': 'FREE_ATTACK_POWER', 'amount': 1}]
        for card in raw['player']['hand']:
            if card['type'] == 'Attack':
                card['cost'] = '0'
        before = copy.deepcopy(raw)
        payload = model_context_payload(build_decision_context(raw))
        self.assertEqual(payload['cost_scope']['free_attacks_remaining'], 1)
        self.assertEqual(payload['cost_scope']['basis'], 'current_state')
        self.assertTrue(payload['cost_scope']['reobserve_after_attack'])
        self.assertIsNone(payload['cost_scope']['later_attack_costs'])
        self.assertEqual(raw, before)

    def test_model_comparison_override_keeps_the_same_production_defaults(self):
        env = {'OPENAI_MODEL': 'deepseek-flash', 'OPENAI_API_KEY': 'test',
               'OPENAI_BASE_URL': 'https://api.deepseek.com', 'OPENAI_THINKING': 'disabled'}
        with patch.dict(os.environ, env, clear=True):
            backend = cli.create_remote_backend(model_name='deepseek-v4-pro')
        try:
            self.assertEqual(backend.model_name, 'deepseek-v4-pro')
            self.assertEqual(backend.max_tokens, 1024)
            self.assertEqual(backend.thinking, 'disabled')
            self.assertFalse(backend.adaptive_thinking)
        finally:
            backend.client.close()


if __name__ == '__main__':
    unittest.main()
