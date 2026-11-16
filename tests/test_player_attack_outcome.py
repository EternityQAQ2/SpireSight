import copy
import json
from pathlib import Path
import unittest

from sts2_advisor.game.state import build_decision_context
from sts2_advisor.game.combat.calculator import calculate


class PlayerAttackOutcomeTests(unittest.TestCase):
    def scene(self):
        raw = json.loads((Path(__file__).parent / 'fixtures/combat_state.json').read_text(encoding='utf-8'))
        raw['player'].update(hp=66, block=5)
        raw['player']['hand'][0].update(id='HEMOKINESIS', name='御血术', description='失去2点生命。造成19点伤害。')
        return raw

    def test_life_payment_bypasses_block_and_leaves_target_calculation_unchanged(self):
        raw = self.scene()
        ctx = build_decision_context(raw)
        before = copy.deepcopy(ctx.to_dict())
        result = calculate(ctx, 'card:0:NIBBIT_0')
        self.assertEqual(result['target_hp_after'], 23)
        self.assertEqual(result['player_outcome']['status'], 'calculated')
        self.assertEqual(result['player_outcome']['hp_after'], 64)
        self.assertEqual(result['player_outcome']['block_after'], 5)
        self.assertEqual(ctx.to_dict(), before)

    def test_plain_attack_and_printed_block_gain_are_not_modified_twice(self):
        raw = self.scene()
        raw['player']['hand'][0].update(id='IRON_WAVE', description='获得8点格挡。造成7点伤害。')
        raw['player']['status'] = [{'id':'DEXTERITY_POWER','amount':3}]
        result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')['player_outcome']
        self.assertEqual(result['hp_after'], 66)
        self.assertEqual(result['block_after'], 13)

    def test_unknown_target_effect_has_explicit_unknown_player_values(self):
        raw = self.scene()
        raw['battle']['enemies'][0]['status'] = [{'id':'THORNS_POWER','amount':2}]
        result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['player_outcome']['status'], 'unknown')
        self.assertIsNone(result['player_outcome']['hp_after'])

    def test_unknown_player_values_and_additional_card_effect_are_not_guessed(self):
        for variant in ('hp', 'block', 'card'):
            raw = self.scene()
            if variant in ('hp', 'block'):raw['player'][variant] = None
            else:raw['player']['hand'][0]['description'] += '回复3点生命。'
            result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')
            self.assertEqual(result['player_outcome']['status'], 'unknown')
            self.assertIsNone(result['player_outcome']['hp_after'])

    def test_battle_end_and_other_enemies_unverified_triggers_are_unknown(self):
        raw = self.scene()
        raw['battle']['enemies'][0]['hp'] = 3
        result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')
        self.assertTrue(result['target_defeated'])
        self.assertEqual(result['player_outcome']['status'], 'unknown')
        raw = self.scene()
        other = copy.deepcopy(raw['battle']['enemies'][0])
        other.update(entity_id='OTHER_1', status=[{'id':'UNKNOWN_TRIGGER_POWER','amount':1}])
        raw['battle']['enemies'].append(other)
        self.assertEqual(calculate(build_decision_context(raw), 'card:0:NIBBIT_0')['player_outcome']['status'], 'unknown')

    def test_pain_and_unknown_life_payment_cannot_claim_exact_player_hp(self):
        raw = self.scene()
        raw['player']['hand'].append({'id':'PAIN','type':'Curse','name':'疼痛','description':'每当你打出一张牌，失去1点生命。','index':3,'cost':'-1','can_play':False})
        self.assertEqual(calculate(build_decision_context(raw), 'card:0:NIBBIT_0')['player_outcome']['status'], 'unknown')
        raw = self.scene()
        raw['player']['hand'][0]['description'] = '失去X点生命。造成19点伤害。'
        self.assertEqual(calculate(build_decision_context(raw), 'card:0:NIBBIT_0')['player_outcome']['status'], 'unknown')

    def test_lethal_life_payment_cannot_guarantee_that_the_attack_resolves(self):
        raw = self.scene()
        raw['player']['hp'] = 1
        result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('target_hp_after', result)

    def test_end_turn_infection_is_excluded_but_draw_effect_stays_unknown(self):
        raw = self.scene()
        raw['player']['hand'].append({'id':'INFECTION', 'type':'Status',
            'description':'不能被打出。 在你的回合结束时，如果这张牌在你的手牌中，你受到3点伤害。',
            'index':3, 'cost':'-1', 'can_play':False})
        result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')['player_outcome']
        self.assertEqual(result['status'], 'calculated')
        self.assertEqual(result['hp_after'], 64)
        raw['player']['hand'][0].update(id='POMMEL_STRIKE', description='造成9点伤害。抽1张牌。')
        self.assertEqual(calculate(build_decision_context(raw), 'card:0:NIBBIT_0')['player_outcome']['status'], 'unknown')

    def test_multi_hit_does_not_repeat_life_cost_or_change_block(self):
        raw = self.scene()
        raw['player']['hand'][0].update(id='TWIN_STRIKE', description='造成5点伤害两次。')
        result = calculate(build_decision_context(raw), 'card:0:NIBBIT_0')
        self.assertEqual(result['hits'], 2)
        self.assertEqual(result['target_hp_after'], 32)
        self.assertEqual(result['player_outcome']['hp_after'], 66)
        self.assertEqual(result['player_outcome']['block_after'], 5)


if __name__ == '__main__':
    unittest.main()
