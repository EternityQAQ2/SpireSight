import copy
import json
from pathlib import Path
import unittest

from evals.compare_prompts import restore
from evals.compare_attack_delivery import replay_decision
from sts2_advisor.contracts import Advice
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.combat.resources import known_resource_error
from sts2_advisor.game.state import build_decision_context


class ResourcePrecheckTests(unittest.TestCase):
    def test_paid_comparison_cannot_exceed_authorized_half_million_budget(self):
        from evals.compare_resource_precheck import validate_budget
        validate_budget(500000, 96)
        for tokens, requests in ((500001, 96), (0, 96), (500000, 97), (500000, 0)):
            with self.assertRaises(ValueError): validate_budget(tokens, requests)

    def scene(self):
        raw = json.loads(Path('tests/fixtures/combat_state.json').read_text(encoding='utf-8'))
        raw['battle']['enemies'][0]['status'] = [{'id': 'SLOW_POWER', 'amount': 0}]
        return raw

    def recorded_case(self):
        return next(row for row in (json.loads(line) for line in
            Path('evals/fixtures/player_outcome_medium_cases.jsonl').read_text(encoding='utf-8').splitlines())
            if row['case_id'] == 'm07')

    def test_six_energy_plan_is_rejected_even_when_damage_simulation_is_unknown(self):
        context, _ = restore(self.recorded_case())
        ids = ['card:2:BYGONE_EFFIGY_0', 'card:3:BYGONE_EFFIGY_0',
               'card:4:BYGONE_EFFIGY_0', 'card:1:BYGONE_EFFIGY_0', 'combat:end_turn']
        error = check_plan(context, ids).error
        self.assertIsNotNone(error)
        self.assertIn('能量', error)
        self.assertIn('第4步', error)

    def test_valid_resources_do_not_certify_damage_or_extend_the_execution_prefix(self):
        context = build_decision_context(self.scene())
        result = check_plan(context, ['card:0:NIBBIT_0', 'card:2', 'combat:end_turn'])
        self.assertIsNone(result.error)
        self.assertEqual(result.candidate_ids, ['card:0:NIBBIT_0'])
        self.assertIsNotNone(result.boundary)

    def test_duplicate_card_is_provable_without_target_simulation(self):
        context = build_decision_context(self.scene())
        self.assertIn('重复', check_plan(context, ['card:0:NIBBIT_0', 'card:0:NIBBIT_0']).error)

    def test_unknown_gain_or_discount_stops_resource_inference(self):
        for ident, description in (
            ('BLOODLETTING', '失去3点生命。获得[ironclad_energy_icon.png][ironclad_energy_icon.png]。'),
            ('UNRELENTING', '造成18点伤害。你打出的下一张攻击牌耗能变为0。'),
            ('UNKNOWN', '改变后续能量费用。'),
        ):
            raw = self.scene()
            raw['player']['hand'][0].update(id=ident, cost='0', description=description)
            context = build_decision_context(raw)
            self.assertIsNone(known_resource_error(context, ['card:0:NIBBIT_0', 'card:1:NIBBIT_0', 'card:2', 'card:1:NIBBIT_0']))

    def test_draw_boundary_is_not_treated_as_fixed_cost_future(self):
        raw = self.scene()
        raw['player']['hand'][0].update(id='POMMEL_STRIKE', description='造成9点伤害。抽1张牌。')
        context = build_decision_context(raw)
        ids = ['card:0:NIBBIT_0', 'card:1:NIBBIT_0', 'card:2']
        self.assertIsNone(known_resource_error(context, ids))
        self.assertIsNone(check_plan(context, ids).error)

    def test_free_attack_and_energy_relics_remain_unknown(self):
        for variant in ('free', 'nunchaku', 'horn', 'enemy', 'player'):
            raw = self.scene()
            if variant == 'free':
                raw['player']['status'] = [{'id': 'FREE_ATTACK_POWER', 'amount': 1}]
                for card in raw['player']['hand']:
                    if card['type'] == 'Attack': card['cost'] = '0'
            elif variant in ('nunchaku', 'horn'):
                raw['player']['relics'].append({'id': 'NUNCHAKU' if variant == 'nunchaku' else 'GREMLIN_HORN', 'counter': 9})
            elif variant == 'enemy':
                raw['battle']['enemies'][0]['status'].append({'id':'UNKNOWN_ENERGY_TRIGGER', 'amount':1})
            else:
                raw['player']['status'] = [{'id':'UNKNOWN_ENERGY_TRIGGER', 'amount':1}]
            context = build_decision_context(raw)
            self.assertIsNone(known_resource_error(context, ['card:0:NIBBIT_0', 'card:1:NIBBIT_0', 'card:2']))

    def test_known_cost_modifier_is_not_added_twice(self):
        raw = self.scene()
        raw['player']['status'] = [{'id':'TANGLED_POWER', 'amount':1}]
        raw['player']['hand'][0]['cost'] = '2'
        context = build_decision_context(raw)
        self.assertIsNone(known_resource_error(context, ['card:0:NIBBIT_0', 'card:2']))

    def test_missing_or_unknown_values_are_not_zero(self):
        for field in ('energy', 'status', 'relics'):
            context = build_decision_context(self.scene())
            context.player[field] = None
            self.assertIsNone(known_resource_error(context, ['card:0:NIBBIT_0', 'card:1:NIBBIT_0', 'card:2']))
        raw = self.scene()
        raw['player']['hand'][0]['cost'] = 'X'
        self.assertIsNone(known_resource_error(build_decision_context(raw), ['card:0:NIBBIT_0', 'card:1:NIBBIT_0']))

    def test_stars_checked_only_with_known_balance(self):
        raw = self.scene()
        raw['player']['stars'] = 1
        raw['player']['hand'][0]['star_cost'] = 1
        raw['player']['hand'][2]['star_cost'] = 1
        context = build_decision_context(raw)
        self.assertIn('辉星', known_resource_error(context, ['card:0:NIBBIT_0', 'card:2']))
        context.player['stars'] = None
        self.assertIsNone(known_resource_error(context, ['card:0:NIBBIT_0', 'card:2']))

    def test_already_confirmed_battle_end_keeps_its_valid_prefix(self):
        raw = self.scene()
        raw['battle']['enemies'][0].update(hp=1, status=[])
        context = build_decision_context(raw)
        result = check_plan(context, ['card:0:NIBBIT_0', 'card:1:NIBBIT_0', 'card:2'])
        self.assertIsNone(result.error)
        self.assertEqual(result.candidate_ids, ['card:0:NIBBIT_0'])

    def test_new_error_uses_existing_one_redecision_without_tactical_fallback(self):
        row = self.recorded_case()
        ids = ['card:2:BYGONE_EFFIGY_0', 'card:3:BYGONE_EFFIGY_0',
               'card:4:BYGONE_EFFIGY_0', 'card:1:BYGONE_EFFIGY_0']

        class Backend:
            def __init__(self): self.feedback = []
            def advise(self, ctx, ev):
                return Advice(ctx.decision_id, ids[0], '测试原提案', .8, plan_candidate_ids=ids)
            def advise_with_feedback(self, ctx, ev, feedback):
                self.feedback.append(feedback)
                return Advice(ctx.decision_id, 'card:1:BYGONE_EFFIGY_0', '模型重选三费重锤', .8,
                              plan_candidate_ids=['card:1:BYGONE_EFFIGY_0'])

        backend = Backend()
        result = replay_decision(copy.deepcopy(row), backend)
        self.assertIsNone(result['error'])
        self.assertEqual(len(backend.feedback), 1)
        self.assertEqual(len(result['proposals']), 2)
        self.assertEqual(result['advice']['candidate_id'], 'card:1:BYGONE_EFFIGY_0')
