from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from evals.compare_attack_delivery import replay_decision
from sts2_advisor.contracts import Advice
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.state import build_decision_context


def scene():
    raw = json.loads((Path(__file__).parent / 'fixtures/combat_state.json').read_text(encoding='utf-8'))
    raw['player'].update(hp=4, energy=2)
    strike, _, defend = raw['player']['hand']
    raw['player']['hand'] = [strike, {**defend, 'index': 1}, {**defend, 'index': 2}]
    raw['battle']['enemies'][0]['intents'][0].update(label='8', description='造成8点伤害。')
    return raw, build_decision_context(raw)


BAD = ['card:0:NIBBIT_0', 'combat:end_turn']
GOOD = ['card:0:NIBBIT_0', 'card:1', 'combat:end_turn']


class EndTurnSafetyTests(unittest.TestCase):
    def test_lethal_end_is_rejected_with_numeric_feedback_when_escape_is_verified(self):
        _, ctx = scene()
        before = copy.deepcopy(ctx.to_dict())
        result = check_plan(ctx, BAD)
        self.assertIsNotNone(result.error)
        self.assertIn('预计生命', result.error)
        self.assertIn('-4', result.error)
        self.assertEqual(ctx.to_dict(), before)

    def test_a_safe_attack_block_tradeoff_remains_models_choice(self):
        _, ctx = scene()
        self.assertIsNone(check_plan(ctx, GOOD).error)

    def test_offline_audit_separates_raw_forecast_and_runtime_rejection(self):
        from evals.replay_end_turn import audit_row
        raw, ctx = scene()
        result = audit_row({'raw_state': raw, 'context': ctx.to_dict(),
                            'advice': {'candidate_id': BAD[0], 'plan_candidate_ids': BAD}})
        self.assertEqual(result['forecast']['status'], 'retry')
        self.assertTrue(result['runtime_redecision'])

    def test_no_verified_escape_does_not_block_a_forced_loss(self):
        from sts2_advisor.game.combat.forecast import check_end_turn
        raw, _ = scene()
        raw['player']['energy'] = 0
        ctx = build_decision_context(raw)
        result = check_end_turn(ctx, ['combat:end_turn'])
        self.assertEqual(result['status'], 'lethal_without_verified_escape')
        self.assertIsNone(check_plan(ctx, ['combat:end_turn']).error)

    def test_killing_last_enemy_is_not_predicted_to_take_its_attack(self):
        from sts2_advisor.game.combat.forecast import project_end_turn
        raw, _ = scene()
        raw['battle']['enemies'][0]['hp'] = 6
        result = project_end_turn(build_decision_context(raw), BAD)
        self.assertEqual(result['status'], 'calculated')
        self.assertGreater(result['hp_after'], 0)
        self.assertEqual(result['incoming'], 0)

    def test_unknown_mechanics_missing_values_and_end_turn_cards_are_not_zero(self):
        from sts2_advisor.game.combat.forecast import project_end_turn
        for field in ('enemy_hp', 'enemy_thorns', 'negative_thorns', 'player_block', 'status_card', 'unknown_power'):
            with self.subTest(field=field):
                raw, _ = scene()
                if field == 'enemy_hp':
                    ctx = build_decision_context(raw)
                    ctx.situation['battle']['enemies'][0]['hp'] = None
                if field == 'enemy_thorns': raw['battle']['enemies'][0]['status'] = [{'id':'THORNS_POWER','amount':None}]
                if field == 'negative_thorns': raw['battle']['enemies'][0]['status'] = [{'id':'THORNS_POWER','amount':-1}]
                if field == 'player_block': raw['player']['block'] = None
                if field == 'status_card': raw['player']['hand'].append({'index':3,'id':'INFECTION','type':'Status','cost':'0','description':'回合结束时受到3点伤害。','can_play':False})
                if field == 'unknown_power': raw['player']['status'] = [{'id':'UNKNOWN_POWER','amount':1}]
                ctx = ctx if field == 'enemy_hp' else build_decision_context(raw)
                self.assertEqual(project_end_turn(ctx, BAD)['status'], 'unknown')

    def test_a_draw_boundary_is_not_used_as_the_verified_escape(self):
        from sts2_advisor.game.combat.forecast import check_end_turn
        raw, _ = scene()
        raw['player']['hand'] = raw['player']['hand'][:2]
        raw['player']['hand'][1].update(id='SHRUG_IT_OFF', name='耸肩无视', description='获得8点格挡。抽1张牌。')
        self.assertEqual(check_end_turn(build_decision_context(raw), BAD)['status'],
                         'lethal_without_verified_escape')

    def test_unparsed_attack_count_cannot_certify_a_surviving_path(self):
        from sts2_advisor.game.combat.forecast import check_end_turn
        raw, _ = scene()
        raw['player'].update(hp=1, energy=1)
        raw['player']['hand'] = [raw['player']['hand'][1]]
        raw['battle']['enemies'][0]['intents'][0].update(label='3×2', description='造成3点伤害两次。')
        ctx = build_decision_context(raw)
        self.assertEqual(check_end_turn(ctx, ['combat:end_turn'])['status'], 'unknown')
        self.assertIsNone(check_plan(ctx, ['combat:end_turn']).error)

    def test_drawing_is_an_observation_boundary_not_a_survival_proof(self):
        from sts2_advisor.game.combat.forecast import project_end_turn
        raw, _ = scene()
        raw['player']['hand'][1].update(id='SHRUG_IT_OFF', name='耸肩无视', description='获得8点格挡。抽1张牌。')
        result = project_end_turn(build_decision_context(raw), ['card:1','combat:end_turn'])
        self.assertEqual(result['status'], 'unknown')

    def test_open_plan_is_not_treated_as_end_turn(self):
        from sts2_advisor.game.combat.forecast import project_end_turn
        _, ctx = scene()
        self.assertEqual(project_end_turn(ctx, ['card:0:NIBBIT_0'])['status'], 'not_ending_turn')

    def test_service_redecides_once_then_accepts_models_own_safe_plan(self):
        raw, ctx = scene()
        class Backend:
            calls = 0
            feedback = ''
            def advise(self, c, ev):
                self.calls += 1
                return Advice(c.decision_id, BAD[0], '先攻击再结束', .8, plan_candidate_ids=BAD)
            def advise_with_feedback(self, c, ev, feedback):
                self.calls += 1
                self.feedback = feedback
                return Advice(c.decision_id, GOOD[0], '攻击后补一张防御', .8, plan_candidate_ids=GOOD)
        backend = Backend()
        result = replay_decision({'raw_state':raw,'context':ctx.to_dict(),'evidence':[]}, backend)
        self.assertIsNone(result['error'])
        self.assertEqual(backend.calls, 2)
        self.assertIn('预计生命', backend.feedback)
        self.assertEqual(result['advice']['plan_candidate_ids'], GOOD)
        self.assertEqual(len(result['proposals']), 2)

    def test_second_lethal_plan_stops_without_rule_takeover_or_third_request(self):
        raw, ctx = scene()
        class Backend:
            calls = 0
            def advise(self, c, ev):
                self.calls += 1
                return Advice(c.decision_id, BAD[0], '错误计划', .8, plan_candidate_ids=BAD)
        backend = Backend()
        result = replay_decision({'raw_state':raw,'context':ctx.to_dict(),'evidence':[]}, backend)
        self.assertEqual(backend.calls, 2)
        self.assertIsNone(result['advice'])
        self.assertIsNotNone(result['error'])


if __name__ == '__main__':
    unittest.main()
