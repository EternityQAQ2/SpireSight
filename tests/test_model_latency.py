import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

from sts2_advisor.decision.tools import DecisionTools
from sts2_advisor.game.state import build_decision_context
from sts2_advisor.integrations.models.remote import OpenAIAdvisor


def raw_scene():
    return json.loads(Path('tests/fixtures/combat_state.json').read_text(encoding='utf-8'))


class ModelLatencyTests(unittest.TestCase):
    def test_experiment_budget_and_descriptive_distribution(self):
        from evals.compare_model_latency import validate_budget, percentile
        validate_budget(250000, 80)
        for tokens, requests in [(250001, 80), (250000, 81), (0, 80)]:
            with self.assertRaises(ValueError): validate_budget(tokens, requests)
        self.assertEqual(percentile([1, 2, 3, 4, 5], .5), 3)
        self.assertEqual(percentile([1, 2, 3, 4, 5], .95), 4.8)
        self.assertIsNone(percentile([], .95))
    def backend(self, create, **kwargs):
        class Client:
            def __init__(self, **options):
                self.chat = NS(completions=NS(create=create))
        with patch.dict(sys.modules, {'openai': NS(OpenAI=Client)}):
            return OpenAIAdvisor('deepseek-flash', api_key='test', base_url='https://api.deepseek.com', **kwargs)

    def response(self, ctx):
        return NS(choices=[NS(finish_reason='stop', message=NS(content=json.dumps({
            'decision_id': ctx.decision_id, 'candidate_id': ctx.candidates[0].candidate_id,
            'reason': 'test', 'confidence': .8}), tool_calls=None))], usage=None)

    def test_thinking_uses_exact_configured_cap_and_low_on_every_request(self):
        ctx = build_decision_context(raw_scene())
        for cap in [1024, 2048, 4096]:
            calls = []
            backend = self.backend(lambda **request: (calls.append(request), self.response(ctx))[1],
                                   thinking='enabled', max_tokens=cap, reasoning_effort='low')
            backend.advise(ctx, [])
            self.assertEqual(calls[0]['max_tokens'], cap)
            self.assertEqual(calls[0]['reasoning_effort'], 'low')

    def test_truncated_reasoning_never_silently_increases_budget_or_retries(self):
        ctx = build_decision_context(raw_scene())
        for thinking in ['disabled', 'enabled']:
            calls = []
            def create(**request):
                calls.append(request)
                return NS(choices=[NS(finish_reason='length', message=NS(
                    content='', reasoning_content='unfinished', tool_calls=None))], usage=None)
            backend = self.backend(create, thinking=thinking, max_tokens=2048)
            with self.assertRaisesRegex(ValueError, 'reasoning_content'): backend.advise(ctx, [])
            self.assertEqual([r['max_tokens'] for r in calls], [2048])

    def test_tool_roundtrip_preserves_low_reasoning_and_same_cap(self):
        ctx = build_decision_context(raw_scene()); calls = []
        def create(**request):
            calls.append(copy.deepcopy(request))
            if len(calls) == 1:
                return NS(choices=[NS(finish_reason='tool_calls', message=NS(
                    content=None, reasoning_content='checking', tool_calls=[NS(id='call-1', function=NS(
                        name='calculate_attack', arguments=json.dumps({
                            'decision_id': ctx.decision_id, 'candidate_id': 'card:0:NIBBIT_0'})))]))], usage=None)
            return self.response(ctx)
        self.backend(create, thinking='enabled', reasoning_effort='low', max_tokens=2048).advise(ctx, [])
        self.assertEqual([r['max_tokens'] for r in calls], [2048, 2048])
        self.assertEqual([r['reasoning_effort'] for r in calls], ['low', 'low'])
        self.assertEqual(calls[1]['messages'][-2]['reasoning_content'], 'checking')

    def test_all_unknown_attacks_hide_tool_and_disclose_limit_without_inventing_damage(self):
        raw = raw_scene(); raw['player']['relics'].append({'id': 'UNKNOWN_RELIC'})
        ctx = build_decision_context(raw); calls = []
        backend = self.backend(lambda **request: (calls.append(request), self.response(ctx))[1], thinking='disabled')
        backend.advise(ctx, [])
        self.assertNotIn('tools', calls[0])
        self.assertIn('无法核对', calls[0]['messages'][0]['content'])
        self.assertIn('未知不等于零', calls[0]['messages'][0]['content'])

    def test_partially_known_tool_remains_available_and_probe_does_not_mutate_state(self):
        raw = raw_scene()
        raw['player']['relics'].append({'id': 'PHIAL_HOLSTER'})
        raw['player']['hand'][1]['description'] = '随机造成伤害。'
        ctx = build_decision_context(raw); before = copy.deepcopy(ctx.to_dict())
        status = DecisionTools(ctx).attack_tool_availability()
        self.assertTrue(status['available'])
        self.assertEqual(status['supported_candidates'], 1)
        self.assertEqual(ctx.to_dict(), before)

    def test_no_legal_attack_means_no_tool(self):
        raw = raw_scene()
        for card in raw['player']['hand']:
            if card['type'] == 'Attack': card['can_play'] = False
        ctx = build_decision_context(raw)
        self.assertFalse(DecisionTools(ctx).attack_tool_availability()['available'])

    def test_calculator_input_error_is_unavailability_not_a_model_request_crash(self):
        ctx = build_decision_context(raw_scene())
        ctx.player['relics'] = None
        self.assertFalse(DecisionTools(ctx).attack_tool_availability()['available'])

    def test_damage_known_player_outcome_unknown_is_still_useful(self):
        raw = raw_scene(); raw['player']['relics'].append({'id': 'PHIAL_HOLSTER'})
        ctx = build_decision_context(raw)
        ctx.player['hp'] = None
        self.assertEqual(DecisionTools(ctx).call('calculate_attack', json.dumps({
            'decision_id': ctx.decision_id, 'candidate_id': 'card:0:NIBBIT_0'}))['player_outcome']['status'], 'unknown')
        self.assertTrue(DecisionTools(ctx).attack_tool_availability()['available'])
