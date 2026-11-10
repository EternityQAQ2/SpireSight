from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sts2_advisor import cli
from sts2_advisor.decision.context import model_context_payload
from sts2_advisor.game.state import build_decision_context
from sts2_advisor.integrations.models.remote import OpenAIAdvisor


FIXTURE = Path(__file__).parent / 'fixtures/combat_state.json'


class PrecomputedAttackTests(unittest.TestCase):
    def context(self, *, unknown=False):
        raw = json.loads(FIXTURE.read_text(encoding='utf-8'))
        raw['battle']['enemies'][0].update(hp=8, block=2)
        other = copy.deepcopy(raw['battle']['enemies'][0])
        other.update(entity_id='OTHER_1', hp=8, block=0,
                     status=[{'id': 'VULNERABLE', 'amount': 2}])
        raw['battle']['enemies'].append(other)
        if unknown:
            raw['player']['relics'].append({'id': 'UNSUPPORTED_RELIC'})
        return build_decision_context(raw)

    def test_inline_results_are_independent_per_target_and_leave_context_unchanged(self):
        context = self.context()
        before = context.to_dict()
        payload = model_context_payload(context, precompute_attacks=True)
        by_id = {c['id']: c for c in payload['candidates']}
        self.assertEqual(by_id['card:0:NIBBIT_0']['attack_result'],
                         {'status': 'calculated', 'hp_after': 4, 'killed': False})
        self.assertEqual(by_id['card:0:OTHER_1']['attack_result'],
                         {'status': 'calculated', 'hp_after': 0, 'killed': True})
        self.assertNotIn('attack_result', by_id['card:2'])
        self.assertEqual([c['id'] for c in payload['candidates']],
                         [c.candidate_id for c in context.candidates])
        self.assertIn('同一快照', payload['attack_result_scope'])
        self.assertEqual(context.to_dict(), before)

    def test_unknown_is_explicit_without_fabricated_hp(self):
        payload = model_context_payload(self.context(unknown=True), precompute_attacks=True)
        result = payload['candidates'][0]['attack_result']
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('hp_after', result)
        self.assertIn('UNSUPPORTED_RELIC', result['reason'])

    def test_existing_prompt_does_not_get_inline_fields_by_default(self):
        payload = model_context_payload(self.context())
        self.assertNotIn('attack_result_scope', payload)
        self.assertTrue(all('attack_result' not in c for c in payload['candidates']))

    def test_inline_backend_sends_one_request_without_registered_tools(self):
        context = self.context()
        calls = []

        def complete(**request):
            calls.append(request)
            advice = {'decision_id': context.decision_id, 'candidate_id': 'card:0:OTHER_1',
                      'reason': '击杀目标', 'confidence': 0.9,
                      'plan_candidate_ids': ['card:0:OTHER_1']}
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(advice)), finish_reason='stop')])

        backend = OpenAIAdvisor('test', api_key='test', thinking='disabled', precompute_attacks=True)
        with patch.object(backend.client.chat.completions, 'create', side_effect=complete):
            advice = backend.advise(context, [])
        backend.client.close()
        self.assertEqual(advice.candidate_id, 'card:0:OTHER_1')
        self.assertEqual(len(calls), 1)
        self.assertNotIn('tools', calls[0])
        self.assertNotIn('tool_choice', calls[0])
        self.assertIn('"hp_after":0', calls[0]['messages'][1]['content'])
        self.assertNotIn('请使用已提供的 function tools', calls[0]['messages'][0]['content'])

    def test_cli_factory_retains_production_config_and_allows_only_mode_override(self):
        config = {'OPENAI_MODEL': 'deepseek-flash', 'OPENAI_API_KEY': 'test',
                  'OPENAI_BASE_URL': 'https://api.deepseek.com',
                  'OPENAI_MAX_TOKENS': '1024', 'OPENAI_THINKING': 'disabled'}
        with patch.dict(os.environ, config, clear=True):
            normal = cli.create_remote_backend()
            inline = cli.create_remote_backend(precompute_attacks=True)
        try:
            self.assertFalse(normal.precompute_attacks)
            self.assertTrue(inline.precompute_attacks)
            for backend in (normal, inline):
                self.assertEqual(backend.model_name, 'deepseek-flash')
                self.assertEqual(backend.max_tokens, 1024)
                self.assertEqual(backend.thinking, 'disabled')
                self.assertFalse(backend.adaptive_thinking)
        finally:
            normal.client.close()
            inline.client.close()


if __name__ == '__main__':
    unittest.main()
