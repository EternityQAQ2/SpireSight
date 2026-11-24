import json
from pathlib import Path

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sts2_advisor.contracts import Advice, Evidence
from sts2_advisor.game.state import build_decision_context
from evals.prompt_experiment import (
    RequestBudget, build_compact_prompt, build_compact_system_prompt, inspect_advice, summarize,
)


def context():
    raw = json.loads(Path('tests/fixtures/calculator_log_cases.json').read_text(encoding='utf-8'))
    return build_decision_context(raw['group'])


def test_compact_keeps_sqlite_and_vector_evidence_verbatim():
    evidence = [
        Evidence('sqlite:cards:x', '当前事实：伤害8。', '关系数据库'),
        Evidence('guide:x', '攻略原文。', '向量攻略'),
        Evidence('builtin-combat-survival-001', '旧的原则', 'seed'),
    ]
    text = build_compact_prompt(context(), evidence)
    payload = json.loads(text.split('输入 JSON：', 1)[1])
    by_id = {e['id']: e['text'] for e in payload['evidence']}
    assert by_id['sqlite:cards:x'] == evidence[0].text
    assert by_id['guide:x'] == evidence[1].text
    assert by_id[evidence[2].evidence_id] != evidence[2].text
    assert '方案' in by_id[evidence[2].evidence_id]
    assert set(by_id) == {e.evidence_id for e in evidence}
    assert 'rubric' not in payload
    system = build_compact_system_prompt()
    assert 'python' not in system.lower()
    assert 'selection_analysis' in system


def test_request_budget_counts_tool_followups_and_reserves_output():
    budget = RequestBudget(max_requests=2, max_tokens=1000)
    request = {'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 10}
    budget.reserve(request)
    budget.record_usage({'prompt_tokens': 10, 'completion_tokens': 5})
    budget.reserve(request)
    budget.record_usage({'prompt_tokens': 10, 'completion_tokens': 5})
    with unittest.TestCase().assertRaisesRegex(RuntimeError, '请求预算'):
        budget.reserve(request)
    assert budget.requests == 2
    assert budget.tokens == 30
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'token预算'):
        RequestBudget(max_requests=2, max_tokens=5).reserve(request)


def test_unknown_usage_keeps_conservative_reservation():
    budget = RequestBudget(max_requests=4, max_tokens=1000)
    request = {'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 10}
    budget.reserve(request)
    reserved = budget.tokens
    budget.record_usage(None)
    assert budget.tokens == reserved
    assert reserved > 10


def test_raw_plan_boundary_is_reported_before_validation_truncation():
    ctx = context()
    draw_card = next(c for c in ctx.candidates if '剥夺' in c.label)
    advice = Advice(ctx.decision_id, draw_card.candidate_id, '先打出再观察', .7,
                    plan_candidate_ids=[draw_card.candidate_id, 'combat:end_turn'])
    result = inspect_advice(ctx, [], advice.to_dict())
    assert result['boundary_crossed'] is True
    assert result['raw_plan'] == advice.plan_candidate_ids
    assert result['checked_prefix'] == [draw_card.candidate_id]


def test_raw_repeated_card_rejected_even_when_effects_cannot_be_simulated():
    ctx = context()
    attack = next(c for c in ctx.candidates if '打击' in c.label and '双重' not in c.label)
    advice = Advice(ctx.decision_id, attack.candidate_id, '两次打击', .7,
                    plan_candidate_ids=[attack.candidate_id, attack.candidate_id])
    assert inspect_advice(ctx, [], advice.to_dict())['resource_error']


def test_frozen_twenty_cases_keep_review_rubrics_out_of_model_input():
    rows = [json.loads(s) for s in Path('evals/fixtures/prompt_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({r['case_id'] for r in rows}) == 20
    assert sum(r['split'] == 'heldout' for r in rows) == 8
    from evals.compare_prompts import restore
    for row in rows:
        ctx, evidence = restore(row)
        prompt = build_compact_prompt(ctx, evidence)
        assert row['rubric'] not in prompt
        assert 'rubric' not in prompt
        assert row['evidence_sources'].get('sqlite', 0) > 0
    assert any(r['evidence_sources'].get('vector', 0) > 0 for r in rows)


def test_global_budget_stops_tool_followup_before_network():
    from tools.compare_thinking import run_case
    ctx = context()
    attack = next(c for c in ctx.candidates if c.action_type == 'play_card' and c.params.get('target'))
    args = json.dumps({'decision_id': ctx.decision_id, 'candidate_id': attack.candidate_id, 'expected_hp_after': 0})
    call = SimpleNamespace(id='call1', function=SimpleNamespace(name='calculate_attack', arguments=args))
    message = SimpleNamespace(content=None, reasoning_content=None, tool_calls=[call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='tool_calls')],
                               usage={'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30})
    network_calls = []
    def create(**kwargs):
        network_calls.append(kwargs)
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    budget = RequestBudget(max_requests=1, max_tokens=100000)
    with patch.dict('os.environ', {'OPENAI_MODEL': 'test', 'OPENAI_API_KEY': 'test-not-a-secret'}), \
         patch('openai.OpenAI', return_value=client):
        result = run_case({'context': ctx.to_dict(), 'evidence': []}, 'disabled', 500, 'compact', budget=budget)
    assert len(network_calls) == budget.requests == result['requests'] == 1
    assert '请求预算' in result['error']
    assert 'advice' not in result
    summary = summarize([result])['compact']
    assert summary['responses'] == 0
    assert summary['attempts'] == 1


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for key, value in globals().items() if key.startswith("test_") and callable(value))

