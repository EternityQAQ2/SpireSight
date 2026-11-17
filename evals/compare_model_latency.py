"""Audit real request caps and tool gating; explicitly authorized DS-low test only."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import logging
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

from evals.compare_attack_delivery import replay_decision
from evals.compare_prompts import digest, restore
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import create_remote_backend, load_environment
from sts2_advisor.decision.tools import DecisionTools

FIXTURES = Path('evals/fixtures/fixed_regression_cases.jsonl')
CASES = ['m01', 'm04', 'm07', 'm24', 'm29']
PROFILES = [('ordinary_before', 'disabled', 1024, False),
            ('ordinary_after', 'disabled', 1024, True),
            ('low_2k', 'enabled', 2048, True), ('low_4k', 'enabled', 4096, True)]


def validate_budget(tokens, requests):
    if not 0 < tokens <= 250000 or not 0 < requests <= 80:
        raise ValueError('本轮上限250000 token、80请求')


def percentile(values, fraction):
    if not values: return None
    values = sorted(values); pos = (len(values) - 1) * fraction
    return round(values[math.floor(pos)] + (values[math.ceil(pos)] - values[math.floor(pos)]) * (pos % 1), 4)


def summary(results):
    report = {}
    for profile, *_ in PROFILES:
        rows = [r for r in results if r['profile'] == profile]
        if not rows: continue
        success = [r for r in rows if r['advice'] is not None]
        usages = [u for r in rows for u in r['usage'] if u]
        durations = [r['elapsed_seconds'] for r in rows]
        complete = [r['elapsed_seconds'] for r in success]
        report[profile] = {'attempts': len(rows), 'delivered': len(success),
            'requests': sum(r['requests'] for r in rows),
            'redecisions': sum(max(0, len(r['proposals']) - 1) for r in rows),
            'known_usage_tokens': sum(u.get('total_tokens', 0) for u in usages),
            'input_tokens': sum(u.get('prompt_tokens', 0) for u in usages),
            'output_tokens': sum(u.get('completion_tokens', 0) for u in usages),
            'reasoning_tokens': sum((u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0) for u in usages),
            'cache_hit_tokens': sum(u.get('prompt_cache_hit_tokens', 0) for u in usages),
            'all_attempt_seconds': {'median': percentile(durations, .5), 'p95': percentile(durations, .95),
                                    'max': max(durations)},
            'successful_delivery_seconds': {'median': percentile(complete, .5), 'p95': percentile(complete, .95),
                                           'max': max(complete) if complete else None},
            'empty_or_truncated_failures': sum(bool(r['error']) and any(
                (response.get('choices') or [{}])[0].get('finish_reason') == 'length'
                for response in r['model_responses']) for r in rows),
            'tool_calls': sum(len(r['tools']) for r in rows),
            'requests_with_tools': sum('tools' in req for r in rows for req in r['request_details'])}
    return report


def trial(row, profile, budget):
    name, thinking, cap, gate = profile
    backend = create_remote_backend(max_tokens=cap, timeout=45, precompute_attacks=False, model_name='deepseek-flash')
    backend.thinking = thinking; backend.adaptive_thinking = False; backend.reasoning_effort = 'low'
    backend.client.max_retries = 0
    context, _ = restore(row)
    availability = DecisionTools(context).attack_tool_availability()
    requests, responses, tools, events, request_seconds = [], [], [], [], []
    original = backend.client.chat.completions.create
    def create(**kwargs):
        if len(requests) >= 4: raise RuntimeError('单局面最多4请求')
        if kwargs['max_tokens'] != cap: raise ValueError('实际预算被意外扩容，未发送')
        if kwargs.get('extra_body', {}).get('thinking', {}).get('type') != thinking:
            raise ValueError('思考模式与实验不一致')
        if thinking == 'enabled' and kwargs.get('reasoning_effort') != 'low':
            raise ValueError('思考强度不是low')
        budget.reserve(kwargs); requests.append(copy.deepcopy(kwargs))
        start = time.perf_counter()
        try:
            response = original(**kwargs); raw = response.model_dump()
            budget.record_usage(raw.get('usage')); responses.append(raw)
            return response
        finally:
            request_seconds.append(round(time.perf_counter() - start, 4))
    def capture(kind, **data):
        if kind == 'model.tool_result': tools.append(copy.deepcopy(data))
        if kind == 'model.tool_availability': events.append(copy.deepcopy(data))
    try:
        with patch.object(backend.client.chat.completions, 'create', side_effect=create), \
             patch('sts2_advisor.integrations.models.remote.record_event', capture), \
             (nullcontext() if gate else patch.object(DecisionTools, 'attack_tool_availability',
                return_value={**availability, 'available': True})):
            result = replay_decision(row, backend)
        return {**result, 'requests': len(requests), 'request_details': requests,
            'model_responses': responses, 'usage': [r.get('usage') for r in responses],
            'tools': tools, 'tool_availability': availability, 'availability_events': events,
            'request_seconds': request_seconds,
            'config': {'model': backend.model_name, 'thinking': thinking, 'reasoning_effort': 'low' if thinking == 'enabled' else None,
                       'max_tokens': cap, 'tool_gating': gate, 'sdk_retries': 0}}
    finally: backend.client.close()


def run(args):
    validate_budget(args.token_budget, args.request_budget)
    if args.output.exists(): raise ValueError('禁止覆盖旧结果')
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower(): raise ValueError('仅已授权DeepSeek')
    if os.environ.get('OPENAI_THINKING') != 'disabled': raise ValueError('线上配置发生变化，请先核实')
    all_rows = [json.loads(s) for s in FIXTURES.read_text(encoding='utf-8').splitlines()]
    rows = [copy.deepcopy(next(r for r in all_rows if r['case_id'] == ident)) for ident in CASES]
    for row in rows:
        if digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)) != row['context_sha256']:
            raise ValueError('冻结状态变动')
        row['evidence'] = row['current_evidence']
    report = {'baseline_revision': 'e962ffa', 'limits': {'tokens': args.token_budget, 'requests': args.request_budget},
              'cases': CASES, 'profiles': PROFILES, 'expected_decisions': 20, 'repetitions': 1,
              'latency_threshold_seconds': None, 'user_preference': 'show distribution; no pass/fail deadline',
              'fixture_sha256': digest(FIXTURES.read_text(encoding='utf-8')), 'results': [],
              'limitations': 'known old cases, 5 samples per arm; descriptive p95 only; no legacy 16k thinking run; no other-provider credentials'}
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    stop = False
    for i, row in enumerate(rows):
        profiles = PROFILES[i % 4:] + PROFILES[:i % 4]
        for profile in profiles:
            print(row['case_id'], profile[0], 'max_tokens', profile[2], flush=True)
            result = trial(row, profile, budget)
            report['results'].append({'case_id': row['case_id'], 'profile': profile[0], **result})
            report['summary'] = summary(report['results'])
            report['budget_used'] = {'requests': budget.requests, 'tokens_or_conservative_reserve': budget.tokens}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(' ', result['elapsed_seconds'], result['error'] or result['advice']['reason'], budget.tokens, flush=True)
            if '预算' in (result['error'] or '') or result['error'] and not result['model_responses']:
                stop = True; break
        if stop: break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('tmp/model-latency-20261117-api.json'))
    parser.add_argument('--token-budget', type=int, default=250000)
    parser.add_argument('--request-budget', type=int, default=80)
    args = parser.parse_args(); logging.disable(logging.CRITICAL)
    if args.run: run(args)
    else: print('离线；5旧题×4组=20决策，--run才付费，上限25万token；无游戏、无延迟门槛、无Pro')


if __name__ == '__main__': main()
