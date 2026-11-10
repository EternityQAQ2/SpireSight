"""Four bounded non-thinking experiments on the established eight-case benchmark."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

from evals.compare_prompts import SMOKE, digest, restore
from evals.prompt_experiment import RequestBudget, inspect_advice, summarize
from tools.compare_thinking import run_case
from sts2_advisor.decision.tools import DecisionTools, TOOL_SCHEMAS


PROFILES = ('legacy_flash', 'tool_flash', 'compare_flash', 'tool_pro')
COMPARISON_RULE = '''\n输出协议补充：战斗JSON在原字段外增加combat_comparison数组，恰含两个可行方案的结果摘要。每项含plan_candidate_ids、energy_cost、certain_kills、hp_loss（未知用null）、future_benefit（不超过30字）。数值必须计入当前意图和回合结束效果；不确定的后续结果保持未知。reason简述选择的主要取舍。不输出推导过程。非战斗仍用原格式。'''


def load_legacy(revision='b264d85'):
    def module(path):
        source = subprocess.check_output(['git', '-c', f'safe.directory={Path.cwd().as_posix()}',
                                          'show', f'{revision}:src/sts2_advisor/{path}.py'],
                                         encoding='utf-8')
        result = types.ModuleType('legacy_' + path.replace('/', '_'))
        exec(compile(source, f'{revision}/{path}.py', 'exec'), result.__dict__)
        return result
    calculator = module('game/combat/calculator')
    tools = module('decision/tools')
    tools.calculate = calculator.calculate
    return tools


def transform_request(request, *, old_system=None, compare=False):
    result = {**request, 'messages': copy.deepcopy(request['messages'])}
    if old_system is not None:
        result['messages'][0]['content'] = old_system
    if compare:
        result['messages'][0]['content'] += COMPARISON_RULE
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case-ids', nargs='+', default=SMOKE)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=PROFILES)
    parser.add_argument('--fixtures', type=Path, default=Path('evals/fixtures/prompt_cases.jsonl'))
    parser.add_argument('--retrieval', type=Path, default=Path('tmp/retrieval-relevance-20261110-final.json'))
    parser.add_argument('--previous', type=Path, default=Path('tmp/retrieval-decisions-20261110-api.json'))
    parser.add_argument('--request-budget', type=int, default=64)
    parser.add_argument('--token-budget', type=int, default=360000)
    parser.add_argument('--output', type=Path, default=Path('tmp/decision-options.json'))
    args = parser.parse_args()
    if min(args.request_budget, args.token_budget) <= 0:
        parser.error('预算必须大于0')
    fixtures = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines() if s.strip()]
    retrieval = json.loads(args.retrieval.read_text(encoding='utf-8'))
    previous = json.loads(args.previous.read_text(encoding='utf-8'))
    cases = {r['case_id']: r for r in fixtures}
    evidence = {r['case_id']: r['after']['evidence'] for r in retrieval['cases']}
    old_systems = {r['case_id']: r['initial_input'][0]['content'] for r in previous['results']
                   if r['retrieval_version'] == 'after'}
    if (len(set(args.case_ids)) != len(args.case_ids) or len(set(args.profiles)) != len(args.profiles) or
            set(args.case_ids) - (cases.keys() & evidence.keys() & old_systems.keys())):
        parser.error('局面或配置重复/不存在，或缺少冻结输入')
    if not args.run:
        print('cases', args.case_ids, 'profiles', args.profiles,
              'decisions', len(args.case_ids) * len(args.profiles))
        return
    if args.output.exists():
        parser.error('输出已存在，请使用新路径')
    from sts2_advisor.cli import load_environment
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        parser.error('仅使用已授权的DeepSeek提供商')
    logging.disable(logging.CRITICAL)
    legacy = load_legacy()
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    results = []
    report = {'baseline_revision': 'b264d85', 'thinking': 'disabled', 'max_tokens': 1600,
              'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'retrieval_sha256': digest(args.retrieval.read_text(encoding='utf-8')),
              'expected_decisions': len(args.case_ids) * len(args.profiles),
              'limits': {'requests': args.request_budget, 'tokens': args.token_budget}, 'results': results}
    for pos, case_id in enumerate(args.case_ids):
        profiles = list(args.profiles)
        offset = pos % len(profiles)
        for profile in profiles[offset:] + profiles[:offset]:
            row = {'context': cases[case_id]['context'], 'evidence': evidence[case_id]}
            is_legacy = profile == 'legacy_flash'
            compare = profile == 'compare_flash' and row['context']['decision_type'] == 'combat_next_action'
            def transform(request):
                return transform_request(request, old_system=old_systems[case_id] if is_legacy else None,
                                         compare=compare)
            print(case_id, profile, flush=True)
            with patch('sts2_advisor.integrations.models.remote.DecisionTools',
                       legacy.DecisionTools if is_legacy else DecisionTools), \
                 patch('sts2_advisor.integrations.models.remote.TOOL_SCHEMAS',
                       legacy.TOOL_SCHEMAS if is_legacy else TOOL_SCHEMAS):
                result = run_case(row, 'disabled', 1600, 'current', budget=budget,
                                  model_name='deepseek-v4-pro' if profile == 'tool_pro' else 'deepseek-flash',
                                  request_transform=transform)
            if result.get('advice'):
                context, ev = restore(row)
                result['checks'] = inspect_advice(context, ev, result['advice'])
            results.append({'case_id': case_id, 'profile': profile, **result})
            report['summary'] = summarize([{**r, 'prompt_variant': r['profile']} for r in results])
            report['budget_used'] = {'requests': budget.requests, 'tokens': budget.tokens}
            report['coverage'] = len(results) / report['expected_decisions']
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(' ', result.get('error') or result['advice']['reason'], flush=True)
            if result.get('error') and not result.get('model_responses'):
                raise SystemExit('请求失败或预算不足，已保留结果并停止')
    if any(r.get('error') for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
