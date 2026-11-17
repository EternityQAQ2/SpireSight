"""Audit saved proposals for free, or run the authorized 500k-token comparison."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from evals import compare_attack_delivery as replay
from evals.compare_prompts import digest, restore
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import load_environment
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.decision.prompts import SYSTEM_PROMPT

BASELINE = '2e1ecab'
CASES = ['m01', 'm04', 'm06', 'm07', 'm11', 'm14', 'm17', 'm22', 'm24', 'm26', 'm29', 'm32']
FIXTURES = Path('evals/fixtures/player_outcome_medium_cases.jsonl')
HISTORY = Path('tmp/player-outcome-medium-20261116-api.json')
MANIFEST = Path('evals/fixtures/resource_precheck_manifest.json')


def baseline():
    def source(path):
        return subprocess.run(['git', '-c', f'safe.directory={Path.cwd().as_posix()}',
                               'show', f'{BASELINE}:{path}'], check=True, capture_output=True,
                              encoding='utf-8').stdout
    checker = source('src/sts2_advisor/decision/plan_check.py')
    prompt = source('src/sts2_advisor/decision/prompts.py')
    namespace = {'__name__': 'baseline_plan_check'}
    exec(compile(checker, f'git:{BASELINE}:plan_check.py', 'exec'), namespace)
    for node in ast.parse(prompt).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in node.targets):
            return namespace['check_plan'], ast.literal_eval(node.value), digest(checker)
    raise ValueError('旧提示不存在')


def validate_budget(tokens, requests):
    if not 0 < tokens <= 500000 or not 0 < requests <= 96:
        raise ValueError('本轮上限50万token、96请求')


@contextmanager
def configuration(checker, prompt):
    with ExitStack() as stack:
        for target in ('sts2_advisor.decision.plan_check.check_plan',
                       'sts2_advisor.decision.validation.check_plan',
                       'sts2_advisor.decision.tools.check_plan',
                       'sts2_advisor.decision.plans.check_plan',
                       'evals.prompt_experiment.check_plan'):
            stack.enter_context(patch(target, checker))
        stack.enter_context(patch('sts2_advisor.decision.prompts.SYSTEM_PROMPT', prompt))
        yield


def audit(rows, old_check):
    inputs = {r['case_id']: r for r in rows}
    saved = json.loads(HISTORY.read_text(encoding='utf-8'))
    results = []
    for result in saved['results']:
        context, _ = restore(inputs[result['case_id']])
        for index, proposal in enumerate(result['proposals']):
            ids = proposal.get('plan_candidate_ids') or [proposal['candidate_id']]
            before, after = old_check(context, ids), check_plan(context, ids)
            results.append({'case_id': result['case_id'], 'repeat': result['repeat'],
                            'profile': result['profile'], 'proposal_index': index,
                            'final_proposal': index == len(result['proposals']) - 1,
                            'plan': ids, 'old_error': before.error, 'new_error': after.error,
                            'new_rejection': not before.error and bool(after.error),
                            'old_prefix': before.candidate_ids, 'new_prefix': after.candidate_ids})
    return {'source_sha256': digest(HISTORY.read_text(encoding='utf-8')),
            'decisions': len(saved['results']), 'proposals': len(results),
            'new_rejections': sum(x['new_rejection'] for x in results), 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('tmp/resource-precheck.json'))
    parser.add_argument('--token-budget', type=int, default=500000)
    parser.add_argument('--request-budget', type=int, default=96)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        validate_budget(args.token_budget, args.request_budget)
    except ValueError as exc:
        parser.error(str(exc))
    rows = [json.loads(s) for s in FIXTURES.read_text(encoding='utf-8').splitlines() if s.strip()]
    for row in rows:
        if row['context_sha256'] != digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)):
            parser.error('冻结状态哈希不一致')
    if not args.audit and not args.run:
        print('先--audit免费核对128决策；--run为12题×2组=24决策，上限50万token/96请求')
        return
    if args.output.exists():
        parser.error('禁止覆盖已有报告，使用新路径')
    if args.output.exists():
        parser.error('禁止覆盖旧结果')
    old_check, old_prompt, old_hash = baseline()
    if args.audit:
        report = audit(rows, old_check)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(report['decisions'], '决策', report['proposals'], '提案，新增拦截', report['new_rejections'], '，0 API token')
        return
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        parser.error('仅已授权DeepSeek')
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    selected = [next(r for r in rows if r['case_id'] == case) for case in CASES]
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if (manifest['fixtures_sha256'] != digest(FIXTURES.read_text(encoding='utf-8'))
            or [c['case_id'] for c in manifest['cases']] != CASES):
        parser.error('预先冻结的实验范围发生变化')
    report = {'baseline_revision': BASELINE, 'baseline_checker_sha256': old_hash,
              'manifest_sha256': digest(MANIFEST.read_text(encoding='utf-8')),
              'cases': CASES, 'fixtures_sha256': digest(FIXTURES.read_text(encoding='utf-8')),
              'limits': {'tokens': args.token_budget, 'requests': args.request_budget},
              'comparison': 'old checker+prompt versus resource precheck+two corrected rules; no executor',
              'expected_decisions': 24, 'repetitions': 1, 'results': []}
    failures = 0
    for pos, row in enumerate(selected):
        arms = [('old', old_check, old_prompt), ('new', check_plan, SYSTEM_PROMPT)]
        if pos % 2: arms.reverse()
        for profile, checker, prompt in arms:
            print(row['case_id'], profile, flush=True)
            with configuration(checker, prompt):
                result = replay.model_trial(row, False, budget, model_name='deepseek-flash')
            report['results'].append({'case_id': row['case_id'], 'profile': profile, **result})
            report['summary'] = replay.mechanical_summary(report['results'])
            report['budget_used'] = {'requests': budget.requests, 'tokens_or_conservative_reserve': budget.tokens}
            failures = failures + 1 if result['error'] and not result['model_responses'] else 0
            if '预算' in (result['error'] or '') or failures >= 3: report['stop_reason'] = result['error']
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix('.partial')
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary.replace(args.output)
            print(result['error'] or 'OK', budget.tokens, flush=True)
            if report.get('stop_reason'): return


if __name__ == '__main__':
    main()
