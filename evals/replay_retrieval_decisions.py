"""Paired model replay of frozen before/after retrieval; --run spends API tokens."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from evals.compare_prompts import SMOKE, digest, restore
from evals.prompt_experiment import RequestBudget, inspect_advice, summarize
from tools.compare_thinking import run_case


def paired_rows(fixtures: list[dict], retrieval: dict, case_ids: list[str]) -> list[dict]:
    contexts = {row['case_id']: row for row in fixtures}
    evidence = {row['case_id']: row for row in retrieval['cases']}
    if len(contexts) != len(fixtures) or len(evidence) != len(retrieval['cases']):
        raise ValueError('输入含重复局面ID')
    if len(set(case_ids)) != len(case_ids) or set(case_ids) - (contexts.keys() & evidence.keys()):
        raise ValueError('待测局面重复或缺少对应状态/检索结果')
    pairs = []
    for pos, case_id in enumerate(case_ids):
        # Alternate execution order; both sides use the CURRENT system prompt,
        # context, calculator, model and seed, differing only in retrieval output.
        for version in (['before', 'after'] if pos % 2 == 0 else ['after', 'before']):
            pairs.append({'case_id': case_id, 'split': contexts[case_id]['split'],
                          'retrieval_version': version, 'context': contexts[case_id]['context'],
                          'evidence': evidence[case_id][version]['evidence']})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=Path('evals/fixtures/prompt_cases.jsonl'))
    parser.add_argument('--retrieval', type=Path, default=Path('tmp/retrieval-relevance-20261110-final.json'))
    parser.add_argument('--case-ids', nargs='+', default=SMOKE)
    parser.add_argument('--max-tokens', type=int, default=1600)
    parser.add_argument('--request-budget', type=int, default=32)
    parser.add_argument('--token-budget', type=int, default=160000)
    parser.add_argument('--output', type=Path, default=Path('tmp/retrieval-decision-replay.json'))
    args = parser.parse_args()
    if min(args.max_tokens, args.request_budget, args.token_budget) <= 0:
        parser.error('预算必须为正数')
    fixtures = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines() if s.strip()]
    retrieval = json.loads(args.retrieval.read_text(encoding='utf-8'))
    pairs = paired_rows(fixtures, retrieval, args.case_ids)
    if not args.run:
        for row in pairs:
            print(row['case_id'], row['retrieval_version'], [e['evidence_id'] for e in row['evidence']])
        return
    if args.output.exists():
        parser.error('结果文件已存在；请指定新路径，保留原始结果')
    from sts2_advisor.cli import load_environment
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        parser.error('只复用已授权的DeepSeek服务，不切换提供商')
    logging.disable(logging.CRITICAL)
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    results = []
    report = {'model': os.environ['OPENAI_MODEL'], 'thinking': 'disabled', 'prompt_profile': 'current',
              'max_tokens': args.max_tokens, 'baseline_revision': retrieval['baseline'],
              'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'retrieval_sha256': digest(args.retrieval.read_text(encoding='utf-8')),
              'limits': {'requests': args.request_budget, 'tokens': args.token_budget},
              'expected_decisions': len(pairs), 'results': results}
    for row in pairs:
        print(row['case_id'], row['retrieval_version'], flush=True)
        result = run_case(row, 'disabled', args.max_tokens, 'current', budget=budget)
        if result.get('advice'):
            context, evidence = restore(row)
            result['checks'] = inspect_advice(context, evidence, result['advice'])
        results.append({k: row[k] for k in ('case_id', 'split', 'retrieval_version')} | result)
        report['budget_used'] = {'requests': budget.requests, 'tokens': budget.tokens}
        report['summary'] = summarize([{**r, 'prompt_variant': r['retrieval_version']} for r in results])
        report['coverage'] = len(results) / len(pairs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(' ', result.get('error') or result['advice']['reason'], flush=True)
        if result.get('error') and not result.get('model_responses'):
            raise SystemExit('连接失败或预算不足；已保存部分结果，停止后续请求')
    if any(r.get('error') for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
