"""Re-answer old fixtures with pre-change/current retrieval and feedback. No game I/O."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

from evals import compare_attack_delivery as replay
from evals.compare_prompts import digest, restore
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import load_environment
from sts2_advisor.integrations.knowledge.examples import EXAMPLE_KIND

BASELINE = '6d509b0'
SOURCE = Path('evals/fixtures/player_outcome_medium_cases.jsonl')
FROZEN = Path('evals/fixtures/fixed_regression_cases.jsonl')
# Prior 12 resource-check cases + four declared breadth cases. Frozen before API.
CASES = ['m01', 'm03', 'm04', 'm06', 'm07', 'm10', 'm11', 'm14',
         'm17', 'm19', 'm22', 'm24', 'm26', 'm29', 'm30', 'm32']


def git_source(path):
    return subprocess.run(['git', '-c', f'safe.directory={Path.cwd().as_posix()}',
                           'show', f'{BASELINE}:{path}'], check=True, capture_output=True,
                          encoding='utf-8').stdout


def validate_budget(tokens, requests):
    if not 0 < tokens <= 500000 or not 0 < requests <= 128:
        raise ValueError('本轮上限500000 token、128请求')


class PreExampleCollection:
    """Read-only view of the existing collection, hiding the four new records."""
    def __init__(self, collection): self.collection = collection
    def count(self): return self.collection.count()
    def query(self, **kwargs):
        kwargs['where'] = {'kind': {'$ne': EXAMPLE_KIND}}
        result = self.collection.query(**kwargs)
        if any((m or {}).get('kind') == EXAMPLE_KIND for batch in result.get('metadatas', []) for m in batch):
            raise ValueError('旧组混入新案例')
        return result


def prepare(path):
    from sts2_advisor.integrations.knowledge.factory import create_retriever
    from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
    from sts2_advisor.integrations.knowledge.sqlite import SqliteFactsRetriever
    from sts2_advisor.integrations.knowledge.seeds import SeedInjectingRetriever
    if path.exists(): raise ValueError('禁止覆盖冻结输入')
    os.environ['HF_HUB_OFFLINE'] = '1'
    retriever = create_retriever('auto', Path('data/chroma'), Path('data/knowledge.jsonl'))
    current = getattr(retriever.inner, 'fallback', None)
    if not isinstance(current, ChromaRetriever): raise ValueError('需要SQLite/Chroma/seed真实组合')
    source = git_source('src/sts2_advisor/integrations/knowledge/chroma.py')
    namespace = {'__name__': 'baseline_chroma'}
    exec(compile(source, f'git:{BASELINE}:chroma.py', 'exec'), namespace)
    old_chroma = namespace['ChromaRetriever'].__new__(namespace['ChromaRetriever'])
    old_chroma.collection = PreExampleCollection(current.collection)
    old = SeedInjectingRetriever(SqliteFactsRetriever(retriever.inner.path, old_chroma), retriever.seeds)
    # All other decision inputs/logic stay the same; resource trace is optional.
    for name in ['integrations/models/remote.py', 'decision/prompts.py', 'decision/plan_check.py',
                 'integrations/knowledge/sqlite.py', 'integrations/knowledge/seeds.py']:
        file = Path('src/sts2_advisor') / name
        if git_source(file.as_posix()).replace('\r\n', '\n') != file.read_text(encoding='utf-8'):
            raise ValueError(f'不可混入其他生产差异：{name}')
    teachers = {r['metadata']['source_encounter'] for r in
                (json.loads(s) for s in Path('data/decision_examples.jsonl').read_text(encoding='utf-8').splitlines())}
    fixtures = [json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines()]
    rows = []
    for case_id in CASES:
        row = copy.deepcopy(next(r for r in fixtures if r['case_id'] == case_id))
        context, _ = restore(row)
        if digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)) != row['context_sha256']:
            raise ValueError('旧状态哈希不符')
        durations, evidence = {}, {}
        for profile, adapter in [('baseline', old), ('current', retriever)]:
            start = time.perf_counter()
            evidence[profile] = [e.to_dict() for e in adapter.retrieve(context, 4)]
            durations[profile] = time.perf_counter() - start
        encounter = row['source'].replace('\\', '/') + '#floor=' + str(context.run['floor'])
        row.update(evidence=evidence['baseline'], current_evidence=evidence['current'],
                   retrieval_top_k=4, retrieval_seconds_cold_ordered=durations,
                   teaching_encounter=encounter in teachers, prior_repair_diagnostic=case_id == 'm07',
                   baseline_chroma_sha256=digest(source), original_fixture_sha256=digest(SOURCE.read_text(encoding='utf-8')))
        if case_id == 'm03':
            row['review_addendum'] = ('此前已核实：痛击+→两愤怒→御血术对神官可约51伤，'
                '历石第7回合末52伤可清除神官，再承受信徒9攻可存活；旧rubric的未见斩杀不可用作无解金标。')
        if any(e['metadata'].get('kind') == EXAMPLE_KIND for e in row['evidence']):
            raise ValueError('旧证据含新案例')
        old_seeds = [e for e in row['evidence'] if e['evidence_id'].startswith('seed-')]
        new_seeds = [e for e in row['current_evidence'] if e['evidence_id'].startswith('seed-')]
        if old_seeds != new_seeds: raise ValueError('seed不一致')
        print(case_id, row['group'], 'evidence', len(row['evidence']), len(row['current_evidence']),
              'example', [e['metadata'].get('mechanism') for e in row['current_evidence']
                          if e['metadata'].get('kind') == EXAMPLE_KIND], flush=True)
        rows.append(row)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def run(args):
    validate_budget(args.token_budget, args.request_budget)
    if args.output.exists(): raise ValueError('禁止覆盖旧结果')
    rows = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines()]
    if [r['case_id'] for r in rows] != CASES: raise ValueError('冻结范围不符')
    for row in rows:
        if digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)) != row['context_sha256']:
            raise ValueError('状态哈希不符')
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower(): raise ValueError('仅已授权DeepSeek')
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    report = {'baseline_revision': BASELINE, 'current_revision': 'ae49ac6',
              'comparison': 'full retrieval+feedback change; ordinary Flash 1024; no executor',
              'limits': {'tokens': args.token_budget, 'requests': args.request_budget},
              'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'expected_decisions': len(rows) * 2, 'repetitions': 1, 'results': []}
    stop = False
    for index, row in enumerate(rows):
        for profile in (['baseline', 'current'] if index % 2 == 0 else ['current', 'baseline']):
            used = copy.deepcopy(row)
            if profile == 'current': used['evidence'] = used['current_evidence']
            print(row['case_id'], profile, flush=True)
            with (patch('sts2_advisor.decision.feedback.build_repair_feedback',
                        side_effect=lambda ctx, prop, err: err) if profile == 'baseline' else nullcontext()):
                result = replay.model_trial(used, False, budget, model_name='deepseek-flash')
            report['results'].append({'case_id': row['case_id'], 'profile': profile,
                                      'group': row['group'], 'teaching_encounter': row['teaching_encounter'],
                                      'prior_repair_diagnostic': row['prior_repair_diagnostic'], **result})
            report['summary'] = replay.mechanical_summary(report['results'])
            report['budget_used'] = {'requests': budget.requests, 'tokens_or_conservative_reserve': budget.tokens}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(' ', result['error'] or result['advice']['reason'], budget.tokens, flush=True)
            if '预算' in (result['error'] or '') or result['error'] and not result['model_responses']:
                stop = True; break
        if stop: break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--run', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=FROZEN)
    parser.add_argument('--output', type=Path, default=Path('tmp/fixed-regression-20261117-api.json'))
    parser.add_argument('--token-budget', type=int, default=500000)
    parser.add_argument('--request-budget', type=int, default=128)
    args = parser.parse_args(); logging.disable(logging.CRITICAL)
    if args.prepare: prepare(args.fixtures)
    elif args.run: run(args)
    else: print('离线；16旧题×2版本，--prepare冻结检索，--run才付费，上限50万token')


if __name__ == '__main__': main()
