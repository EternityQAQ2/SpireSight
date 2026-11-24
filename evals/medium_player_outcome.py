"""Frozen 32-case old/new calculator package replay. Offline unless --run."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from evals import compare_attack_delivery as replay
from evals.compare_player_outcome import BASELINE, baseline_calculator
from evals.compare_prompts import digest
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import load_environment
from sts2_advisor.decision.tools import TOOL_SCHEMAS
from sts2_advisor.game.combat.calculator import calculate

MANIFEST = Path('evals/fixtures/player_outcome_medium_manifest.json')
FIXTURES = Path('evals/fixtures/player_outcome_medium_cases.jsonl')


def validate_limits(tokens, requests):
    if not 0 < tokens <= 2000000 or not 0 < requests <= 512:
        raise ValueError('本轮硬上限为200万token和512请求')


def validate_suite(manifest, rows):
    cases = manifest['cases']
    keys = {(str(Path(c['source'])), c['floor']) for c in cases}
    if len(cases) != 32 or len(rows) != 32 or len(keys) != 32:
        raise ValueError('必须32个不同录制+楼层的快照')
    if len({r['context']['decision_id'] for r in rows}) != 32:
        raise ValueError('重复decision_id')
    for case, row in zip(cases, rows):
        if row.get('context_sha256') != digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)):
            raise ValueError('冻结context内容哈希不一致')
        if (case['case_id'], case['decision_id'], str(Path(case['source'])), case['floor']) != (
                row['case_id'], row['context']['decision_id'], str(Path(row['source'])),
                row['context']['run']['floor']):
            raise ValueError('冻结输入与预先声明来源不一致')


def schedule(rows):
    for repeat in range(2):
        for pos, row in enumerate(rows):
            arms = ['old', 'new'] if (repeat + pos) % 2 == 0 else ['new', 'old']
            for arm in arms:
                yield repeat + 1, row, arm


def prepare(manifest, path):
    old_ids, old_encounters = set(), set()
    for source in Path('evals/fixtures').glob('*.jsonl'):
        for line in source.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            context = row.get('context', {})
            old_ids.add(context.get('decision_id'))
            if context and row.get('source'):
                old_encounters.add((str(Path(row['source'])), context.get('run', {}).get('floor')))
    for case in manifest['cases']:
        if case['decision_id'] in old_ids or (str(Path(case['source'])), case['floor']) in old_encounters:
            raise ValueError('与先前固定题ID或同录制楼层重叠')
    items = [(c['case_id'], Path(c['source']).stem.removeprefix('session-'),
              c['source_index'], c['rubric']) for c in manifest['cases']]
    with patch.object(replay, 'MANIFEST', items):
        replay.prepare(path)
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    validate_suite(manifest, rows)
    for row, case in zip(rows, manifest['cases']):
        row['group'] = case['group']
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def old_schema():
    source = subprocess.run(['git', '-c', f'safe.directory={Path.cwd().as_posix()}',
                             'show', f'{BASELINE}:src/sts2_advisor/decision/tools.py'],
                            check=True, capture_output=True, encoding='utf-8').stdout
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TOOL_SCHEMAS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('旧工具schema不存在')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('tmp/player-outcome-medium-api.json'))
    parser.add_argument('--token-budget', type=int, default=2000000)
    parser.add_argument('--request-budget', type=int, default=512)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        validate_limits(args.token_budget, args.request_budget)
    except ValueError as exc:
        parser.error(str(exc))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if args.prepare:
        if FIXTURES.exists():
            parser.error('禁止覆盖冻结输入')
        prepare(manifest, FIXTURES)
        print('32快照三路证据已冻结，未请求模型或游戏')
        return
    rows = [json.loads(s) for s in FIXTURES.read_text(encoding='utf-8').splitlines() if s.strip()]
    validate_suite(manifest, rows)
    if not args.run:
        print('32快照×2组×2重复=128决策；普通Flash；200万token/512请求上限；--run才付费')
        return
    if args.output.exists():
        parser.error('禁止覆盖已有结果')
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        parser.error('仅已授权DeepSeek')
    old_calculate, old_hash = baseline_calculator()
    schemas = {'old': old_schema(), 'new': copy.deepcopy(TOOL_SCHEMAS)}
    functions = {'old': old_calculate, 'new': calculate}
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    report = {'manifest_sha256': digest(MANIFEST.read_text(encoding='utf-8')),
              'fixtures_sha256': digest(FIXTURES.read_text(encoding='utf-8')),
              'baseline_revision': BASELINE, 'baseline_calculator_sha256': old_hash,
              'comparison': 'old calculator+description versus new calculator+description; other input/pipeline identical',
              'limits': {'tokens': args.token_budget, 'requests': args.request_budget},
              'expected_decisions': 128, 'results': []}
    failures = 0
    for repeat, row, profile in schedule(rows):
        print(len(report['results']) + 1, repeat, row['case_id'], profile, flush=True)
        with patch('sts2_advisor.decision.tools.calculate', functions[profile]), \
             patch('sts2_advisor.integrations.models.remote.TOOL_SCHEMAS', schemas[profile]):
            result = replay.model_trial(row, False, budget, model_name='deepseek-flash')
        report['results'].append({'case_id': row['case_id'], 'group': row['group'],
                                  'repeat': repeat, 'profile': profile, **result})
        report['summary'] = replay.mechanical_summary(report['results'])
        report['budget_used'] = {'requests': budget.requests, 'tokens_or_conservative_reserve': budget.tokens}
        failures = failures + 1 if result['error'] and not result['model_responses'] else 0
        if '预算' in (result['error'] or '') or failures >= 3:
            report['stop_reason'] = result['error']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.partial')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(args.output)
        print(result['error'] or 'OK', '累计预算占用', budget.tokens, flush=True)
        if report.get('stop_reason'):
            break


if __name__ == '__main__':
    main()
