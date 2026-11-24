"""Compare only calculator result fields. Offline by default; --run spends API tokens."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from evals import compare_attack_delivery as replay
from evals.compare_prompts import digest
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import load_environment
from sts2_advisor.game.combat.calculator import calculate


FIXTURES = Path('evals/fixtures/player_outcome_cases.jsonl')
BASELINE = 'b827915'
MANIFEST = [
    ('avoid-unnecessary-self-loss', '20261109-214328-690834', 1,
     '3血攻击者：打击与御血术同费同杀，前者不自损。打大怪御血术28仍不能杀30血。防御换血可讨论，不虚称非致死攻击阻止攻击。此题是此前免费攻击题同场后续快照，非独立新战斗。'),
    ('useful-self-loss', '20261109-221829-452472', 4,
     '御血术19伤可杀19血攻击者，自损2有价值；勿一概禁用。无情猛攻18伤和下张免费也可合理组合。感染3伤在回合末，不混入当前单张出牌后HP。免费仅覆盖下一张攻击。'),
    ('unknown-trigger-control', '20261109-214328-690834', 69,
     '62血寄生虫不攻击，有未支持感染状态；工具未知不是0或确定预测。模型可独立算可见自损2但须不冒充完整核实。合理输出/升级均允许，不凭没有HP引用断言忽略工具。'),
]


def baseline_calculator():
    result = subprocess.run(
        ['git', '-c', f'safe.directory={Path.cwd().as_posix()}', 'show',
         f'{BASELINE}:src/sts2_advisor/game/combat/calculator.py'],
        check=True, capture_output=True, encoding='utf-8')
    namespace = {'__name__': 'baseline_attack_calculator'}
    exec(compile(result.stdout, f'git:{BASELINE}:calculator.py', 'exec'), namespace)
    return namespace['calculate'], digest(result.stdout)


def prepare(path):
    old_ids = set()
    for source in Path('evals/fixtures').glob('*.jsonl'):
        for line in source.read_text(encoding='utf-8').splitlines():
            if line.strip():
                old_ids.add(json.loads(line).get('context', {}).get('decision_id'))
    for _, session, index, _ in MANIFEST:
        row = json.loads(Path(f'recordings/session-{session}.jsonl').read_text(encoding='utf-8').splitlines()[index])
        if row['context']['decision_id'] in old_ids:
            raise ValueError('新快照与已有固定题重叠')
    with patch.object(replay, 'MANIFEST', MANIFEST):
        replay.prepare(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--output', type=Path, default=Path('tmp/player-outcome-api.json'))
    parser.add_argument('--token-budget', type=int, default=200000)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.prepare:
        if args.fixtures.exists():
            parser.error('冻结输入已存在，禁止覆盖')
        prepare(args.fixtures)
        print('3个新快照已冻结；使用真实三路检索，未调用模型或游戏')
        return
    rows = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not args.run:
        print('3快照×2计算器×2重复=12决策，普通Flash，tools auto，预算上限20万token/48请求；--run才付费')
        return
    if args.output.exists() or len(rows) != 3 or not 0 < args.token_budget <= 200000:
        parser.error('禁止覆盖；本轮限3题和20万token')
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        parser.error('仅使用已授权DeepSeek')
    old_calculate, source_hash = baseline_calculator()
    budget = RequestBudget(max_requests=48, max_tokens=args.token_budget)
    report = {'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'baseline_revision': BASELINE, 'baseline_calculator_sha256': source_hash,
              'pipeline': 'AdvisorService, same frozen retrieval/prompts/schema; only tool implementation differs; no executor',
              'limits': {'tokens': args.token_budget, 'requests': 48}, 'results': []}
    for repeat in range(2):
        for pos, row in enumerate(rows):
            arms = [('old', old_calculate), ('new', calculate)]
            if (repeat + pos) % 2:
                arms.reverse()
            for profile, calculator in arms:
                print(repeat + 1, row['case_id'], profile, flush=True)
                with patch('sts2_advisor.decision.tools.calculate', calculator):
                    result = replay.model_trial(row, False, budget, model_name='deepseek-flash')
                report['results'].append({'case_id': row['case_id'], 'repeat': repeat + 1,
                                          'profile': profile, **result})
                report['summary'] = replay.mechanical_summary(report['results'])
                report['budget_used'] = {'requests': budget.requests, 'tokens': budget.tokens}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                print(result['error'] or '返回完成', '累计token', budget.tokens, flush=True)
                if result['error'] and not result['model_responses']:
                    raise SystemExit('预算或请求失败；保留结果并停止，不补跑')


if __name__ == '__main__':
    main()
