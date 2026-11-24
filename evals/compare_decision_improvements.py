"""Separate retrieval and repair A/B tests. Offline by default; never executes game actions."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch

from evals import compare_attack_delivery as replay
from evals.compare_prompts import digest, restore
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import load_environment
from sts2_advisor.contracts import Advice
from sts2_advisor.decision.feedback import build_repair_feedback
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.integrations.knowledge.examples import EXAMPLE_KIND

FIXTURES = Path('evals/fixtures/decision_improvement_cases.jsonl')
MANIFEST = [
    ('discount-transfer', 'session-20261026-202702-348990.jsonl', 63,
     '3能量：猛攻2→免费打击→1费打击可打26；三张普通打击18。残酷无易伤目标无当前伤害收益。不能把所有后续攻击算免费；覆甲等未来机制未知，不编造净损血。'),
    ('slippery-transfer', 'session-20261103-223110-656928.jsonl', 153,
     '三个滑溜1：第一次实际失去生命限1并耗层，非伤害减1。突破可同时耗三敌滑溜且自损1；可再狂宴斩杀，但须重算生命。不可声称单张狂宴杀满血13/14/16。评价计划与理由分别记录，不强制唯一打法。'),
    ('vulnerable-buffing-target', 'session-20261103-215528-121347.jsonl', 7,
     '拆卸对17血易伤强化敌，两次floor(8*1.5)=24，可杀；对22血攻击者仅8不能杀。仅剩1费，防御可少掉5血；允许明确长期收益换血，不能说杀强化敌取消另一个敌人的7攻。'),
    ('existing-block-critical-hp', 'session-20261103-224528-020935.jsonl', 184,
     '8HP已有6盾，对面6×2=12。铁斩波再防御合计16盾可挡完；只铁斩波合11盾尚掉1，铁斩波加打击也不是零损。头槌选牌后重观测，不虚构易伤击杀61血。不能把部分格挡说完全无用。'),
    ('kill-current-attacker', 'session-20261020-185952.jsonl', 65,
     '2能量，4血攻击者、6血塞牌者、48血减益者均易伤。打击6*1.5=9可分别杀两小怪，先杀4血攻击者能取消4攻；完美打击打大怪不能取消小怪攻击。允许有依据防御，不能把杀非攻击者算减当前4攻。'),
    ('partial-block', 'session-20261020-185952.jsonl', 96,
     '71HP已有5盾，只剩1费。敌55血6×2=12；防御+5仍掉2，比只攻击后掉7少5。打击6不能击杀，飞剑随机3×3不是确定击杀。允许具体收益换血，但理由不得漏已有格挡或把部分防御当无效。'),
]


def validate_budget(tokens, requests):
    if not 0 < tokens <= 200000 or not 0 < requests <= 64:
        raise ValueError('本轮上限20万token、64请求')


def prepare(path):
    from sts2_advisor.integrations.knowledge.factory import create_retriever
    from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
    if path.exists():
        raise ValueError('禁止覆盖冻结题目')
    os.environ['HF_HUB_OFFLINE'] = '1'
    old_ids, excluded = set(), set()
    for fixture in Path('evals/fixtures').glob('*.jsonl'):
        for line in fixture.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('context'):
                old_ids.add(row['context']['decision_id'])
    for line in Path('data/decision_examples.jsonl').read_text(encoding='utf-8').splitlines():
        source, floor = json.loads(line)['metadata']['source_encounter'].split('#floor=')
        excluded.add((Path(source).name, int(floor)))
    # Resolve ALL previously evaluated IDs back to encounters, even old fixture
    # formats that have no source filename. Adjacent snapshots are not held out.
    for source in Path('recordings').glob('*.jsonl'):
        for line in source.read_text(encoding='utf-8').splitlines():
            row = json.loads(line); ctx = row.get('context') or {}
            if ctx.get('decision_id') in old_ids:
                excluded.add((source.name, ctx['run'].get('floor')))
    retriever = create_retriever('auto', Path('data/chroma'), Path('data/knowledge.jsonl'))
    if not isinstance(getattr(retriever.inner, 'fallback', None), ChromaRetriever):
        raise ValueError('本轮必须使用现有SQLite/Chroma/seed组合，不能把回退当向量检索')
    rows = []
    for case_id, filename, index, rubric in MANIFEST:
        source = Path('recordings') / filename
        row = json.loads(source.read_text(encoding='utf-8').splitlines()[index])
        context, _ = restore(row)
        if (filename, context.run.get('floor')) in excluded:
            raise ValueError(f'{case_id} 与旧测试或教学战斗重叠')
        with patch('sts2_advisor.integrations.knowledge.chroma.matching_mechanisms', return_value=[]), \
             patch('sts2_advisor.integrations.knowledge.chroma.select_examples',
                   side_effect=lambda ctx, ranked, k: [e for e in ranked if e.metadata.get('kind') != EXAMPLE_KIND][:k]):
            started = time.perf_counter(); before = retriever.retrieve(context, 4)
            old_seconds = time.perf_counter() - started
        started = time.perf_counter(); after = retriever.retrieve(context, 4)
        new_seconds = time.perf_counter() - started
        old = [e.to_dict() for e in before]; new = [e.to_dict() for e in after]
        if len(old) != len(new) or sum(e['metadata'].get('kind') == EXAMPLE_KIND for e in new) != 1:
            raise ValueError('必须只替换一条攻略，证据数不增加')
        removed = [e for e in old if e not in new]; added = [e for e in new if e not in old]
        if len(removed) != 1 or len(added) != 1 or not removed[0]['evidence_id'].startswith('guide-'):
            raise ValueError('证据改动超出单条攻略；不得替换seed/事实')
        rows.append({'case_id': case_id, 'source': str(source), 'source_index': index,
                     'rubric': rubric, 'retrieval_top_k': 4, 'context': row['context'],
                     'raw_state': row['raw_state'], 'evidence': old, 'example_evidence': new,
                     'context_sha256': digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)),
                     'retrieval_seconds_cold_first_old_warm_new': [old_seconds, new_seconds]})
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def repair_cases():
    rows = [json.loads(s) for s in Path('evals/fixtures/player_outcome_medium_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    row = next(r for r in rows if r['case_id'] == 'm07')
    plans = [
        ['card:2:BYGONE_EFFIGY_0', 'card:3:BYGONE_EFFIGY_0', 'card:4:BYGONE_EFFIGY_0', 'card:1:BYGONE_EFFIGY_0'],
        ['card:2:BYGONE_EFFIGY_0', 'card:1:BYGONE_EFFIGY_0'],
    ]
    context, _ = restore(row)
    for i, ids in enumerate(plans, 1):
        if not check_plan(context, ids).error:
            raise ValueError('修复诊断必须从同一个已确认非法计划开始')
        yield {**row, 'case_id': f'repair-{i}'}, Advice(context.decision_id, ids[0],
            '先打出前面的牌，再接重锤结束。', .8, plan_candidate_ids=ids)


def run(args):
    validate_budget(args.token_budget, args.request_budget)
    if args.output.exists():
        raise ValueError('禁止覆盖旧结果')
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        raise ValueError('仅已授权DeepSeek')
    rows = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 6 or [r['case_id'] for r in rows] != [x[0] for x in MANIFEST]:
        raise ValueError('冻结范围不符')
    for row in rows:
        if digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)) != row['context_sha256']:
            raise ValueError('状态哈希不符')
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    report = {'baseline_revision': '6d509b0', 'shared_change': 'bounded local guide overfetch in BOTH retrieval arms',
              'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'limits': {'tokens': args.token_budget, 'requests': args.request_budget},
              'expected_decisions': 16, 'repetitions': 1, 'results': [],
              'repair_limitations': '2 known bad plans in ONE previously tested encounter; not held-out tactics'}
    jobs = []
    for i, row in enumerate(rows):
        arms = ['baseline', 'example'] if i % 2 == 0 else ['example', 'baseline']
        jobs.extend(('retrieval', arm, row, None) for arm in arms)
    for i, (row, proposal) in enumerate(repair_cases()):
        arms = ['baseline', 'structured'] if i % 2 == 0 else ['structured', 'baseline']
        jobs.extend(('repair', arm, row, proposal) for arm in arms)
    # One global reservation shared across both independent experiments.
    for experiment, profile, row, proposal in jobs:
        used = copy.deepcopy(row)
        if profile == 'example': used['evidence'] = used['example_evidence']
        old_feedback = patch('sts2_advisor.decision.feedback.build_repair_feedback',
                             side_effect=lambda ctx, prop, err: err)
        factory = replay.create_remote_backend
        def seeded_factory(**kwargs):
            backend = factory(**kwargs)
            real_advise = backend.advise
            backend.advise = lambda ctx, ev, **kw: (copy.deepcopy(proposal) if not kw.get('feedback')
                                                   else real_advise(ctx, ev, **kw))
            return backend
        print(experiment, row['case_id'], profile, flush=True)
        with (patch.object(replay, 'create_remote_backend', side_effect=seeded_factory) if proposal else nullcontext()), \
             (nullcontext() if profile == 'structured' else old_feedback):
            result = replay.model_trial(used, False, budget, model_name='deepseek-flash')
        report['results'].append({'experiment': experiment, 'profile': profile, 'case_id': row['case_id'],
                                  'seeded_invalid_plan': proposal.to_dict() if proposal else None, **result})
        report['budget_used'] = {'requests': budget.requests, 'tokens_or_conservative_reserve': budget.tokens}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(' ', result['error'] or result['advice']['reason'], budget.tokens, flush=True)
        if '预算' in (result['error'] or '') or result['error'] and not result['model_responses']:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--output', type=Path, default=Path('tmp/decision-improvements-20261117-api.json'))
    parser.add_argument('--token-budget', type=int, default=200000)
    parser.add_argument('--request-budget', type=int, default=64)
    args = parser.parse_args(); logging.disable(logging.CRITICAL)
    if args.prepare: prepare(args.fixtures)
    elif args.run: run(args)
    else: print('离线；--prepare冻结6题，--run运行12首轮对照+4修复诊断，上限20万token，无游戏执行器')


if __name__ == '__main__': main()
