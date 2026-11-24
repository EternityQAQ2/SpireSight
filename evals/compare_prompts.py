"""Prepare frozen cases or run a bounded, paired DeepSeek prompt experiment.

Default is offline inspection. Only --prepare accesses local knowledge stores;
only --run loads API credentials. No game client or action executor is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path

from sts2_advisor.contracts import Candidate, DecisionContext, Evidence
from sts2_advisor.decision.prompts import build_prompt, build_system_prompt
from evals.prompt_experiment import (
    RequestBudget, build_compact_prompt, build_compact_system_prompt, inspect_advice, summarize,
)


FIXTURES = Path('evals/fixtures/prompt_cases.jsonl')
OLD = 'recordings/session-20261105-201253-679615.jsonl'
NEW = 'recordings/session-20261109-221829-452472.jsonl'
# Rubrics are for the reviewer, never for the model. Held-out cases are not used
# to edit the experimental prompts after API evaluation starts.
MANIFEST = [
    ('debuff-order', OLD, 40, 'development', '无攻击意图，不为格挡花费；比较易伤与攻击的顺序和总费用。'),
    ('last-energy', OLD, 65, 'development', '只有1费；显示伤害不重复加力量；不能再接另一张1费牌。'),
    ('partial-block', OLD, 69, 'development', '防御可少掉5血；攻击不能击杀，唯一可抽埋葬4费不可打；允许有依据的换血。'),
    ('slippery', OLD, 90, 'development', '攻击2乘3共6；滑溜逐段；飞剑非13伤斩杀；药水效果须计入。'),
    ('draw-first', OLD, 120, 'development', '灵魂0费抽2可先用；抽牌后停下；不能假设抽牌结果或漏掉绞缠。'),
    ('four-enemies', OLD, 150, 'development', '双重14加打击8可杀21血攻击虫；感染3；剥夺12不等于14；明确换血理由。'),
    ('reward-exhaust', OLD, 77, 'development', '比较当前贡献、2费和消耗手牌代价、0费输出与攻防；不得因稀有自动选。'),
    ('reward-scaling', OLD, 98, 'development', '全身撞击需要格挡；无情猛攻降费只对下一攻击；结合已有牌而非想象流派。'),
    ('rest-low', OLD, 108, 'development', '实际回24至56；根据近期掉血与升级收益比较，不机械阈值。'),
    ('rest-middle', OLD, 130, 'development', '49至73，比较具体升级且不虚构升级效果。'),
    ('event-spring', OLD, 133, 'development', '药水随机但免费；移除同时加入愧疚并非免费删牌。'),
    ('rest-boss', OLD, 170, 'development', '31至55；首领前风险与具体升级比较，不只说首领前必须休息。'),
    ('two-enemies', NEW, 21, 'heldout', '攻击4；防御5可覆盖；燃烧加强未来输出；抽牌选牌之后需观察。'),
    ('already-blocked', NEW, 23, 'heldout', '已有7盾覆盖4攻；防御额外收益需证明；不要重复加力量；抽牌后停止。'),
    ('zero-cost-kill', NEW, 28, 'heldout', '0能量仍可用0费打击10杀8血攻击甲虫，避免7伤；不能直接结束。'),
    ('cheap-lethal', NEW, 35, 'heldout', '打击10加愤怒10可杀18血；避免先随机消耗关键输出；不重复加力量。'),
    ('reward-defense', NEW, 38, 'heldout', '结合已有牌、费用与消耗评价30盾，说明与跳过比较，不因稀有自动选。'),
    ('rest-heldout', NEW, 42, 'heldout', '43至67，比较已知牌组和首领风险；未知升级收益不能写成事实。'),
    ('boss-opening', NEW, 45, 'heldout', '敌攻5加8共13；随从在首领死后逃离；坚毅随机消耗前先用关键牌；邪眼增强需满足消耗条件。'),
    ('boss-blocked', NEW, 50, 'heldout', '13盾覆盖13攻；额外盾无直接减伤，坚毅消耗可有长期价值；允许结束或有依据的消耗。'),
]
SMOKE = ['debuff-order', 'partial-block', 'four-enemies', 'reward-exhaust',
         'zero-cost-kill', 'cheap-lethal', 'boss-opening', 'reward-defense']


def restore(row: dict) -> tuple[DecisionContext, list[Evidence]]:
    raw = dict(row['context'])
    raw['candidates'] = [Candidate(**c) for c in raw['candidates']]
    return DecisionContext(**raw), [Evidence(**e) for e in row['evidence']]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def prepare(path: Path) -> None:
    from sts2_advisor.integrations.knowledge.factory import create_retriever
    from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
    from sts2_advisor.integrations.knowledge.embeddings import cached_model_path, EMBEDDING_MODEL
    # Avoid a hidden model download; report missing dependencies rather than
    # silently claiming the vector database was part of this experiment.
    if cached_model_path(EMBEDDING_MODEL) == EMBEDDING_MODEL:
        raise RuntimeError('本地嵌入模型未缓存；未开始实验，也未下载模型')
    retriever = create_retriever('auto', Path('data/chroma'), Path('data/knowledge.jsonl'))
    sqlite_retriever = retriever.inner
    vector = getattr(sqlite_retriever, 'fallback', None)
    if not isinstance(vector, ChromaRetriever):
        raise RuntimeError('auto 未启用向量检索；请检查数据库，避免把回退测试当成三路测试')
    rows_by_file = {}
    frozen = []
    for case_id, source, index, split, rubric in MANIFEST:
        if source not in rows_by_file:
            rows_by_file[source] = [json.loads(s) for s in Path(source).read_text(encoding='utf-8').splitlines()]
        original = rows_by_file[source][index]
        context, _ = restore(original)
        evidence = retriever.retrieve(context, 6)
        sources = Counter('seed' if e.evidence_id.startswith('builtin-') else
                          'sqlite' if e.evidence_id.startswith('sqlite:') else 'vector'
                          for e in evidence)
        frozen.append({
            'case_id': case_id, 'split': split, 'source': source, 'source_index': index,
            'context_sha256': digest(json.dumps(original['context'], ensure_ascii=False, sort_keys=True)),
            'rubric': rubric, 'context': original['context'],
            'evidence': [e.to_dict() for e in evidence], 'evidence_sources': dict(sources),
        })
        print(case_id, dict(sources), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in frozen), encoding='utf-8')
    audit = {'chain': ['SeedInjectingRetriever', 'SqliteFactsRetriever', 'ChromaRetriever'],
             'vector_documents': vector.collection.count(), 'top_k': 6, 'seeds_outside_top_k': True,
             'cases': [{'case_id': r['case_id'], 'sources': r['evidence_sources'],
                        'ids': [e['evidence_id'] for e in r['evidence']]} for r in frozen]}
    path.with_suffix('.audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run(rows: list[dict], args) -> dict:
    from sts2_advisor.cli import load_environment
    from tools.compare_thinking import run_case
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        raise RuntimeError('此实验只复用已授权的 DeepSeek 配置，不自动切换其他服务')
    logging.disable(logging.CRITICAL)
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    results = []
    report = {'model': os.environ['OPENAI_MODEL'], 'thinking': 'disabled', 'max_tokens': args.max_tokens,
              'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'seed_examples_sha256': digest(Path('evals/fixtures/seed_examples.jsonl').read_text(encoding='utf-8')),
              'limits': {'requests': args.request_budget, 'tokens': args.token_budget}, 'results': results}
    for pos, row in enumerate(rows):
        # Alternate order to reduce cache/order bias without changing case inputs.
        variants = ['current', 'compact'] if pos % 2 == 0 else ['compact', 'current']
        for variant in variants:
            print(f"{row['case_id']} {variant}…", flush=True)
            result = run_case(row, 'disabled', args.max_tokens, variant, budget=budget)
            if result.get('advice'):
                context, evidence = restore(row)
                result['checks'] = inspect_advice(context, evidence, result['advice'])
            results.append({'case_id': row['case_id'], 'split': row['split'], **result})
            report['budget_used'] = {'requests': budget.requests, 'tokens': budget.tokens}
            report['summary'] = summarize(results)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print('  ', result.get('error') or result['advice']['reason'], flush=True)
            if result.get('error') and not result.get('model_responses'):
                # A transport outage or exhausted budget cannot teach us about
                # decision quality. Keep incomplete coverage and stop spending.
                report['stopped_early'] = True
                report['missing_pairs'] = [r['case_id'] for r in rows
                    if sum(x['case_id'] == r['case_id'] for x in results) < 2]
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                return report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--case-ids', nargs='+', default=SMOKE)
    parser.add_argument('--request-budget', type=int, default=32)
    parser.add_argument('--token-budget', type=int, default=160000)
    parser.add_argument('--max-tokens', type=int, default=1600)
    parser.add_argument('--output', type=Path, default=Path('tmp/prompt-experiment.json'))
    args = parser.parse_args()
    if min(args.request_budget, args.token_budget, args.max_tokens) <= 0:
        parser.error('预算必须大于0')
    if args.prepare:
        prepare(args.fixtures)
        return
    rows = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines() if s.strip()]
    ids = [r['case_id'] for r in rows]
    if len(set(ids)) != len(ids) or len(set(args.case_ids)) != len(args.case_ids) or set(args.case_ids) - set(ids):
        parser.error('局面ID重复或不存在')
    rows = [next(r for r in rows if r['case_id'] == case_id) for case_id in args.case_ids]
    if args.run:
        if args.output.exists():
            parser.error('输出文件已存在；请指定新路径，以保留原始测试结果')
        report = run(rows, args)
        if report.get('stopped_early') or any(r.get('error') for r in report['results']):
            raise SystemExit(1)
        return
    for row in rows:
        context, evidence = restore(row)
        current = build_system_prompt() + build_prompt(context, evidence)
        compact = build_compact_system_prompt() + build_compact_prompt(context, evidence)
        print(row['case_id'], row['split'], row['evidence_sources'],
              f'chars: {len(current)} -> {len(compact)}')


if __name__ == '__main__':
    main()
