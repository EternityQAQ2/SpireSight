"""Compare tool roundtrips with inline attack results through the real service.

Default is offline. --prepare freezes local three-source retrieval. Only --run
uses the authorized API; neither mode creates a game client or action executor.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch

from evals.compare_prompts import digest, restore
from evals.prompt_experiment import RequestBudget, inspect_advice, summarize
from sts2_advisor.application.advisor import AdvisorService
from sts2_advisor.cli import create_remote_backend, load_environment
from sts2_advisor.contracts import Advice


FIXTURES = Path('evals/fixtures/attack_delivery_cases.jsonl')
# Rubrics are frozen BEFORE paid testing and are never part of model input.
MANIFEST = [
    ('slime-targets', '20261104-220148-771355', 9,
     '5血攻击者可用一张打击击杀，再两张防御覆盖8攻；也可有依据换血。未知遗物不当0，不虚构击杀。'),
    ('last-energy-sludge', '20261104-231325-155591', 5,
     '只剩1费；防御减5血，打击不能击杀；允许具体长期换血理由，不许虚称攻击减少当前伤害。'),
    ('thorn-tadpoles', '20261104-231325-155591', 21,
     '一敌23血带荆棘、攻3乘3，另一敌4血准备强化；攻击带荆棘目标计反伤；抽牌后重新观察。'),
    ('shielded-nibbit', '20261104-224452-412875', 43,
     '敌12血5盾易伤1且不攻击。重击36可杀；铁斩波7再打击8也可杀。不为纯格挡出防御，不把各独立结果拼成连续剩血。'),
    ('vulnerable-lethal', '20261109-212300-429763', 36,
     '敌20血易伤1、攻11；无情猛攻牌面14乘1.5为21，单张可杀；不可重复加修正，不需要额外防御。'),
    ('temporary-free-attack', '20261109-214328-690834', 0,
     '1能量，下次攻击免费；敌42血攻14与3血攻6。零费打击可杀3血攻击者；当前免费费用不能保证全部后续攻击仍免费；血债自损2。'),
]


def replay_decision(row: dict, backend, *, retriever=None) -> dict:
    """Run production retrieval/validation/redecision, retaining original proposals."""
    context, evidence = restore(copy.deepcopy(row))
    events, recorded = [], []

    class RecordedClient:
        def get_singleplayer_state(self):
            return copy.deepcopy(row['raw_state'])

    class FrozenRetriever:
        def retrieve(self, ctx, top_k=6):
            if top_k != row.get('retrieval_top_k', 4):
                raise ValueError('回放检索top_k与冻结输入不一致')
            return evidence

    class Recorder:
        def record(self, raw, ctx, ev, advice, name):
            recorded.append(advice.to_dict())

    def capture(kind, **data):
        events.append({'type': kind, **copy.deepcopy(data)})

    service = AdvisorService(
        RecordedClient(), retriever or FrozenRetriever(), backend, Recorder(),
        backend_name='openai', top_k=6, action_executor=None,
        state_builder=lambda raw: context, output=lambda text: None,
    )
    started = time.perf_counter()
    error = None
    with patch('sts2_advisor.application.advisor.record_event', capture):
        try:
            service.run_once(force=True)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    proposals = [e['advice'] for e in events if e['type'] == 'decision.proposal']
    used_evidence = next((e['evidence'] for e in events if e['type'] == 'retrieval.result'), [])
    _, ev = restore({'context': row['context'], 'evidence': used_evidence})
    return {
        'advice': recorded[-1] if recorded else None, 'error': error,
        'elapsed_seconds': round(time.perf_counter() - started, 4),
        'proposals': proposals, 'application_events': events, 'evidence': used_evidence,
        'checks': [inspect_advice(context, ev, p) for p in proposals],
    }


def prepare(path: Path):
    from sts2_advisor.integrations.knowledge.factory import create_retriever
    from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
    from sts2_advisor.integrations.knowledge.embeddings import cached_model_path, EMBEDDING_MODEL

    os.environ['HF_HUB_OFFLINE'] = '1'
    if cached_model_path(EMBEDDING_MODEL) == EMBEDDING_MODEL:
        raise RuntimeError('缺少本地嵌入模型；未下载或开始实验')
    retriever = create_retriever('auto', Path('data/chroma'), Path('data/knowledge.jsonl'))
    if not isinstance(getattr(retriever.inner, 'fallback', None), ChromaRetriever):
        raise RuntimeError('向量检索不可用，不能把回退当成三路检索')

    class PreparationBackend:
        def advise(self, ctx, ev):
            ident = ctx.candidates[0].candidate_id
            return Advice(ctx.decision_id, ident, '冻结输入，无游戏动作', 1.0,
                          plan_candidate_ids=[ident])

    old_ids = {json.loads(s)['context']['decision_id'] for s in
               Path('evals/fixtures/prompt_cases.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()}
    rows = []
    for case_id, session, index, rubric in MANIFEST:
        source = Path(f'recordings/session-{session}.jsonl')
        row = json.loads(source.read_text(encoding='utf-8').splitlines()[index])
        if row['context']['decision_id'] in old_ids:
            raise ValueError('新题与旧20题重叠')
        result = replay_decision(row, PreparationBackend(), retriever=retriever)
        if result['error']:
            raise RuntimeError(result['error'])
        request = next(e for e in result['application_events'] if e['type'] == 'retrieval.request')
        rows.append({
            'case_id': case_id, 'source': str(source), 'source_index': index, 'rubric': rubric,
            'context_sha256': digest(json.dumps(row['context'], ensure_ascii=False, sort_keys=True)),
            'retrieval_top_k': request['top_k'], 'raw_state': row['raw_state'],
            'context': row['context'], 'evidence': result['evidence'],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def model_trial(row: dict, inline: bool, budget: RequestBudget, *, model_name: str | None = None) -> dict:
    backend = create_remote_backend(precompute_attacks=inline, model_name=model_name)
    backend.client.max_retries = 0
    if backend.thinking != 'disabled' or backend.adaptive_thinking:
        backend.client.close()
        raise ValueError('本轮仅批准普通模式，不覆盖线上配置后偷偷调用')
    requests, responses, tools = [], [], []
    original = backend.client.chat.completions.create

    def complete(**kwargs):
        # Up to two model decisions, each with at most one tool/length roundtrip.
        if len(requests) >= 4:
            raise RuntimeError('单局面请求预算耗尽')
        budget.reserve(kwargs)
        requests.append(copy.deepcopy(kwargs))
        response = original(**kwargs)
        raw = response.model_dump()
        budget.record_usage(raw.get('usage'))
        responses.append(raw)
        return response

    def capture(kind, **data):
        if kind == 'model.tool_result':
            tools.append(copy.deepcopy(data))

    try:
        with patch.object(backend.client.chat.completions, 'create', side_effect=complete), \
             patch('sts2_advisor.integrations.models.remote.record_event', capture):
            result = replay_decision(row, backend)
        return {**result, 'requests': len(requests), 'request_details': requests,
                'model_responses': responses, 'tools': tools,
                'usage': [r.get('usage') for r in responses],
                'config': {'model': backend.model_name, 'max_tokens': backend.max_tokens,
                           'thinking': backend.thinking, 'adaptive_thinking': backend.adaptive_thinking,
                           'precompute_attacks': inline}}
    finally:
        backend.client.close()


def mechanical_summary(results):
    # Existing summarizer accepts one check per decision; preserve ALL originals
    # separately so a corrected retry does not erase the first bad proposal.
    summary = summarize([{**r, 'prompt_variant': r['profile'],
                          'checks': r['checks'][-1] if r['checks'] else None} for r in results])
    for profile, item in summary.items():
        rows = [r for r in results if r['profile'] == profile]
        item['total_tokens'] = item['input_tokens'] + item['output_tokens']
        item['tokens_per_success'] = round(item['total_tokens'] / item['responses'], 2) if item['responses'] else None
        item['mean_seconds'] = round(item['seconds'] / len(rows), 3)
        item['redecisions'] = sum(max(0, len(r['proposals']) - 1) for r in rows)
        item['pending_manual_reviews'] = len(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--output', type=Path, default=Path('tmp/attack-delivery.json'))
    parser.add_argument('--token-budget', type=int, default=200000)
    parser.add_argument('--request-budget', type=int, default=48)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.prepare:
        if args.fixtures.exists():
            parser.error('冻结题目已存在，请勿覆盖')
        prepare(args.fixtures)
        print('已冻结6题，检索经过AdvisorService，未调用模型或游戏')
        return
    rows = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not args.run:
        print('6题 × 2配置 × 2次 = 24决策；--run才付费；top_k', sorted({r['retrieval_top_k'] for r in rows}))
        return
    if args.output.exists():
        parser.error('输出已存在，请使用新路径')
    if len(rows) != 6 or not 0 < args.token_budget <= 200000 or not 0 < args.request_budget <= 48:
        parser.error('本轮限6题、20万token、48请求')
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        parser.error('仅使用已授权的DeepSeek提供商')
    budget = RequestBudget(max_requests=args.request_budget, max_tokens=args.token_budget)
    report = {'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'pipeline': 'AdvisorService.run_once(force=True), frozen observed context, no executor',
              'repetitions': 2, 'expected_decisions': 24,
              'limits': {'tokens': args.token_budget, 'requests': args.request_budget}, 'results': []}
    for repeat in range(2):
        for pos, row in enumerate(rows):
            profiles = [('tools', False), ('inline', True)]
            if (repeat + pos) % 2:
                profiles.reverse()
            for profile, inline in profiles:
                print(repeat + 1, row['case_id'], profile, flush=True)
                result = model_trial(row, inline, budget)
                report['results'].append({'case_id': row['case_id'], 'repeat': repeat + 1,
                                          'profile': profile, **result})
                report['summary'] = mechanical_summary(report['results'])
                report['budget_used'] = {'requests': budget.requests, 'tokens': budget.tokens}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                print(' ', result['error'] or result['advice']['reason'], 'budget', budget.tokens, flush=True)
                if result['error'] and not result['model_responses']:
                    raise SystemExit('请求失败或预算不足，保留结果并停止')


if __name__ == '__main__':
    main()
