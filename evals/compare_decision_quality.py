"""Bounded, action-focused comparison: old Flash / clarified Flash / clarified Pro.

Reuses the production-service replay, shared SDK budget and configuration factory.
Default is offline; --prepare reads local stores, --run spends API tokens.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from evals.compare_attack_delivery import model_trial, replay_decision, mechanical_summary
from evals.compare_prompts import digest
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import load_environment
from sts2_advisor.contracts import Advice


BASELINE = 'c1dd6b6'
FIXTURES = Path('evals/fixtures/decision_quality_cases.jsonl')
PROFILES = ('old_flash', 'clarified_flash', 'clarified_pro')
# Frozen before paid calls. These are diagnostic regressions, not an independent
# held-out benchmark. The two new recorded states are explicitly labeled.
MANIFEST = [
    ('thorn-target', '20261104-231325-155591', 21, 'regression',
     '两防御共10盾挡9攻，剑柄杀4血无荆棘者后停在抽牌边界；先剑柄抽牌再观察也合理。对23血怪攻击需计2荆棘。不能把无荆棘目标算成有荆棘。'),
    ('temporary-free', '20261109-214328-690834', 0, 'regression',
     '只有下1张攻击免费，之后费用需重读；打击/御血术基础1费、痛击2费。接受先杀3血攻击者或有具体攻防收益的其他顺序，不要求永远先防御。原始超费后缀独立计错；御血术自损2。'),
    ('certain-lethal', '20261109-212300-429763', 36, 'regression',
     '无情猛攻14经易伤21可杀唯一20血敌人，防御后仍够2费击杀也可接受。两张打击各9共18留2血，不应以不可杀组合替代确定无损击杀。'),
    ('four-enemies', '20261105-201253-679615', 150, 'regression',
     '集中双重14加打击8可杀攻击者，感染另3伤；同费用少吃6攻击应优先考虑。若杀强化者需给出具体长期收益，不凭压血线当减伤。剥夺牌面12，不等于14；不要求最低当前掉血为唯一策略。'),
    ('partial-block', '20261105-201253-679615', 69, 'regression',
     '最后1费，5盾仍可省5血；攻击不能击杀。允许有具体依据的长期换血；不把非致命攻击当作减少当前伤害。已知只剩4费埋葬不能保证抽出后本回合可打。'),
    ('boss-potion', '20261105-201253-679615', 190, 'new_recording_state',
     '9HP面对9+8+10=27攻；灵魂0费抽2后观察是有效求生手段。药水可减目标未来4回合攻击30%，上勾拳虚弱只作用指定敌人。不得仅双防御后结束并保证存活；不得将单体药水当全体减伤。抽牌未知，不保证最终可活。'),
    ('reward-draw', '20261104-224452-412875', 46, 'new_recording_state',
     '结合实际观测的持牌、费用、缺口和跳过分析选战栗/耸肩无视/惊逃；接受多种有依据选择，不凭稀有度或虚构流派组件。历史快照不完整不能称拥有某张未观测卡。'),
    ('rest-before-boss', '20261105-201253-679615', 170, 'regression',
     '31HP休息回24至55；考虑首领风险和实际升级收益。休息合理；锻造若有具体可验证收益也可接受。不仅凭固定阈值评分，不把未知升级效果说成事实。'),
]


def load_baseline(revision=BASELINE):
    """Freeze the four input modules; no second implementation of game rules."""
    def module(name):
        source = subprocess.check_output(
            ['git', '-c', f'safe.directory={Path.cwd().as_posix()}', 'show',
             f'{revision}:src/sts2_advisor/{name}.py'], encoding='utf-8')
        loaded = ModuleType('baseline_' + name.replace('/', '_'))
        loaded.__file__ = str((Path('src/sts2_advisor') / (name + '.py')).resolve())
        exec(compile(source, f'{revision}/{name}.py', 'exec'), loaded.__dict__)
        return loaded

    facts = module('game/combat/facts')
    metrics = module('decision/metrics')
    for name, value in vars(facts).items():
        if not name.startswith('_') and name in vars(metrics):
            vars(metrics)[name] = value
    context = module('decision/context')
    context.combat_summary = facts.combat_summary
    context.dynamic_card_metrics = metrics.dynamic_card_metrics
    prompts = module('decision/prompts')
    prompts.model_context_payload = context.model_context_payload
    return SimpleNamespace(model_context_payload=context.model_context_payload,
                           build_prompt=prompts.build_prompt, build_system_prompt=prompts.build_system_prompt)


def prepare(path):
    from sts2_advisor.integrations.knowledge.factory import create_retriever
    from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
    from sts2_advisor.integrations.knowledge.embeddings import cached_model_path, EMBEDDING_MODEL

    os.environ['HF_HUB_OFFLINE'] = '1'
    if cached_model_path(EMBEDDING_MODEL) == EMBEDDING_MODEL:
        raise RuntimeError('本地嵌入模型未缓存；未下载或调用API')
    retriever = create_retriever('auto', Path('data/chroma'), Path('data/knowledge.jsonl'))
    if not isinstance(getattr(retriever.inner, 'fallback', None), ChromaRetriever):
        raise RuntimeError('向量检索不可用，停止准备')

    class PreparationBackend:
        def advise(self, ctx, ev):
            ident = ctx.candidates[0].candidate_id
            return Advice(ctx.decision_id, ident, '冻结局面，不操作游戏', 1.0,
                          plan_candidate_ids=[ident], selection_analysis={
                              'gap': '仅冻结输入', 'contribution': '不评价策略', 'skip_comparison': '不发送动作'})

    old_ids = {json.loads(s)['context']['decision_id'] for p in
               ('evals/fixtures/prompt_cases.jsonl', 'evals/fixtures/attack_delivery_cases.jsonl')
               for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()}
    rows = []
    for case_id, session, index, role, rubric in MANIFEST:
        source = Path(f'recordings/session-{session}.jsonl')
        original = json.loads(source.read_text(encoding='utf-8').splitlines()[index])
        if role == 'new_recording_state' and original['context']['decision_id'] in old_ids:
            raise ValueError('新增局面与旧题重复')
        result = replay_decision(original, PreparationBackend(), retriever=retriever)
        if result['error']:
            raise RuntimeError(result['error'])
        request = next(e for e in result['application_events'] if e['type'] == 'retrieval.request')
        rows.append({'case_id': case_id, 'source': str(source), 'source_index': index,
                     'role': role, 'rubric': rubric,
                     'context_sha256': digest(json.dumps(original['context'], ensure_ascii=False, sort_keys=True)),
                     'retrieval_top_k': request['top_k'], 'raw_state': original['raw_state'],
                     'context': original['context'], 'evidence': result['evidence']})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def summarize_reviews(results, reviews):
    """Keep action quality, rejected suffixes and prose errors independent."""
    indexed = {(r['case_id'], r['repeat'], r['profile']): r for r in reviews}
    if len(indexed) != len(reviews):
        raise ValueError('复核条目重复')
    summary = {}
    for profile in sorted({r['profile'] for r in results}):
        rows = [r for r in results if r['profile'] == profile]
        reviewed = [indexed[(r['case_id'], r['repeat'], profile)] for r in rows
                    if (r['case_id'], r['repeat'], profile) in indexed]
        summary[profile] = {
            'attempts': len(rows), 'responses': sum(bool(r.get('advice')) for r in rows),
            'acceptable_actions': sum(r['action_grade'] == 'acceptable' for r in reviewed),
            'explicit_action_mistakes': sum(r['action_grade'] == 'explicit_mistake' for r in reviewed),
            'unresolved_actions': sum(r['action_grade'] == 'unresolved' for r in reviewed),
            'no_responses': sum(r['action_grade'] == 'no_response' for r in reviewed),
            'not_sent': sum(r['action_grade'] == 'not_sent' for r in reviewed),
            'unreviewed': len(rows) - len(reviewed),
            'answers_with_reason_errors': sum(bool(r['reason_errors']) for r in reviewed),
            'answers_with_raw_plan_errors': sum(bool(r['raw_plan_errors']) for r in reviewed),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--output', type=Path, default=Path('tmp/decision-quality.json'))
    parser.add_argument('--token-budget', type=int, default=500000)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if not 0 < args.token_budget <= 500000:
        parser.error('本轮总预算最多500000 token')
    if args.prepare:
        if args.fixtures.exists():
            parser.error('输入已冻结，请勿覆盖')
        prepare(args.fixtures)
        print('已冻结8题及评分标准，未调用模型')
        return
    rows = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not args.run:
        print('8题 × 3配置 × 2次 = 48决策，--run才付费；只有人工复核才评价行动质量')
        return
    if args.output.exists() or len(rows) != 8:
        parser.error('输出已存在或题目数量错误')
    load_environment(Path('.env'))
    if 'deepseek.com' not in os.environ.get('OPENAI_BASE_URL', '').lower():
        parser.error('本轮只使用已授权的DeepSeek')
    old = load_baseline()
    budget = RequestBudget(max_requests=96, max_tokens=args.token_budget)
    report = {'baseline_revision': BASELINE, 'fixtures_sha256': digest(args.fixtures.read_text(encoding='utf-8')),
              'expected_decisions': 48, 'repeats': 2, 'limits': {'tokens': args.token_budget, 'requests': 96},
              'pipeline': 'production AdvisorService, frozen context/evidence, no executor', 'results': []}
    for repeat in range(2):
        for pos, row in enumerate(rows):
            offset = (repeat + pos) % len(PROFILES)
            for profile in PROFILES[offset:] + PROFILES[:offset]:
                baseline = profile == 'old_flash'
                print(repeat + 1, row['case_id'], profile, flush=True)
                with (patch('sts2_advisor.integrations.models.remote.build_prompt', old.build_prompt)
                      if baseline else nullcontext()), \
                     (patch('sts2_advisor.integrations.models.remote.build_system_prompt', old.build_system_prompt)
                      if baseline else nullcontext()):
                    result = model_trial(row, False, budget,
                                         model_name='deepseek-v4-pro' if profile == 'clarified_pro' else 'deepseek-flash')
                report['results'].append({'case_id': row['case_id'], 'repeat': repeat + 1,
                                          'profile': profile, **result})
                report['summary'] = mechanical_summary(report['results'])
                report['budget_used'] = {'requests': budget.requests, 'tokens': budget.tokens}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                print(' ', result['error'] or result['advice']['reason'], 'budget', budget.tokens, flush=True)
                if result['error'] and not result['model_responses']:
                    raise SystemExit('请求失败或预算不足；保留原始结果并停止')


if __name__ == '__main__':
    main()
