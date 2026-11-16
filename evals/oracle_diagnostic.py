"""Small Flash diagnostic; no game actions, tools, feedback or live changes."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from unittest.mock import patch

from evals.compare_prompts import restore, digest
from evals.prompt_experiment import RequestBudget
from sts2_advisor.cli import create_remote_backend, load_environment
from sts2_advisor.decision.prompts import build_prompt, build_system_prompt
from sts2_advisor.decision.parsing import extract_json_object


FIXTURES = Path('evals/fixtures/oracle_diagnostic_cases.jsonl')
VARIANTS = ('state_only', 'plans_only', 'verified')


def make_cases():
    source = {r['case_id']: r for r in map(json.loads,
              Path('evals/fixtures/attack_delivery_cases.jsonl').read_text(encoding='utf-8').splitlines())}
    specs = [
        ('temporary-free-attack',
         [['card:0:NIBBIT_1'], ['card:2:NIBBIT_1']],
         [{'scope':'仅打出这一张后的立即结果', 'energy_cost':0, 'hp_after_action':66,
           'target_hp_after':0, 'free_attacks_remaining':0, 'energy_after':1},
          {'scope':'仅打出这一张后的立即结果', 'energy_cost':0, 'hp_after_action':64,
           'target_hp_after':0, 'free_attacks_remaining':0, 'energy_after':1}],
         '同样免费击杀3血目标，打击不自损，御血术自损2；两者都消耗免费攻击。自由规划可选择其他有具体收益的路线，不按未选菜单判错。'),
        ('vulnerable-lethal',
         [['card:1:FUZZY_WURM_CRAWLER_0'],
          ['card:2:FUZZY_WURM_CRAWLER_0','card:4:FUZZY_WURM_CRAWLER_0','combat:end_turn']],
         [{'scope':'单张击杀结束战斗', 'energy_cost':2, 'damage':21, 'target_hp_after':0,
           'incoming':0, 'burning_blood_heal':4, 'hp_after':80},
          {'scope':'两张打击后结束回合，仅按已显示攻击结算', 'energy_cost':2, 'damage':18,
           'target_hp_after':2, 'incoming':11, 'block':0, 'hp_after':65}],
         '易伤后14变21，一张无情猛攻杀20血敌；燃烧之血回复6受80上限限制实际回4。双打击各9合18，留下2血并受11攻击。'),
        ('thorn-tadpoles',
         [['card:3','card:4','card:2:TOADPOLE_0','combat:end_turn'],
          ['card:2:TOADPOLE_0','card:3','card:4','combat:end_turn']],
         [{'scope':'仅已显示9点攻击和2点荆棘，另一敌人强化的未显示影响未知',
           'energy_cost':3, 'target_hp_after':17, 'block_before_enemy':8,
           'immediate_hp_loss':0, 'hp_after_displayed_attacks':59},
          {'scope':'仅已显示9点攻击和2点荆棘，另一敌人强化的未显示影响未知',
           'energy_cost':3, 'target_hp_after':17, 'block_before_enemy':10,
           'immediate_hp_loss':2, 'hp_after_displayed_attacks':58}],
         '同卡同目标仅顺序不同：先双防御10，再攻击被荆棘扣2盾，挡9攻后掉1；先攻击直接掉2，再双防御可挡9攻。自由组杀无荆棘4血怪可更好，不当作菜单不匹配错误。'),
    ]
    cases = []
    for ident, plans, outcomes, rubric in specs:
        row = copy.deepcopy(source[ident])
        row.pop('rubric', None)
        row.update(plans=[{'candidate_ids': p, 'verified_outcome': o} for p,o in zip(plans,outcomes)],
                   preferred_plan=0, review_rubric=rubric,
                   diagnostic_role='previously_used_regression_not_heldout')
        cases.append(row)
    return cases


def presented_plans(case, repeat):
    plans = copy.deepcopy(case['plans'])
    return plans if repeat == 1 else list(reversed(plans))


def make_messages(case, variant, repeat):
    context, evidence = restore(case)
    system = build_system_prompt().replace('请使用已提供的 function tools；', '本次没有可调用工具；')
    prompt = build_prompt(context, evidence)
    if variant != 'state_only':
        plans = presented_plans(case, repeat)
        shown = [{'option':f'P{i+1}', 'candidate_ids': p['candidate_ids'],
                  **({'verified_outcome':p['verified_outcome']} if variant == 'verified' else {})}
                 for i,p in enumerate(plans)]
        prompt += ('\n本次仅作离线计划比较，不执行游戏动作。请只在下面两个计划中选一个，'
                   '按原建议JSON返回，candidate_id填所选计划首项，plan_candidate_ids逐字填完整计划；'
                   '不要用P1/P2作为动作ID，不截短或另造计划。未知后续保持未知。\n'
                   + json.dumps(shown, ensure_ascii=False, separators=(',',':')))
    return [{'role':'system','content':system},{'role':'user','content':prompt}]


def grade(case, variant, payload):
    if variant == 'state_only':
        return {'selection':'manual_review'}
    ids = payload.get('plan_candidate_ids') or [payload.get('candidate_id')]
    for index, p in enumerate(case['plans']):
        if ids == p['candidate_ids'] and payload.get('candidate_id', ids[0]) == ids[0]:
            return {'selection':'preferred' if index == case['preferred_plan'] else 'dominated'}
    return {'selection':'other_plan'}


def trial(case, variant, repeat, budget):
    backend = create_remote_backend(model_name='deepseek-flash', precompute_attacks=False)
    backend.client.max_retries = 0
    requests, responses = [], []
    original = backend.client.chat.completions.create
    def once(**kwargs):
        if requests:
            raise RuntimeError('诊断每项只允许一次模型请求')
        budget.reserve(kwargs)
        requests.append(copy.deepcopy(kwargs))
        response = original(**kwargs)
        raw = response.model_dump()
        responses.append(raw)
        budget.record_usage(raw.get('usage'))
        return response
    result = {'case_id':case['case_id'], 'variant':variant, 'repeat':repeat, 'error':None}
    try:
        if backend.thinking != 'disabled' or backend.adaptive_thinking or backend.max_tokens != 1024:
            raise ValueError('诊断要求普通模式/1024输出，与当前配置不符则停止')
        import time
        started = time.perf_counter()
        with patch.object(backend.client.chat.completions, 'create', side_effect=once):
            text, _, finish, _ = backend._complete(make_messages(case,variant,repeat),
                backend.max_tokens, case['context']['decision_id'], thinking='disabled')
        result.update(content=text, finish_reason=finish, elapsed_seconds=round(time.perf_counter()-started,4))
        payload = extract_json_object(text)
        result.update(payload=payload, grade=grade(case, variant, payload))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        backend.client.close()
    return {**result, 'requests':requests, 'responses':responses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('tmp/oracle-diagnostic.json'))
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.prepare:
        if FIXTURES.exists(): parser.error('题目已冻结，不覆盖')
        FIXTURES.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in make_cases()),encoding='utf-8')
        print('冻结3题，评分与后果分离，未调用API')
        return
    cases = [json.loads(s) for s in FIXTURES.read_text(encoding='utf-8').splitlines()]
    if not args.run:
        print('3题 × 3组 × 2次 = 18个单请求，上限150000 token；--run才付费')
        return
    if args.output.exists(): parser.error('输出已存在，请换新路径')
    load_environment(Path('.env'))
    from urllib.parse import urlsplit
    import os
    if urlsplit(os.environ.get('OPENAI_BASE_URL','')).hostname != 'api.deepseek.com':
        parser.error('仅使用当前已授权DeepSeek')
    budget = RequestBudget(max_requests=18, max_tokens=150000)
    report = {'fixtures_sha256':digest(FIXTURES.read_text(encoding='utf-8')),
              'limits':{'requests':18,'tokens':150000},
              'limitations':['3 previously used snapshots, not held out; 2 repetitions per group',
                             'menu groups constrain plan generation; compare plans_only vs verified for numeric aid',
                             'thorn baseline already contains partial turn_search outcomes',
                             'reference outcomes checked by assistant arithmetic and local facts, not independent human labels',
                             'no tools, validation feedback or production-service retries; not an online performance benchmark'],
              'results':[]}
    for repeat in (1,2):
        for pos,case in enumerate(cases):
            offset=(pos+repeat-1)%3
            for variant in VARIANTS[offset:]+VARIANTS[:offset]:
                print(repeat,case['case_id'],variant,flush=True)
                result=trial(case,variant,repeat,budget)
                report['results'].append(result)
                report['budget_used']={'requests':budget.requests,'tokens':budget.tokens}
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                print(' ',result.get('error') or result['payload'].get('reason'), 'budget',budget.tokens,flush=True)
                if result['error'] and not result['responses']:
                    raise SystemExit('请求/预算失败，保留记录并停止')


if __name__ == '__main__':
    main()
