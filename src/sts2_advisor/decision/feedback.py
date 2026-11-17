"""Explain a rejected proposal without choosing or executing a replacement tactic."""
from __future__ import annotations

import json

from sts2_advisor.contracts import Advice, DecisionContext
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.combat.resources import known_resource_error


def build_repair_feedback(context: DecisionContext, proposal: Advice, error: str) -> str:
    if context.decision_type != 'combat_next_action':
        return error
    ids = (proposal.plan_candidate_ids or [proposal.candidate_id])[:8]
    trace: list[dict] = []
    known_resource_error(context, ids, trace=trace)
    options = []
    # These are legal starting points, never a menu of optimal tactical plans.
    # Validate against the unchanged snapshot, not the failed simulated balance.
    candidates = {c.candidate_id: c for c in context.candidates if c.legal}
    for ident in reversed(ids):
        candidate = candidates.get(ident)
        if candidate is None or candidate.action_type != 'play_card':
            continue
        checked = check_plan(context, [ident])
        resources: list[dict] = []
        resource_error = known_resource_error(context, [ident], trace=resources)
        if (checked.error or checked.candidate_ids != [ident] or resource_error or not resources
                or resources[0]['status'] != 'known_resource_step'):
            continue
        if any(r['candidate_ids'] == [ident] for r in options):
            continue
        options.append({'candidate_ids': [ident], 'label': candidate.label,
                        'energy_after': resources[0]['energy_after'],
                        'scope': '仅验证当前单步合法与资源；执行后重读，不代表战术优选'})
        if len(options) == 2:
            break
    return json.dumps({
        'error': error, 'actions_executed': False, 'decision_id': context.decision_id,
        'observed_energy': context.player.get('energy'),
        'previous_plan': ids, 'previous_reason': proposal.reason[:600],
        'resource_trace': trace,
        'trace_scope': '仅模拟，未知机制处停止；轨迹为空表示无法证明。当前观察未改变。',
        'repair_options': options,
        'instruction': '原提案未执行。请从当前完整观察重新决策；可选任意合法候选，不限于修复起点。仅重试这一次。',
    }, ensure_ascii=False, separators=(',', ':'))
