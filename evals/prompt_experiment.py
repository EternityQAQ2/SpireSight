"""Experimental prompts and read-only scoring; never used by the live advisor."""
from __future__ import annotations

import json
from pathlib import Path

from sts2_advisor.contracts import Advice, DecisionContext, Evidence
from sts2_advisor.decision.context import CodexCache, model_context_payload
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.decision.validation import validate_advice, AdviceValidationError


EXAMPLE_SEEDS = Path(__file__).parent / 'fixtures/seed_examples.jsonl'


def build_compact_system_prompt() -> str:
    return '''你是《杀戮尖塔2》决策建议器。依据当前状态和证据选择行动。
事实：当前手牌描述和费用优先于图鉴；显示伤害已含力量，不重复加算。stable.deck 是图鉴；player.deck_snapshot 是持牌观测，complete=false/stale=true 表示不完整/过时，不能臆造流派组件。hand/抽牌堆/弃牌堆/消耗堆不同。未知不是零。证据中的攻略和案例不保证适合当前状态。
战斗：比较可行方案的费用、确定击杀、格挡、生命损失和后续收益。非致死攻击本身不阻止敌人攻击；部分格挡仍能减伤。计入多段伤害、自伤和回合结束效果。换血须说明具体收益。没有攻击时，格挡应有明确转化/保留收益；尽可能有效利用能量。
计划：所有 ID 逐字来自 candidates，首项等于 candidate_id，共1至8步；同一手牌不能重复使用，费用不能透支，目标必须存活。只规划确定的连续动作；在抽牌、生成/消耗手牌、随机、选牌或不确定状态变化的动作后停止，等待新观察。确定无需继续行动时使用结束回合候选。
工具 calculate_attack 仅核对当前快照中单张牌的直接伤害；每次调用都从同一快照开始，结果不能相加当作序列试算。无法核对的效果保持未知。工具不负责选择战术。
非战斗：地图、事件、选牌、休息只走一步；购物/领奖可连续到余额不足或新选择。地图 nodes 为[列,行,类型,子节点]。选择界面不引用已选卡，满足要求时确认。选牌须依据已有牌，说明当前缺口、实际贡献、启动代价与跳过相比的收益；快照不足时说明不确定性。
只返回以下 JSON，reason 用不超过80个汉字比较一个备选；confidence为0至1，空数组用[]，其他局面selection_analysis用{}，证据ID只能引用输入：
{"decision_id":"输入值","candidate_id":"动作ID","reason":"理由","confidence":0.0,"alternative_ids":[],"evidence_ids":[],"warnings":[],"plan_candidate_ids":["动作ID"],"selection_analysis":{"gap":"当前缺口及依据","contribution":"此牌贡献和代价","skip_comparison":"与跳过相比"}}'''


def build_compact_prompt(context: DecisionContext, evidence: list[Evidence], *,
                         codex_cache: CodexCache | None = None) -> str:
    replacements = {r['evidence_id']: r['text'] for r in (
        json.loads(line) for line in EXAMPLE_SEEDS.read_text(encoding='utf-8').splitlines() if line.strip()
    )}
    payload = model_context_payload(context, codex_cache=codex_cache)
    payload['evidence'] = [
        {'id': e.evidence_id, 'title': e.source_title,
         'text': replacements.get(e.evidence_id, e.text)} for e in evidence
    ]
    payload['decision_id'] = context.decision_id
    return '输入 JSON：' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


class RequestBudget:
    """Shared across cases/variants, including tool requests. Reserve before I/O.

    UTF-8 bytes plus output ceiling conservatively bound ordinary text tokens;
    retain the reservation if a response has no trustworthy usage or fails.
    This is a token/request guard, not a monetary price calculation.
    """

    def __init__(self, *, max_requests: int, max_tokens: int):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.requests = 0
        self.tokens = 0
        self._reservation = 0

    def reserve(self, request: dict) -> None:
        if self.requests >= self.max_requests:
            raise RuntimeError('实验请求预算已耗尽')
        content = {'messages': request['messages'], 'tools': request.get('tools', [])}
        reserve = len(json.dumps(content, ensure_ascii=False).encode('utf-8')) + 128 + request['max_tokens']
        if self.tokens + reserve > self.max_tokens:
            raise RuntimeError('实验token预算不足，停止后续请求')
        self.requests += 1
        self.tokens += reserve
        self._reservation = reserve

    def record_usage(self, usage: dict | None) -> None:
        if isinstance(usage, dict):
            total = usage.get('total_tokens')
            if total is None and isinstance(usage.get('prompt_tokens'), int) and isinstance(usage.get('completion_tokens'), int):
                total = usage['prompt_tokens'] + usage['completion_tokens']
            if isinstance(total, int) and total >= 0:
                self.tokens += total - self._reservation
        self._reservation = 0


def inspect_advice(context: DecisionContext, evidence: list[Evidence], raw: dict) -> dict:
    """Inspect the ORIGINAL plan, before runtime validation shortens its prefix.

    Unknown-effect boundaries limit certification; no tactic is substituted.
    Tactical quality is reviewed separately against the fixture rubric.
    """
    advice = Advice(**{k: v for k, v in raw.items() if k in Advice.__dataclass_fields__})
    ids = advice.plan_candidate_ids or [advice.candidate_id]
    candidates = {c.candidate_id: c for c in context.candidates if c.legal}
    structural_error = None
    if not ids or ids[0] != advice.candidate_id or any(i not in candidates for i in ids):
        structural_error = '计划首项不一致或含非法候选'
    used = set()
    resource_error = None
    for pos, ident in enumerate(ids):
        candidate = candidates.get(ident)
        if candidate is None:
            continue
        if candidate.action_type == 'end_turn' and pos != len(ids) - 1:
            structural_error = '结束回合后仍有动作'
        if candidate.action_type == 'play_card':
            index = candidate.params.get('card_index')
            if index in used:
                resource_error = '原始计划重复使用同一张手牌'
            used.add(index)
    checked = check_plan(context, ids)
    validation_error = None
    try:
        validate_advice(advice, context, evidence)
    except AdviceValidationError as exc:
        validation_error = str(exc)
    return {
        'raw_plan': ids, 'checked_prefix': checked.candidate_ids,
        'structural_error': structural_error, 'resource_error': resource_error or checked.error,
        'validation_error': validation_error, 'boundary': checked.boundary,
        'boundary_crossed': bool(checked.boundary and checked.candidate_ids != ids),
        'boundary_kind': ('simulation_limit' if checked.boundary and
                          '当前机制无法精确试算' in checked.boundary else
                          'observation_change' if checked.boundary else None),
        'tactical_grade': 'pending_manual_review',
    }


def summarize(results: list[dict]) -> dict:
    """Mechanical metrics only; pending tactical reviews never count as passes."""
    summary = {}
    for variant in sorted({r['prompt_variant'] for r in results}):
        rows = [r for r in results if r['prompt_variant'] == variant]
        usages = [u for r in rows for u in r.get('usage', []) if isinstance(u, dict)]
        checks = [r['checks'] for r in rows if r.get('checks')]
        tools = [t for r in rows for t in r.get('tools', [])]
        summary[variant] = {
            'attempts': len(rows), 'responses': sum(bool(r.get('advice')) for r in rows),
            'requests': sum(r['requests'] for r in rows),
            'input_tokens': sum(u.get('prompt_tokens', 0) for u in usages),
            'output_tokens': sum(u.get('completion_tokens', 0) for u in usages),
            'cached_input_tokens': sum(u.get('prompt_cache_hit_tokens',
                                           (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)) or 0
                                       for u in usages),
            'seconds': round(sum(r['elapsed_seconds'] for r in rows), 3),
            'structural_errors': sum(bool(c['structural_error']) for c in checks),
            'resource_errors': sum(bool(c['resource_error']) for c in checks),
            'unverified_plan_suffixes': sum(c['boundary_crossed'] for c in checks),
            'tool_calls': len(tools),
            'unknown_tool_results': sum(t['result'].get('status') == 'unknown' for t in tools),
            'pending_tactical_reviews': sum(c['tactical_grade'] == 'pending_manual_review' for c in checks),
        }
    return summary
