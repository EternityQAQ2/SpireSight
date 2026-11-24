"""Conservative end-turn certificates using the existing simulator, without I/O.

Only reject a lethal projection when a deterministic surviving path is also
verified. A failed bounded search never proves that death is unavoidable.
"""
from __future__ import annotations

from sts2_advisor.contracts import DecisionContext
from sts2_advisor.game.combat.facts import ATTACK_DAMAGE_RE
from sts2_advisor.game.combat.search import rank_turns
from sts2_advisor.game.combat.simulation import initial_state, play, outcome


def _unknown_input(context: DecisionContext) -> str | None:
    player = context.player
    enemies = (context.situation.get('battle') or {}).get('enemies')
    if not isinstance(enemies, list) or not enemies:
        return '敌人信息不完整'
    for entity, fields in [(player, ('hp', 'block', 'energy'))] + [
            (e, ('hp', 'block')) for e in enemies]:
        if any(type(entity.get(k)) is not int or entity[k] < 0 for k in fields):
            return '生命、格挡或能量未知'
        statuses = entity.get('status')
        if not isinstance(statuses, list) or any(
                not isinstance(s, dict) or type(s.get('amount')) is not int for s in statuses):
            return '状态或层数未知'
        if any(s['amount'] < 0 and str(s.get('id', '')).upper().removesuffix('_POWER')
               not in {'STRENGTH', 'DEXTERITY'} for s in statuses):
            return '状态层数无效'
    for enemy in enemies:
        intents = enemy.get('intents')
        if not isinstance(intents, list) or not intents:
            return '敌人意图未知'
        if any(i.get('type') not in {'Attack', 'Defend'} for i in intents):
            return '非攻击/格挡意图尚未支持回合结算'
        for intent in intents:
            if intent.get('type') != 'Attack':
                continue
            text = ''.join(str(intent.get('description') or '').split())
            for prefix in ('这个敌人将要攻击', '这个敌人将要', '将要攻击', '将要'):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            text = text.removesuffix('。').removesuffix('.')
            if ATTACK_DAMAGE_RE.fullmatch(text) is None:
                return '攻击伤害或次数未完整解析'
    for card in player.get('hand') or []:
        if card.get('type') in {'Status', 'Curse'} or '回合结束' in str(card.get('description') or ''):
            return '手牌回合结束效果尚未支持'
    for relic in player.get('relics') or []:
        if relic.get('id') in {'PEN_NIB', 'KUSARIGAMA'} and type(relic.get('counter')) is not int:
            return '遗物计数未知'
    return None


def project_end_turn(context: DecisionContext, ids: list[str]) -> dict:
    """Certify an explicit end-turn plan or a deterministic combat-ending path."""
    if context.decision_type != 'combat_next_action' or not ids:
        return {'status': 'not_ending_turn'}
    actions = {c.candidate_id: c for c in context.candidates if c.legal}
    if ids[-1] not in actions or actions[ids[-1]].action_type != 'end_turn':
        # Search witnesses can end by killing the last enemy, without end_turn.
        ending = False
    else:
        ending = True
    try:
        reason = _unknown_input(context)
        if reason:
            return {'status': 'unknown', 'reason': reason} if ending else {'status': 'not_ending_turn'}
        initial = initial_state(context)
        if initial is None:
            return {'status': 'unknown', 'reason': '当前机制超出模拟器支持范围'}
        state = initial
        cards = {c.get('index'): c for c in context.player.get('hand') or []}
        for pos, ident in enumerate(ids):
            candidate = actions.get(ident)
            if candidate is None:
                return {'status': 'unknown', 'reason': '不可用动作'}
            if candidate.action_type == 'end_turn':
                if pos != len(ids) - 1:
                    return {'status': 'unknown', 'reason': '结束回合后仍有动作'}
                return {'status': 'calculated', **outcome(state, initial).to_dict()}
            if candidate.action_type != 'play_card':
                return {'status': 'unknown', 'reason': '药水等动作需要重新观察'}
            card = cards.get(candidate.params.get('card_index'))
            if (card is None or str(card.get('cost', '')).upper() == 'X' or
                    card.get('star_cost') not in (None, 0, '0')):
                return {'status': 'unknown', 'reason': '费用或辉星结算未知'}
            state = play(state, candidate, card)
            if state is None or state.boundary or state.hp <= 0:
                return {'status': 'unknown', 'reason': '动作或抽牌等边界无法完整预测'}
            if not any(e.hp > 0 for e in state.enemies):
                if pos == len(ids) - 1 or pos == len(ids) - 2 and ending:
                    return {'status': 'calculated', **outcome(state, initial).to_dict()}
                return {'status': 'unknown', 'reason': '战斗结束后仍有动作'}
        return {'status': 'not_ending_turn'}
    except (TypeError, ValueError, KeyError, AttributeError):
        return {'status': 'unknown', 'reason': '局面字段无法可靠解析'}


def check_end_turn(context: DecisionContext, ids: list[str]) -> dict:
    """Return proof/feedback, never a replacement action or strategy."""
    candidates = {c.candidate_id: c for c in context.candidates}
    if not ids or ids[-1] not in candidates or candidates[ids[-1]].action_type != 'end_turn':
        return {'status': 'not_ending_turn'}
    forecast = project_end_turn(context, ids)
    if forecast.get('status') != 'calculated' or forecast['hp_after'] > 0:
        return forecast
    search = rank_turns(context, max_nodes=256)
    for line in search.get('lines', []):
        if line['requires_observation'] or line['hp_after'] <= 0:
            continue
        witness = project_end_turn(context, line['candidate_ids'])
        if witness.get('status') == 'calculated' and witness['hp_after'] > 0:
            return {**forecast, 'status': 'retry', 'feedback': (
                f"该计划结束回合预计生命{forecast['hp_after']}（剩余敌人攻击{forecast['incoming']}，"
                f"剩余格挡{forecast['block']}）。已验证至少一条非致死出牌路径，"
                '请重新决策；程序不会替你选择打法。')}
    return {**forecast, 'status': 'lethal_without_verified_escape'}
