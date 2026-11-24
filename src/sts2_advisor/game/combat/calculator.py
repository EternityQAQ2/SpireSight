"""Grounded, single-card arithmetic. No game actions or full-turn simulation."""
from __future__ import annotations

import re
from sts2_advisor.contracts import DecisionContext
from sts2_advisor.game.combat.facts import card_hp_loss, card_block
from sts2_advisor.game.combat.effects import known_effect


DIRECT_ATTACK_NEUTRAL_RELICS = {"BURNING_BLOOD", "NEOWS_TALISMAN", "SCROLL_BOXES", "STONE_HUMIDIFIER",
                  "BOOK_OF_FIVE_RINGS", "KALEIDOSCOPE", "LANTERN", "POTION_BELT", "MEAL_TICKET",
                  # Pickup inventory; first POWER block; end-turn-seven damage.
                  # None modifies this Attack card's direct hit on its target.
                  "PHIAL_HOLSTER", "PERMAFROST", "STONE_CALENDAR"}

DISPLAYED_PLAYER_MODIFIERS = {"STRENGTH", "WEAK", "DEXTERITY", "FRAIL", "SHRINK", "FLEX_POTION", "FREE_ATTACK"}


def _unknown(reason: str) -> dict:
    return {"ok": True, "status": "unknown", "reason": reason,
            "player_outcome": {"status": "unknown", "hp_after": None, "block_after": None}}


def _player_outcome(context: DecisionContext, card: dict, target: dict, defeated: bool) -> dict:
    """Narrow immediate effects only; do not create another turn simulator."""
    def unknown(reason):
        return {"status": "unknown", "hp_after": None, "block_after": None, "reason": reason}

    player = context.player
    if any(type(player.get(k)) is not int or player[k] < 0 for k in ('hp', 'block')):
        return unknown('玩家生命或格挡未知')
    text = re.sub(r'\s+', '', str(card.get('description') or ''))
    hemokinesis = card.get('id') == 'HEMOKINESIS' and re.fullmatch(r'失去\d+点生命。造成\d+点伤害。', text)
    if not hemokinesis and known_effect(card) not in {'strike', 'bash', 'iron_wave', 'twin_strike'}:
        return unknown('该牌的额外效果未验证，未计算玩家结算')
    for held in player.get('hand', []):
        if held.get('type') not in {'Status', 'Curse'}:
            continue
        # This observed infection triggers at end turn, outside this scope.
        harmless_now = held.get('id') == 'INFECTION' and re.fullmatch(
            r'不能被打出。在你的回合结束时，如果这张牌在你的手牌中，你受到\d+点伤害。',
            re.sub(r'\s+', '', str(held.get('description') or '')))
        if not harmless_now:
            return unknown('手牌可能存在打牌时触发的未验证效果')
    enemies = (context.situation.get('battle') or {}).get('enemies', [])
    for enemy in enemies:
        if enemy is target:
            continue
        statuses = enemy.get('status')
        if not isinstance(statuses, list) or any(
                not isinstance(s, dict) or str(s.get('id', '')).upper().removesuffix('_POWER')
                not in {'VULNERABLE','SLOW','SLIPPERY','MINION','STRENGTH','WEAK','ARTIFACT','THORNS'}
                for s in statuses):
            return unknown('其他敌人的触发效果未验证')
    if defeated and (not any(e is not target and (e.get('hp') or 0) > 0 for e in enemies) or
                     any(str(s.get('id', '')).upper().removesuffix('_POWER') == 'MINION'
                         for e in enemies for s in e.get('status') or [])):
        return unknown('可能结束战斗；战后回血、离场结算未预测')
    loss = card_hp_loss(card)
    if loss is None:
        return unknown('该牌直接生命消耗未知')
    return {'status': 'calculated', 'hp_after': player['hp'] - loss,
            'block_after': player['block'] + card_block(card),
            'scope': '仅当前单张直接结算；不含敌人回合和回合末；每次从同一快照开始'}


def target_modifiers(target: dict) -> list[dict]:
    """Apply the currently observed target multipliers, in a stable order."""
    statuses = {str(s.get("id", "")).upper().removesuffix("_POWER"): s for s in target.get("status", [])}
    result = []
    if int(statuses.get("VULNERABLE", {}).get("amount") or 0) > 0:
        result.append({"type": "multiply", "numerator": 3, "denominator": 2})
    if "SLOW" in statuses:
        amount = statuses["SLOW"].get("amount")
        if type(amount) is int and 0 <= amount <= 1000:
            result.append({"type": "multiply", "numerator": 100 + amount, "denominator": 100})
    return result


def calculate(context: DecisionContext, candidate_id: str, expected_hp_after: int | None = None) -> dict:
    candidate = next((c for c in context.candidates if c.candidate_id == candidate_id and c.legal), None)
    if candidate is None or candidate.action_type != "play_card":
        return {"ok": False, "error": "动作不是当前可用出牌"}
    card = next((c for c in context.player.get("hand", []) if c.get("index") == candidate.params.get("card_index")), None)
    target = next((e for e in (context.situation.get("battle") or {}).get("enemies", [])
                   if e.get("entity_id") == candidate.params.get("target")), None)
    if not card or card.get("type") != "Attack" or not target:
        return {"ok": False, "error": "仅支持指定目标的单张攻击牌"}
    text = str(card.get("description") or "")
    match = re.search(r"造成(\d+)点伤害(?:(两|\d+)次)?", text.replace(" ", ""))
    if not match or "随机" in text or "所有敌人" in text:
        return _unknown("攻击数值、次数或目标不确定")
    life_cost = card_hp_loss(card)
    if (life_cost is not None and life_cost > 0 and type(context.player.get('hp')) is int
            and context.player['hp'] <= life_cost):
        return _unknown('自损可能先导致死亡，无法保证攻击结算')
    hits = 2 if match.group(2) == "两" else int(match.group(2) or 1)
    reasons = []
    for field in ('hp', 'block'):
        if type(target.get(field)) is not int or target[field] < 0:
            reasons.append('目标' + field + '未知')
    # Current hand damage/cost already includes these observed player modifiers.
    if any(not isinstance(entity.get('status'), list) or
           any(not isinstance(s, dict) for s in entity['status']) for entity in (context.player, target)):
        return _unknown('玩家或目标状态列表未知')
    for status in context.player.get("status", []):
        if str(status.get("id", "")).upper().removesuffix("_POWER") not in DISPLAYED_PLAYER_MODIFIERS:
            reasons.append("玩家状态:" + str(status.get("id")))
    for status in target.get("status", []):
        ident = str(status.get("id", "")).upper().removesuffix("_POWER")
        if ident in {'VULNERABLE', 'SLIPPERY'} and (type(status.get('amount')) is not int or status['amount'] < 0):
            reasons.append('目标状态:' + ident + '数值未知')
        elif ident == "SLOW" and (type(status.get("amount")) is not int or not 0 <= status["amount"] <= 1000):
            reasons.append("目标状态:SLOW数值未知")
        elif ident == "SLOW" and hits > 1:
            reasons.append("缓慢与多段攻击的结算时序尚未确认")
        elif ident not in {"VULNERABLE", "SLOW", "SLIPPERY", "MINION", "STRENGTH", "WEAK", "ARTIFACT"}:
            reasons.append("目标状态:" + ident)
    for relic in context.player.get("relics", []):
        if relic.get("id") not in DIRECT_ATTACK_NEUTRAL_RELICS:
            reasons.append("遗物:" + str(relic.get("id")))
    if reasons:
        return _unknown("未支持的影响: " + ", ".join(reasons))
    applied = target_modifiers(target)
    numerator, denominator = int(match.group(1)), 1
    for modifier in applied:
        numerator *= modifier["numerator"]
        denominator *= modifier["denominator"]
    damage = numerator // denominator
    hp, block = int(target.get("hp") or 0), int(target.get("block") or 0)
    initial_hp, initial_block = hp, block
    slippery = sum(s.get('amount') or 0 for s in target.get('status', [])
                   if str(s.get('id', '')).upper().removesuffix('_POWER') == 'SLIPPERY')
    for _ in range(hits):
        if hp <= 0:
            break
        absorbed = min(block, damage)
        block -= absorbed
        hp_damage = damage - absorbed
        if hp_damage > 0 and slippery > 0:
            hp_damage = min(hp_damage, 1)
            slippery -= 1
        hp = max(0, hp - hp_damage)
    result = {"ok": True, "status": "calculated", "candidate_id": candidate_id,
            "target_hp_before": initial_hp, "target_block_before": initial_block,
            "damage_per_hit": damage, "hits": hits, "target_hp_after": hp,
            "target_block_after": block, "target_defeated": hp == 0,
            'hp_damage': initial_hp - hp,
            "scope": "仅此张牌直接伤害；牌面已含玩家修正，只再计算目标修正；不预测其他效果或回合结果"}
    if any(str(s.get('id', '')).upper().removesuffix('_POWER') == 'SLIPPERY' for s in target.get('status', [])):
        result['slippery_remaining'] = slippery
    if expected_hp_after is not None:
        result.update(expected_hp_after=expected_hp_after, matches=hp == expected_hp_after)
    result['player_outcome'] = _player_outcome(context, card, target, hp == 0)
    return result
