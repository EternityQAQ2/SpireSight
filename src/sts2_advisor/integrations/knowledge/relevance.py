"""Decision-scoped query and fact ordering. Pure rules, no storage or model I/O."""
from __future__ import annotations

import re
from dataclasses import replace

from sts2_advisor.contracts import DecisionContext, Evidence


def entity_name(item: dict) -> str:
    name = next((item.get(k) for k in ('name', 'card_name', 'relic_name', 'potion_name', 'event_name')
                 if item.get(k)), '')
    name = str(name)
    return name + '+' if name and item.get('is_upgraded') and not name.endswith('+') else name


def entities(value):
    if isinstance(value, dict):
        if entity_name(value):
            yield value
        for key, child in value.items():
            if key != 'keywords':
                yield from entities(child)
    elif isinstance(value, list):
        for child in value:
            yield from entities(child)


def names(value) -> set[str]:
    result = {entity_name(item) for item in entities(value)}
    return result | {name[:-1] for name in result if name.endswith('+')}


def unique_text(values) -> str:
    return '、'.join(dict.fromkeys(str(v) for v in values if v is not None and str(v).strip()))


def decision_query(context: DecisionContext) -> str:
    """Short semantic query; observed inventories retain their provenance."""
    player = context.player
    parts = [f"{player.get('character', '')} {context.decision_type} 第{context.run.get('act', '?')}幕"]

    def add(label, text, limit):
        if text:
            parts.append(f'{label}：{text[:limit]}')

    hand = player.get('hand') or []
    if context.decision_type == 'combat_next_action':
        enemies = (context.situation.get('battle') or {}).get('enemies') or []
        add('当前敌人', unique_text(
            f"{entity_name(e)}（{unique_text([entity_name(s) for s in e.get('status', [])] + [i.get('title') or i.get('type') for i in e.get('intents', [])])}）"
            for e in enemies if e.get('hp', 1) != 0), 200)
        add('玩家状态', unique_text(entity_name(s) for s in player.get('status', [])), 80)
        add('当前手牌', unique_text(
            entity_name(c) + (f"({c['cost']}费)" if c.get('cost') is not None else '(费用未知)')
            for c in hand), 160)
        add('手牌效果', unique_text(c.get('description') for c in hand), 160)
        add('可用药水', unique_text(entity_name(p) for p in player.get('potions', [])), 60)
    else:
        add('当前选项', unique_text(c.label for c in context.candidates), 160)
        # Include card/event effects, but never serialize a large map graph or
        # generic interface descriptions such as "Purchase an item".
        choices = {k: v for k, v in context.situation.items() if k != 'battle'}
        descriptions = [e.get('description') or e.get('card_description')
                        for e in entities(choices)]
        if context.decision_type == 'event':
            descriptions += [o.get('description') for o in (choices.get('event') or {}).get('options', [])]
        add('选项效果', unique_text(descriptions), 160)
        snapshot = player.get('deck_snapshot') or {}
        deck = player.get('deck') if isinstance(player.get('deck'), list) else snapshot.get('cards')
        if deck:
            flags = [] if isinstance(player.get('deck'), list) else [
                flag for flag, present in [('不完整', not snapshot.get('complete')), ('过时', snapshot.get('stale'))]
                if present
            ]
            add('已观测牌组' + (f"（{'、'.join(flags)}）" if flags else ''),
                unique_text(entity_name(c) + (f"×{c['count']}" if c.get('count') else '') for c in deck), 180)
        add('遗物', unique_text(entity_name(r) for r in player.get('relics', [])), 80)
        add('玩家状态', unique_text(entity_name(s) for s in player.get('status', [])), 60)
        if player.get('hp') is not None:
            add('当前生命', f"{player['hp']}/{player.get('max_hp', '?')}", 30)
    return '；'.join(parts)[:800]


def rank_facts(context: DecisionContext, facts: list[Evidence]) -> list[Evidence]:
    """Order exact facts before truncation; scores do not select game actions."""
    player = context.player
    combat = context.decision_type == 'combat_next_action'
    enemies = (context.situation.get('battle') or {}).get('enemies') or []
    active_states = [*(player.get('status') or []), *(s for e in enemies for s in e.get('status', []))]
    active = names(active_states)
    active_ids = {str(s.get('id', '')).lower().removesuffix('_power') for s in active_states}
    enemy_names = names(enemies) - active
    hand = names(player.get('hand', []))
    choices = names({k: v for k, v in context.situation.items() if k != 'battle'})
    potions = names(player.get('potions', []))

    def priority(evidence):
        table = evidence.metadata.get('table')
        name = evidence.metadata.get('name')
        pages = set(evidence.metadata.get('pages') or [])
        if not combat and (table in {'cards', 'events'} or name in choices):
            return 0, '当前候选或事件'
        if table in {'powers', 'afflictions'}:
            if name in active or evidence.metadata.get('entity_id') in active_ids:
                return 0 if combat else 1, '当前状态'
            return (2, '可用药水关联效果') if name in potions else (4, '其他关联状态')
        if table == 'monsters':
            return 1, '当前敌人机制'
        if table == 'interactions':
            if pages & (active | enemy_names):
                return 1, '当前状态或敌人的交互'
            if pages & (hand | choices | potions):
                return 2 if combat else 1, '当前手牌、候选或药水的交互'
            return 5, '其他关联说明'
        if table == 'potions':
            return 2, '当前药水'
        if table == 'relics':
            description = evidence.metadata.get('description') or ''
            pickup_only = description.startswith('拾起时') and not re.search(
                r'每.*回合|每场战斗|当|战斗开始|战斗结束', description)
            return (6, '拾起时效果') if pickup_only else (3, '持续遗物效果')
        return 4, '其他可见实体'

    ranked = []
    for evidence in facts:
        score, reason = priority(evidence)
        ranked.append(replace(evidence, metadata={**evidence.metadata, 'priority': score, 'relevance': reason}))
    return sorted(ranked, key=lambda e: (e.metadata['priority'], e.evidence_id))
