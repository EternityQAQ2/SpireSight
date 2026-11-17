"""Prove resource errors independently of damage forecasting; no tactics or I/O."""
from __future__ import annotations

from sts2_advisor.contracts import DecisionContext
from sts2_advisor.game.combat.effects import energy_cost, known_effect


# These existing passive effects cannot grant energy or change card costs here.
# Damage-neutral alone is insufficient: e.g. Nunchaku/Horn must remain unknown.
RESOURCE_NEUTRAL_RELICS = {
    'BURNING_BLOOD', 'NEOWS_TALISMAN', 'SCROLL_BOXES',
    'STONE_HUMIDIFIER', 'BOOK_OF_FIVE_RINGS',
}
PLAYER_RESOURCE_NEUTRAL = {
    'STRENGTH', 'DEXTERITY', 'RAGE', 'WEAK', 'FRAIL', 'VULNERABLE',
    'SHRINK', 'FLEX_POTION', 'SETUP_STRIKE', 'ARTIFACT',
    'TANGLED',  # Current printed costs already include this turn's modifier.
}
ENEMY_RESOURCE_NEUTRAL = {
    'VULNERABLE', 'SLOW', 'SLIPPERY', 'MINION', 'STRENGTH', 'WEAK',
    'ARTIFACT', 'THORNS',
}


def _known_statuses(entity: dict, allowed: set[str]) -> bool:
    statuses = entity.get('status')
    return isinstance(statuses, list) and all(
        isinstance(s, dict) and str(s.get('id', '')).upper().removesuffix('_POWER') in allowed
        for s in statuses)


def known_resource_error(context: DecisionContext, ids: list[str]) -> str | None:
    """Return only provable errors. None means valid OR not sufficiently known.

    This does not certify a target, damage, or executable prefix. A draw, unknown
    card, discount or resource trigger ends this independent proof immediately.
    """
    player = context.player
    energy = player.get('energy')
    relics = player.get('relics')
    enemies = (context.situation.get('battle') or {}).get('enemies')
    if (context.decision_type != 'combat_next_action' or type(energy) is not int or energy < 0
            or player.get('orbs') or player.get('pets')
            or not _known_statuses(player, PLAYER_RESOURCE_NEUTRAL)
            or not isinstance(relics, list) or any(not isinstance(r, dict)
                or r.get('id') not in RESOURCE_NEUTRAL_RELICS for r in relics)
            or not isinstance(enemies, list) or not enemies
            or any(not isinstance(e, dict) or not _known_statuses(e, ENEMY_RESOURCE_NEUTRAL) for e in enemies)):
        return None
    candidates = {c.candidate_id: c for c in context.candidates if c.legal}
    cards = {c.get('index'): c for c in player.get('hand', [])}
    used = set()
    stars = player.get('stars')
    for pos, ident in enumerate(ids, 1):
        candidate = candidates.get(ident)
        if candidate is None or candidate.action_type != 'play_card':
            return None
        index = candidate.params.get('card_index')
        card = cards.get(index)
        if card is None:
            return None
        if index in used:
            return f'计划第{pos}步重复使用同一张手牌；请重新选择计划'
        cost = energy_cost(card, energy)
        star_cost = card.get('star_cost')
        if cost is None or str(card.get('cost', '')).upper() == 'X':
            return None
        if star_cost is not None and not str(star_cost).isdigit():
            return None
        star_cost = int(star_cost or 0)
        if star_cost and (type(stars) is not int or stars < 0):
            return None
        if cost > energy:
            return f'计划第{pos}步 {ident} 需{cost}能量，按已确认步骤仅剩{energy}；请重新选择计划'
        if star_cost and star_cost > stars:
            return f'计划第{pos}步需{star_cost}辉星，按已确认步骤仅剩{stars}；请重新选择计划'
        energy -= cost
        if star_cost:
            stars -= star_cost
        used.add(index)
        # Includes draw/selection, energy gain and next-attack discounts. No
        # static summation beyond them, even if the model supplied a long tail.
        if known_effect(card) in {None, 'shrug', 'pommel'}:
            return None
    return None
