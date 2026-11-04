"""Build model-facing card metrics from current combat facts."""
from __future__ import annotations

from sts2_advisor.contracts import Candidate
from sts2_advisor.game.combat.facts import (
    ATTACK_DAMAGE_RE, BLOCK_RE, ALL_ENEMIES_MARK, card_block, card_hits,
    card_damage_per_hit, target_thorns, enemy_thorns,
)
from sts2_advisor.game.combat.effects import energy_cost, known_effect


def candidate_metrics(
    candidate: Candidate, card: dict | None, enemies: list[dict],
) -> dict[str, int]:
    """一张牌针对当前候选的实际数值：伤害、段数、自伤、格挡。

    只对 play_card 生效；无法从描述解析的攻击（如全身撞击）不猜伤害。
    """
    if card is None or candidate.action_type != "play_card":
        return {}
    metrics: dict[str, int] = {}
    block = card_block(card)
    if block:
        metrics["block"] = block
    text = str(card.get("description") or "")
    per_hit = card_damage_per_hit(card)
    if not per_hit:
        return metrics
    hits = card_hits(card, len(enemies))
    if ALL_ENEMIES_MARK in text:
        metrics["dmg"] = per_hit
        if hits > 1:
            metrics["hits"] = hits
        self_damage = sum(enemy_thorns(enemy) for enemy in enemies)
    else:
        metrics["dmg"] = per_hit * hits
        if hits > 1:
            metrics["hits"] = hits
        self_damage = target_thorns(candidate, enemies) * hits
    if self_damage:
        metrics["self_damage"] = self_damage
    return metrics


def dynamic_card_metrics(candidate: Candidate, card: dict, enemies: list[dict], energy: int) -> dict:
    """Explicit nulls distinguish unparsed values from a verified zero.

    dmg is displayed damage before target modifiers; never claim target-specific
    damage from card text alone. Full supported sequences use combat_engine.
    """

    metrics = candidate_metrics(candidate, card, enemies)
    attack = card.get("type") == "Attack"
    metrics["cost"] = energy_cost(card, energy)
    metrics["star_cost"] = card.get("star_cost")
    metrics["block"] = card_block(card) if BLOCK_RE.search(str(card.get("description", ""))) else None
    metrics["dmg"] = metrics.get("dmg") if attack else None
    match = ATTACK_DAMAGE_RE.search(str(card.get("description", "")))
    if attack and match and metrics["dmg"] is None:
        metrics["dmg"] = int(match[1]) * int(match[2] or 1)
    metrics["effect_supported"] = known_effect(card) is not None
    metrics["damage_basis"] = "当前牌面；目标易伤等加成由逐步试算处理"
    return metrics
