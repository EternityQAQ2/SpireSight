"""Read displayed combat facts. Unknown effects are not simulated here."""
from __future__ import annotations

import re
from typing import Any
from sts2_advisor.contracts import Candidate, DecisionContext


ATTACK_DAMAGE_RE = re.compile(r"造成(\d+)点伤害(?:\s*(\d+)\s*次)?")


BLOCK_RE = re.compile(r"获得(\d+)点格挡")


ALL_ENEMIES_MARK = "对所有敌人"


def enemy_incoming(enemy: dict) -> tuple[int, int]:
    """返回 (本回合总伤害, 伤害段数)；非攻击意图不计入。"""
    total = 0
    hits = 0
    for intent in enemy.get("intents") or []:
        if not isinstance(intent, dict) or intent.get("type") != "Attack":
            continue
        match = ATTACK_DAMAGE_RE.search(str(intent.get("description") or ""))
        if match:
            damage = int(match.group(1))
            count = int(match.group(2) or 1)
            total += damage * count
            hits += count
            continue
        numbers = re.findall(r"\d+", str(intent.get("label") or ""))
        if numbers:
            total += int(numbers[0])
            hits += 1
    return total, hits


def incoming_damage(situation: dict) -> int:
    """兼容旧调用：只返回本回合敌方总伤害。"""
    total = 0
    for enemy in (situation.get("battle") or {}).get("enemies") or []:
        if isinstance(enemy, dict):
            total += enemy_incoming(enemy)[0]
    return total


def enemy_thorns(enemy: dict) -> int:
    for status in enemy.get("status") or []:
        if not isinstance(status, dict):
            continue
        status_id = str(status.get("id") or "").upper()
        name = str(status.get("name") or "")
        if status_id.startswith("THORNS") or "荆棘" in name:
            try:
                return int(status.get("amount") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def observed_thorns(enemy: dict) -> int | None:
    """Prompt facts retain unknown amounts; legacy simulation stays unchanged."""
    statuses = enemy.get("status")
    if not isinstance(statuses, list):
        return None
    total = 0
    for status in statuses:
        if not isinstance(status, dict):
            return None
        if (str(status.get("id") or "").upper().startswith("THORNS") or
                "荆棘" in str(status.get("name") or "")):
            amount = status.get("amount")
            if type(amount) is not int or amount < 0:
                return None
            total += amount
    return total


def card_block(card: dict) -> int:
    match = BLOCK_RE.search(str(card.get("description") or ""))
    return int(match.group(1)) if match else 0


def card_hits(card: dict, enemy_count: int) -> int:
    """攻击牌命中段数；DPS 类多段与 AOE 分别按段数/敌人数计算。"""
    if card.get("type") != "Attack":
        return 0
    text = str(card.get("description") or "")
    if ALL_ENEMIES_MARK in text:
        return max(1, enemy_count)
    match = ATTACK_DAMAGE_RE.search(text)
    if not match:
        return 0
    return int(match.group(2) or 1)


def card_damage_per_hit(card: dict) -> int:
    if card.get("type") != "Attack":
        return 0
    match = ATTACK_DAMAGE_RE.search(str(card.get("description") or ""))
    return int(match.group(1)) if match else 0


def target_thorns(candidate: Candidate, enemies: list[dict]) -> int:
    """定向攻击取目标荆棘；随机/AOE 取场上最高值作为最坏情况。"""
    target = candidate.params.get("target")
    if target:
        for enemy in enemies:
            if enemy.get("entity_id") == target:
                return enemy_thorns(enemy)
    return max((enemy_thorns(enemy) for enemy in enemies), default=0)


def card_hp_loss(card: dict) -> int | None:
    """Only an immediate, numeric life payment is known; never guess a trigger."""
    text = re.sub(r"\s+", "", str(card.get("description") or ""))
    match = re.match(r"失去(\d+)点生命[。.]", text)
    if match:
        return int(match.group(1))
    return None if not text or re.search(r"失去[^。]*生命", text) else 0


def attack_retaliation(candidate: Candidate, card: dict, enemies: list[dict]) -> int | None:
    """Observed thorns before block, assuming all described hits land.

    Random targets, missing hit counts or unknown thorns remain unknown. This
    is not net life loss and does not simulate early target death or other powers.
    """
    if card.get("type") != "Attack":
        return 0
    text = str(card.get("description") or "").replace("两次", "2次")
    match = ATTACK_DAMAGE_RE.search(text)
    target = candidate.params.get("target")
    random_target = "随机" in text
    if not match or re.search(r"[Xx]\s*次", text) or random_target and len(enemies) != 1:
        return None
    targets = ([e for e in enemies if e.get("entity_id") == target] if target else
               enemies if "所有敌人" in text or random_target else [])
    if not targets:
        return None
    total = 0
    for enemy in targets:
        amount = observed_thorns(enemy)
        if amount is None:
            return None
        total += amount
    return total * int(match.group(2) or 1)


def attack_cost_scope(player: dict) -> dict:
    """Current displayed costs cannot certify costs after a temporary discount."""
    result = {"basis": "current_state"}
    for status in player.get("status") or []:
        if str(status.get("id") or "").upper().removesuffix("_POWER") == "FREE_ATTACK":
            amount = status.get("amount")
            if type(amount) is int and amount == 0:
                continue
            result.update(free_attacks_remaining=amount if type(amount) is int and amount > 0 else None,
                          later_attack_costs=None, reobserve_after_attack=True)
    return result


def combat_summary(context: DecisionContext) -> dict[str, Any] | None:
    """给提示词用的本回合威胁摘要；非战斗局面返回 None。"""
    battle = context.situation.get("battle")
    if not isinstance(battle, dict):
        return None
    enemies = [item for item in battle.get("enemies") or [] if isinstance(item, dict)]
    if not enemies:
        return None
    rows = []
    total = 0
    total_hits = 0
    attackers: list[str] = []
    unknown_intents: list[str] = []
    for enemy in enemies:
        damage, hits = enemy_incoming(enemy)
        total += damage
        total_hits += hits
        name = str(enemy.get("name") or enemy.get("entity_id") or "")
        if any(i.get("type") == "Attack" and not ATTACK_DAMAGE_RE.search(str(i.get("description", "")))
               for i in enemy.get("intents", []) if isinstance(i, dict)):
            unknown_intents.append(name)
        if damage:
            attackers.append(name)
        hp = int(enemy.get("hp") or 0)
        block = int(enemy.get("block") or 0)
        rows.append(
            {
                "entity_id": enemy.get("entity_id"),
                "name": name,
                "hp": hp,
                "block": block,
                "effective_hp": hp + block,
                "incoming": damage,
                "hits": hits,
                "thorns": observed_thorns(enemy),
            }
        )
    player_block = int(context.player.get("block") or 0)
    player_hp = int(context.player.get("hp") or 0)
    unblocked = max(0, total - player_block)
    return {
        "incoming_damage": total,
        "incoming_damage_known": not unknown_intents,
        "unknown_attack_intents": unknown_intents,
        "incoming_hits": total_hits,
        "block": player_block,
        "unblocked_damage": unblocked,
        "hp_after_unblocked": player_hp - unblocked,
        "lethal_if_unblocked": player_hp - unblocked <= 0,
        "attackers": attackers,
        "enemies": rows,
    }
