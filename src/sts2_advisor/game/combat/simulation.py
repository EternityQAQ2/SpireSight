"""Execute supported effects and calculate one-turn outcomes without I/O."""
from __future__ import annotations

from copy import deepcopy
import re
from sts2_advisor.contracts import Candidate, DecisionContext
from sts2_advisor.game.combat.facts import card_block, card_damage_per_hit, enemy_incoming, enemy_thorns
from sts2_advisor.game.combat.effects import energy_cost, known_effect
from sts2_advisor.game.combat.state import CombatState, EnemyState, TurnOutcome


def _status_id(status: dict) -> str:
    return str(status.get("id", "")).upper().removesuffix("_POWER")


def _amount(entity: dict, ident: str) -> int:
    return sum(int(s.get("amount") or 0) for s in entity.get("status", []) if _status_id(s) == ident)


def initial_state(context: DecisionContext) -> CombatState | None:
    """Only return a forecast state when all relevant active mechanics are covered."""
    battle = context.situation.get("battle") or {}
    enemies = deepcopy(battle.get("enemies") or [])
    if not enemies or any(not e.get("entity_id") for e in enemies):
        return None
    player = context.player
    if player.get("orbs") or player.get("pets"):
        return None
    if any(_status_id(s) not in {"STRENGTH", "DEXTERITY", "RAGE"} for s in player.get("status", [])):
        return None
    if any(_status_id(s) not in {"VULNERABLE", "ARTIFACT", "THORNS", "STRENGTH", "MINION"}
           for e in enemies for s in e.get("status", [])):
        return None
    # Turn-start/end, card-play and damage-trigger relics require explicit support.
    if any(r.get("id") not in {"BURNING_BLOOD", "BOOMING_CONCH", "KUSARIGAMA", "NEOWS_TALISMAN", "LOST_COFFER", "GREMLIN_HORN", "PEN_NIB"}
           for r in player.get("relics", [])):
        return None
    for e in enemies:
        for intent in e.get("intents", []):
            if intent.get("type") == "Attack" and not re.search(r"造成\d+点伤害", str(intent.get("description", ""))):
                return None
        e.setdefault("block", 0)
        e["vulnerable"] = _amount(e, "VULNERABLE")
        e["artifact"] = _amount(e, "ARTIFACT")
        e["incoming"] = enemy_incoming(e)[0]
    kusari = next((r for r in player.get("relics", []) if r.get("id") == "KUSARIGAMA"), None)
    # Random relic damage is deterministic only with one surviving enemy.
    if kusari and len(enemies) > 1:
        return None
    return CombatState(
        hp=int(player.get("hp") or 0), block=int(player.get("block") or 0),
        energy=int(player.get("energy") or 0), strength=_amount(player, "STRENGTH"),
        strength_delta=0, rage=_amount(player, "RAGE"),
        pen_nib=next((int(r.get("counter") or 0) for r in player.get("relics", []) if r.get("id") == "PEN_NIB"), None),
        gremlin_horn=any(r.get("id") == "GREMLIN_HORN" for r in player.get("relics", [])), kusari=int(kusari.get("counter") or 0) if kusari else None,
        enemies=[EnemyState(entity_id=e["entity_id"], hp=e["hp"], block=e["block"],
                            vulnerable=e["vulnerable"], artifact=e["artifact"],
                            incoming=e["incoming"], thorns=enemy_thorns(e), minion=bool(_amount(e, "MINION"))) for e in enemies],
    )


def _damage(enemy: EnemyState, damage: int) -> None:
    absorbed = min(enemy.block, damage)
    enemy.block -= absorbed
    enemy.hp = max(0, enemy.hp - damage + absorbed)


def play(state: CombatState, candidate: Candidate, card: dict) -> CombatState | None:
    effect = known_effect(card)
    cost = energy_cost(card, state.energy)
    index = card.get("index")
    if state.boundary or effect is None or cost is None or cost > state.energy or index in state.used:
        return None
    nxt = deepcopy(state)
    nxt.energy -= cost
    nxt.used.add(index)
    target = next((e for e in nxt.enemies if e.entity_id == candidate.params.get("target") and e.hp > 0), None)
    if effect in {"strike", "bash", "taunt", "iron_wave", "body_slam", "setup_strike", "twin_strike", "pommel"} and target is None:
        return None
    alive_before = sum(e.hp > 0 for e in nxt.enemies)
    if effect != "rage":
        nxt.block += card_block(card)
    if card.get("type") == "Attack":
        damage = nxt.block if effect == "body_slam" else card_damage_per_hit(card) + nxt.strength_delta
        if effect == "body_slam":
            damage += nxt.strength + nxt.strength_delta
        damage = max(0, damage)
        damage = damage * 3 // 2 if target.vulnerable else damage
        if nxt.pen_nib is not None:
            nxt.pen_nib += 1
            if nxt.pen_nib >= 10:
                damage *= 2
                nxt.pen_nib = 0
        nxt.block += nxt.rage
        hits = 2 if effect == "twin_strike" else 1
        actual_hits = 0
        for _ in range(hits):
            if target.hp <= 0:
                break
            _damage(target, damage)
            actual_hits += 1
        thorns = target.thorns * actual_hits
        absorbed = min(nxt.block, thorns)
        nxt.block -= absorbed
        nxt.hp -= thorns - absorbed
        if nxt.kusari is not None:
            nxt.kusari += 1
            if nxt.kusari >= 3:
                nxt.kusari = 0
                alive = [e for e in nxt.enemies if e.hp > 0]
                if alive:
                    _damage(alive[0], 6)
    if effect in {"bash", "taunt"} and target.hp > 0:
        amount = int(re.search(r"给予(\d+)层易伤", card["description"])[1])
        if target.artifact:
            target.artifact -= 1
        else:
            target.vulnerable += amount
    if effect in {"inflame", "setup_strike"}:
        nxt.strength_delta += int(re.search(r"获得(\d+)点力量", card["description"])[1])
    if effect == "rage":
        nxt.rage += int(re.search(r"获得(\d+)点格挡", card["description"])[1])
    if effect in {"shrug", "pommel"}:
        nxt.boundary = True
    leaders = [e for e in nxt.enemies if not e.minion]
    if leaders and all(e.hp <= 0 for e in leaders):
        for e in nxt.enemies:
            if e.minion:
                e.hp = 0
    if nxt.gremlin_horn and sum(e.hp > 0 for e in nxt.enemies) < alive_before:
        nxt.energy += alive_before - sum(e.hp > 0 for e in nxt.enemies)
        nxt.boundary = any(e.hp > 0 for e in nxt.enemies)
    return nxt


def outcome(state: CombatState, initial: CombatState) -> TurnOutcome:
    incoming = sum(e.incoming for e in state.enemies if e.hp > 0)
    lost = max(0, incoming - state.block)
    return TurnOutcome(
        damage=sum(e.hp for e in initial.enemies) - sum(e.hp for e in state.enemies),
        block=state.block, incoming=incoming,
        hp_loss=initial.hp - state.hp + lost, hp_after=state.hp - lost,
        energy_left=state.energy, kills=sum(e.hp <= 0 for e in state.enemies),
        enemies_remaining=sum(e.hp > 0 for e in state.enemies),
    )
