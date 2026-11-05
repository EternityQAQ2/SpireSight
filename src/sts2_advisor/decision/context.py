"""Assemble provider-independent model context with current values and inventory provenance."""
from __future__ import annotations

from sts2_advisor.contracts import DecisionContext
from sts2_advisor.game.deck import inventory
from sts2_advisor.game.combat.facts import combat_summary
from sts2_advisor.game.combat.search import rank_turns
from sts2_advisor.decision.metrics import dynamic_card_metrics


def model_context_payload(context: DecisionContext, *, codex_cache: CodexCache | None = None) -> dict:
    """Build a smaller prompt with stable fields first for provider prefix caching."""
    player = context.player
    relics = sorted(
        player.get("relics", []),
        key=lambda item: str(item.get("id") or item.get("name") or ""),
    )
    stable = {
        "character": player.get("character"),
        "max_hp": player.get("max_hp"),
        "max_energy": player.get("max_energy"),
        "relics": [
            compact_fields(item, ("id", "name", "description")) for item in relics
        ],
    }
    deck = deck_codex(player, context.run, cache=codex_cache)
    if deck:
        stable["deck"] = deck
    dynamic_player = compact_fields(
        player, ("hp", "block", "energy", "stars", "orbs", "orb_slots", "pets", "gold", "status", "potions", "max_potion_slots")
    )
    dynamic_player["relic_counters"] = [
        {"id": item.get("id"), "counter": item.get("counter")}
        for item in relics if item.get("counter") is not None
    ]
    codex = {row["k"]: row for row in deck}
    dynamic_player["hand"] = [compact_hand_card(card, codex.get(card_key(card))) for card in player.get("hand", [])]
    dynamic_player["deck_snapshot"] = player.get("deck_snapshot") or (
        {"cards": inventory(player["deck"]), "source": "player.deck", "complete": True}
        if isinstance(player.get("deck"), list) else
        {"cards": [], "source": "unavailable", "complete": False}
    )
    for pile in ("draw_pile", "discard_pile", "exhaust_pile"):
        dynamic_player[pile] = pile_counts(player.get(pile, []))
    candidates = [
        {"id": item.candidate_id, "label": item.label} for item in context.candidates
    ]
    payload = {
        "stable": stable,
        "run": context.run,
        "state_type": context.state_type,
        "decision_type": context.decision_type,
        "player": dynamic_player,
        "situation": context.situation,
        "candidates": candidates,
    }
    if context.decision_type == "combat_next_action":
        # 意图原文是「造成6点伤害3次」这类自然语言，模型容易只取第一段。
        # 这里把本回合威胁与每张候选的实际伤害/反伤算成结构化数字，
        # 只做算术，不替模型选目标或排序。
        summary = combat_summary(context)
        if summary:
            payload["combat"] = summary
            battle = context.situation.get("battle") or {}
            enemies = [
                item for item in battle.get("enemies") or [] if isinstance(item, dict)
            ]
            cards = {card.get("index"): card for card in player.get("hand", [])}
            for item, entry in zip(context.candidates, candidates):
                card = cards.get(item.params.get("card_index"))
                if card is not None and item.action_type == "play_card":
                    entry.update(dynamic_card_metrics(item, card, enemies, int(player.get("energy") or 0)))
        payload["turn_search"] = rank_turns(context)
    return payload


def card_key(card: dict) -> str:
    """名称优先的图鉴键：手牌卡与牌堆条目都带名称（牌堆 API 没有 id），
    卡牌在手牌与牌堆间移动时键保持不变，图鉴才可能被前缀缓存命中。"""
    name = str(card.get("name") or card.get("id") or "")
    return name + ("+" if card.get("is_upgraded") and not name.endswith("+") else "")


class CodexCache:
    """Explicit per-backend cache. Independent sessions never share mutable entries."""

    def __init__(self) -> None:
        self.key: tuple[str, str, str] | None = None
        self.entries: dict[str, dict] = {}

    def clear(self) -> None:
        self.key = None
        self.entries.clear()


def deck_codex(player: dict, run: dict, *, cache: CodexCache | None = None) -> list[dict]:
    sources = [
        *player.get("hand", []),
        *player.get("draw_pile", []),
        *player.get("discard_pile", []),
        *player.get("exhaust_pile", []),
        *(player.get("deck_snapshot") or {}).get("cards", []),
        *player.get("deck", []),
    ]
    if not sources:
        return []
    key = (
        str(player.get("character") or ""),
        str(run.get("act")),
        str(run.get("floor")),
    )
    cache = cache if cache is not None else CodexCache()
    if cache.key != key:
        cache.clear()
        cache.key = key
    entries = cache.entries
    for card in sources:
        key_name = card_key(card)
        if not key_name or key_name in entries:
            continue
        entry = {
            "k": key_name,
            "name": card.get("name"),
            "type": card.get("type"),
            "cost": card.get("cost"),
            "description": card.get("description", ""),
        }
        entries[key_name] = {field: value for field, value in entry.items() if value is not None}
    # 插入序输出：已有条目顺序不变，新卡追加在尾部，前缀缓存可完整复用。
    return list(entries.values())


def compact_hand_card(card: dict, reference: dict | None = None) -> dict:
    entry = {
        "k": card_key(card),
        "index": card.get("index"),
        "cost": card.get("cost"),
        "star_cost": card.get("star_cost"),
        "is_upgraded": card.get("is_upgraded"),
        "can": card.get("can_play", True),
    }
    if reference is None or card.get("description") != reference.get("description"):
        entry["description"] = card.get("description")
    if not card.get("can_play", True) and card.get("unplayable_reason"):
        entry["why"] = card.get("unplayable_reason")
    return entry


def pile_counts(cards: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in cards:
        key = card_key(card)
        if key:
            counts[key] = counts.get(key, 0) + int(card.get("count", 1))
    return counts


def compact_fields(value: dict, fields: tuple[str, ...]) -> dict:
    return {field: value[field] for field in fields if field in value}
