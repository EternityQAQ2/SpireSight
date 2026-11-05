"""Normalize game observations into decision state; no network or model dependencies."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any
from sts2_advisor.contracts import DecisionContext
from sts2_advisor.game.candidates import build_candidates
from sts2_advisor.game.deck import DeckTracker


class StateBuilder:
    """Session-scoped observation enrichment shared by advice and action prechecks."""

    def __init__(self) -> None:
        self.deck_tracker = DeckTracker()

    def __call__(self, raw: dict) -> DecisionContext | None:
        return build_decision_context(raw, deck_snapshot=self.deck_tracker.observe(raw))


SUPPORTED_STATES = {
    "monster",
    "elite",
    "boss",
    "hand_select",
    "rewards",
    "card_reward",
    "map",
    "event",
    "rest_site",
    "shop",
    "fake_merchant",
    "treasure",
    "card_select",
    "bundle_select",
    "relic_select",
    "crystal_sphere",
}


def build_decision_context(raw: dict[str, Any], *, deck_snapshot: dict | None = None) -> DecisionContext | None:
    state_type = str(raw.get("state_type", "unknown"))
    if state_type not in SUPPORTED_STATES:
        return None

    candidates = build_candidates(raw)
    if not candidates:
        return None

    compact = {
        "state_type": state_type,
        "run": raw.get("run", {}),
        "player": compact_player(raw.get("player", {})),
        "situation": compact_situation(raw),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    if deck_snapshot is not None:
        compact["player"]["deck_snapshot"] = deck_snapshot
    canonical = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    decision_id = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return DecisionContext(
        schema_version="1.0",
        decision_id=decision_id,
        decision_type=decision_type_for(state_type),
        state_type=state_type,
        run=dict(raw.get("run") or {}),
        player=compact["player"],
        situation=compact["situation"],
        candidates=candidates,
    )


def decision_type_for(state_type: str) -> str:
    if state_type in {"monster", "elite", "boss"}:
        return "combat_next_action"
    return state_type


def compact_player(player: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "character",
        "hp",
        "max_hp",
        "block",
        "energy",
        "max_energy",
        "stars",
        "orbs",
        "orb_slots",
        "pets",
        "deck",
        "gold",
        "status",
        "relics",
        "potions",
        "max_potion_slots",
    )
    result = {key: player[key] for key in keys if key in player}
    result["hand"] = [compact_card(card) for card in player.get("hand", [])]
    for pile_name in ("draw_pile", "discard_pile", "exhaust_pile"):
        result[pile_name] = compact_pile(player.get(pile_name, []))
    return result


def compact_card(card: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "index",
        "id",
        "name",
        "type",
        "cost",
        "star_cost",
        "description",
        "is_upgraded",
        "target_type",
        "can_play",
        "unplayable_reason",
    )
    return {key: card[key] for key in keys if key in card}


def compact_pile(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, bool]] = Counter()
    details: dict[tuple[str, bool], dict[str, Any]] = {}
    for card in cards:
        identity = card_identity(card)
        counts[identity] += 1
        details.setdefault(identity, {
            "id": card.get("id"),
            "name": card.get("name"),
            "cost": card.get("cost"),
            "description": card.get("description", ""),
        })
    return [
        {
            field: value for field, value in details[identity].items() if value is not None
        } | {"is_upgraded": identity[1], "count": count}
        for identity, count in sorted(counts.items())
    ]


def card_identity(card: dict[str, Any]) -> tuple[str, bool]:
    """按名称聚合：牌堆卡没有 id 字段，名称是手牌与牌堆共同的稳定标识。"""
    return (str(card.get("name") or card.get("id") or "?"), bool(card.get("is_upgraded")))


def compact_situation(raw: dict[str, Any]) -> dict[str, Any]:
    state_type = str(raw.get("state_type"))
    if state_type in {"monster", "elite", "boss", "hand_select"}:
        return {"battle": raw.get("battle", {}), "hand_select": raw.get("hand_select")}
    if state_type == "map":
        return {"map": compact_map(raw.get("map", {}))}
    data = raw.get("card_select")
    if state_type == "card_select" and isinstance(data, dict) and data.get("preview_showing") and data.get("can_confirm"):
        # Preview confirms are deterministic (confirm is the only candidate), and the
        # game re-rolls preview_cards on every poll; leaving them in would change
        # decision_id each time and stop the watch loop from seeing a stable state.
        return {"card_select": {k: v for k, v in data.items() if k != "preview_cards"}}
    key = "fake_merchant" if state_type == "fake_merchant" else state_type
    value = raw.get(key, {})
    return {key: value}


def compact_map(map_data: dict[str, Any]) -> dict[str, Any]:
    """压缩地图状态：全图节点改为 [col,row,type,children] 数组编码（去掉键名），
    已访问节点只保留数量。路线规划信息完整保留，体积约为原始的六成。"""
    if not isinstance(map_data, dict):
        return map_data
    slim: dict[str, Any] = {}
    for field in ("current_position", "next_options", "boss", "bosses"):
        if map_data.get(field) is not None:
            slim[field] = map_data[field]
    nodes = map_data.get("nodes")
    if isinstance(nodes, list):
        slim["nodes"] = [
            [
                node.get("col"),
                node.get("row"),
                node.get("type"),
                node.get("children", []),
            ]
            for node in nodes
            if isinstance(node, dict) and "col" in node and "row" in node
        ]
    visited = map_data.get("visited")
    if isinstance(visited, list):
        slim["visited_count"] = len(visited)
    return slim
