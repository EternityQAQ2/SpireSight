"""Run card observations and accepted inventory changes, separate from wiki facts.

Freeze each battle's initial union of hand/draw/discard/exhaust: played powers
disappear from those piles, so replacing the inventory every poll loses cards.
"""
from __future__ import annotations

from copy import deepcopy
import json

from sts2_advisor.contracts import Candidate


def inventory(cards: list[dict]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for card in cards:
        # Pile API omits is_upgraded; its displayed name carries the '+' suffix.
        upgraded = card.get("is_upgraded")
        if upgraded is None:
            upgraded = str(card.get("name", "")).endswith("+")
        key = (str(card.get("name") or card.get("id", "")).rstrip("+"), upgraded)
        if key not in rows:
            rows[key] = {k: deepcopy(card[k]) for k in
                         ("id", "name", "type", "cost", "star_cost", "description") if k in card}
            rows[key].update(is_upgraded=upgraded, count=0)
        rows[key]["count"] += int(card.get("count", 1))
    return list(rows.values())


class DeckTracker:
    """Session-local memory; only explicit master-deck data is authoritative."""

    def __init__(self) -> None:
        self._last: dict | None = None
        self._identity: tuple | None = None
        self._floor = -1
        self._pending: tuple[str, str, dict | None] | None = None
        self._selected: dict | None = None
        self._combat_key: tuple | None = None

    def accepted_action(self, raw: dict, candidate: Candidate) -> None:
        """Queue inventory changes only after POST success, apply after state changes."""
        card = None
        operation = "uncertain"
        index = candidate.params.get("index", candidate.params.get("card_index"))
        if candidate.action_type == "cancel_selection":
            self._selected = None
            return
        if candidate.action_type == "select_card_reward":
            card = next((c for c in raw.get("card_reward", {}).get("cards", []) if c.get("index") == index), None)
            operation = "add"
        elif candidate.action_type == "shop_purchase":
            shop = raw.get("shop") or raw.get("fake_merchant", {}).get("shop", {})
            item = next((c for c in shop.get("items", []) if c.get("index") == index), {})
            if item.get("card_name"):
                card = {"name": item["card_name"], "id": item.get("card_id"),
                        "type": item.get("card_type"), "star_cost": item.get("card_star_cost"),
                        "cost": item.get("card_cost"), "description": item.get("card_description", "")}
                operation = "add"
        elif candidate.action_type == "select_card" and raw.get("state_type") == "card_select":
            data = raw.get("card_select", {})
            self._selected = next((deepcopy(c) for c in data.get("cards", []) if c.get("index") == index), None)
            # Selection can be toggled or cancelled; commit only on confirmation.
            return
        elif candidate.action_type == "confirm_selection" and raw.get("state_type") == "card_select":
            data = raw.get("card_select", {})
            selected = data.get("selected_cards") or ([self._selected] if self._selected else [])
            card = selected[0] if len(selected) == 1 else None
            mode = str(data.get("mode") or data.get("screen_type", ""))
            operation = "upgrade" if "upgrade" in mode else "remove" if "remove" in mode else "uncertain"
            self._selected = None
        elif candidate.action_type not in {"choose_event_option", "choose_rest_option", "confirm_selection"}:
            return
        self._pending = (json.dumps(raw, sort_keys=True), operation, deepcopy(card))

    def _apply_pending(self, raw: dict) -> None:
        if self._pending is None or self._pending[0] == json.dumps(raw, sort_keys=True):
            return
        _, operation, card = self._pending
        self._pending = None
        if self._last is None:
            self._last = {"cards": [], "source": "accepted_actions", "complete": False,
                          "observed_floor": int((raw.get("run") or {}).get("floor") or 0)}
        rows = self._last["cards"]
        if card and operation in {"remove", "upgrade"}:
            match = next((r for r in rows if r.get("name") == card.get("name")), None)
            if match:
                match["count"] -= 1
                rows[:] = [r for r in rows if r["count"] > 0]
                if operation == "upgrade":
                    # Upgraded effect/cost is unknown until next observed hand/pile.
                    card = {"name": str(card.get("name", "")).rstrip("+") + "+", "is_upgraded": True}
                    rows[:] = inventory(rows + [card])
            else:
                operation = "uncertain"
        elif card and operation == "add":
            rows[:] = inventory(rows + [card])
        if operation == "uncertain" or card is None:
            note = "存在未能精确跟踪的牌组变化，等待下一场战斗校正"
            self._last["limitations"] = list(dict.fromkeys([*self._last.get("limitations", []), note]))
        self._last["complete"] = False

    def observe(self, raw: dict) -> dict:
        player, run = raw.get("player") or {}, raw.get("run") or {}
        identity = (player.get("character"), run.get("seed"), run.get("run_id"))
        floor = int(run.get("floor") or 0)
        if raw.get("state_type") in {"menu", "game_over"} or identity != self._identity or floor < self._floor:
            self._last = None
            self._pending = None
            self._selected = None
            self._combat_key = None
        self._identity, self._floor = identity, floor
        self._apply_pending(raw)
        cards = player.get("deck")
        if isinstance(cards, list):
            self._last = {"cards": inventory(cards), "source": "player.deck",
                          "complete": True, "observed_floor": floor}
            return deepcopy(self._last)
        if raw.get("state_type") in {"monster", "elite", "boss"}:
            observed = [c for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
                        for c in player.get(pile, [])]
            combat_key = (*identity, run.get("act"), floor)
            if observed and combat_key != self._combat_key:
                self._combat_key = combat_key
                self._last = {"cards": inventory(observed), "source": "combat_piles",
                              "complete": False, "observed_floor": floor,
                              "combat_only_cards": [],
                              "observed_round": (raw.get("battle") or {}).get("round"),
                              "limitations": ["战斗首次观测快照；可能含开局生成牌；升级状态按名称 + 推断"]}
            elif observed and self._last:
                # Enrich identity/type from later hands without losing played powers
                # or copying transient damage/cost back into the inventory.
                for row in self._last["cards"]:
                    match = next((c for c in observed if c.get("name") == row.get("name")), None)
                    if match:
                        for field in ("id", "type"):
                            if match.get(field) is not None:
                                row[field] = match[field]
                base = {r.get("name"): r["count"] for r in self._last["cards"]}
                self._last["combat_only_cards"] = [dict(r, count=r["count"] - base.get(r.get("name"), 0))
                                                  for r in inventory(observed)
                                                  if r["count"] > base.get(r.get("name"), 0)]
        snapshot = deepcopy(self._last) if self._last else {
            "cards": [], "source": "unavailable", "complete": False,
            "limitations": ["接口未提供本局持牌清单；卡牌知识库不能替代持牌清单"],
        }
        if self._last:
            snapshot["complete"] = False
            snapshot["stale"] = True
            if raw.get("state_type") in {"monster", "elite", "boss"}:
                snapshot["stale"] = False
            else:
                snapshot.setdefault("limitations", []).append("历史快照可能未包含拿牌、删牌、升级或变化")
        return snapshot
