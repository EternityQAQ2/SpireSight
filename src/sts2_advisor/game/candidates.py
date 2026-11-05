"""Legal actions exposed by the current game screen."""
from __future__ import annotations

from typing import Any
from sts2_advisor.contracts import Candidate


def build_candidates(raw: dict[str, Any]) -> list[Candidate]:
    state_type = str(raw.get("state_type"))
    if state_type in {"monster", "elite", "boss"}:
        return combat_candidates(raw)
    if state_type == "hand_select":
        data = raw.get("hand_select", {})
        # Once the game reports a selected card and enables confirmation, the
        # selection requirement has been satisfied. Collapse the candidate set
        # to the confirm action so the model cannot toggle between cards forever.
        # Exact multi-card prompts keep can_confirm false until enough cards are
        # selected; optional "up to N" prompts may confirm early, which is legal.
        # upgrade_select is an exact-one prompt whose selected_cards stays empty
        # (the selection only shows as a card missing from the cards list), so
        # can_confirm alone already proves the choice is complete there.
        if data.get("can_confirm") and (
            data.get("selected_cards") or data.get("mode") == "upgrade_select"
        ):
            return [Candidate("hand_select:confirm", "combat_confirm_selection", "确认选择")]
        result = [
            Candidate(f"hand_select:{card['index']}", "combat_select_card", card.get("name", "选择卡牌"), {"card_index": card["index"]})
            for card in data.get("cards", [])
            if "index" in card
        ]
        if data.get("can_confirm"):
            result.append(Candidate("hand_select:confirm", "combat_confirm_selection", "确认选择"))
        return result
    if state_type == "rewards":
        data = raw.get("rewards", {})
        # Combat rewards are all beneficial and should be collected without
        # spending a model call. Expose exactly one action at a time so the
        # generic deterministic path executes it, using the highest index to
        # remain safe when claiming a reward shifts the remaining indices.
        indexed_items = [item for item in data.get("items", []) if "index" in item]
        player = raw.get("player", {})
        potions = [potion for potion in player.get("potions", []) if "slot" in potion]
        max_slots = player.get("max_potion_slots")
        potion_slots_full = isinstance(max_slots, int) and max_slots > 0 and len(potions) >= max_slots
        claimable_items = [
            item for item in indexed_items
            if not (item.get("type") == "potion" and potion_slots_full)
        ]
        if claimable_items:
            item = max(claimable_items, key=lambda value: int(value["index"]))
            return [Candidate(
                f"reward:{item['index']}",
                "claim_reward",
                item.get("description", item.get("type", "奖励")),
                {"index": item["index"]},
            )]
        result = []
        if data.get("can_proceed"):
            result.append(Candidate("reward:proceed", "proceed", "离开奖励界面"))
        return result
    if state_type == "card_reward":
        data = raw.get("card_reward", {})
        result = [
            Candidate(f"card_reward:{card['index']}", "select_card_reward", card.get("name", "选牌"), {"card_index": card["index"]})
            for card in data.get("cards", [])
            if "index" in card
        ]
        return result
    if state_type == "map":
        return [
            Candidate(
                f"map:{node['index']}",
                "choose_map_node",
                f"{node.get('type', '节点')} ({node.get('col', '?')},{node.get('row', '?')})",
                {"index": node["index"]},
            )
            for node in raw.get("map", {}).get("next_options", [])
            if "index" in node
        ]
    if state_type == "event":
        event = raw.get("event", {})
        if event.get("in_dialogue"):
            return [Candidate("event:advance", "advance_dialogue", "继续对话")]
        return [
            Candidate(
                f"event:{option['index']}",
                "choose_event_option",
                option.get("title") or option.get("description") or "事件选项",
                {"index": option["index"]},
            )
            for option in event.get("options", [])
            if "index" in option and not option.get("is_locked", False)
        ]
    if state_type == "rest_site":
        data = raw.get("rest_site", {})
        result = [
            Candidate(f"rest:{option['index']}", "choose_rest_option", option.get("name", "休息点选项"), {"index": option["index"]})
            for option in data.get("options", [])
            if "index" in option and option.get("is_enabled", True)
        ]
        if data.get("can_proceed"):
            result.append(Candidate("rest:proceed", "proceed", "离开休息点"))
        return result
    if state_type in {"shop", "fake_merchant"}:
        shop = raw.get("shop", {}) if state_type == "shop" else raw.get("fake_merchant", {}).get("shop", {})
        result = []
        for item in shop.get("items", []):
            if "index" not in item or not item.get("is_stocked", True) or not item.get("can_afford", False):
                continue
            name = item.get("card_name") or item.get("relic_name") or item.get("potion_name") or item.get("category", "商品")
            result.append(Candidate(f"shop:{item['index']}", "shop_purchase", str(name), {"index": item["index"]}))
        if shop.get("is_open") or shop.get("can_proceed"):
            label = "关闭商店界面" if shop.get("is_open") else "离开商店"
            result.append(Candidate("shop:proceed", "proceed", label))
        return result
    if state_type == "treasure":
        data = raw.get("treasure", {})
        result = [
            Candidate(f"treasure:{relic['index']}", "claim_treasure_relic", relic.get("name", "遗物"), {"index": relic["index"]})
            for relic in data.get("relics", [])
            if "index" in relic
        ]
        if data.get("can_proceed"):
            result.append(Candidate("treasure:proceed", "proceed", "离开宝箱房"))
        return result
    if state_type == "card_select":
        data = raw.get("card_select", {})
        if data.get("preview_showing") and data.get("can_confirm"):
            return [Candidate("card_select:confirm", "confirm_selection", "Confirm selection")]
        selected_indices = {
            card["index"]
            for card in data.get("selected_cards", [])
            if isinstance(card, dict) and "index" in card
        }
        result = [
            Candidate(f"card_select:{card['index']}", "select_card", card.get("name", "卡牌"), {"index": card["index"]})
            for card in data.get("cards", [])
            if "index" in card and card["index"] not in selected_indices
        ]
        if data.get("can_confirm"):
            result.append(Candidate("card_select:confirm", "confirm_selection", "确认选择"))
        if data.get("can_cancel") or data.get("can_skip"):
            result.append(Candidate("card_select:cancel", "cancel_selection", "取消或跳过"))
        return result
    if state_type == "bundle_select":
        data = raw.get("bundle_select", {})
        result = [Candidate(f"bundle:{item['index']}", "select_bundle", item.get("name", "卡牌包"), {"index": item["index"]}) for item in data.get("bundles", []) if "index" in item]
        if data.get("can_confirm"):
            result.append(Candidate("bundle:confirm", "confirm_bundle_selection", "确认卡牌包"))
        return result
    if state_type == "relic_select":
        data = raw.get("relic_select", {})
        result = [Candidate(f"relic:{item['index']}", "select_relic", item.get("name", "遗物"), {"index": item["index"]}) for item in data.get("relics", []) if "index" in item]
        if data.get("can_skip"):
            result.append(Candidate("relic:skip", "skip_relic_selection", "跳过遗物"))
        return result
    if state_type == "crystal_sphere":
        data = raw.get("crystal_sphere", {})
        result = [Candidate(f"sphere:{cell['x']}:{cell['y']}", "crystal_sphere_click_cell", f"点击 ({cell['x']},{cell['y']})", {"x": cell["x"], "y": cell["y"]}) for cell in data.get("clickable_cells", [])]
        if data.get("can_proceed"):
            result.append(Candidate("sphere:proceed", "crystal_sphere_proceed", "结束占卜"))
        return result
    return []


def combat_candidates(raw: dict[str, Any]) -> list[Candidate]:
    battle = raw.get("battle", {})
    if battle.get("turn") != "player" or not battle.get("is_play_phase", False):
        return []
    enemies = [enemy for enemy in battle.get("enemies", []) if enemy.get("hp", 0) > 0]
    result: list[Candidate] = []
    for card in raw.get("player", {}).get("hand", []):
        if not card.get("can_play", False) or "index" not in card:
            continue
        target_type = card.get("target_type")
        if target_type in {"AnyEnemy", "Enemy"}:
            for enemy in enemies:
                if enemy.get("entity_id"):
                    result.append(
                        Candidate(
                            f"card:{card['index']}:{enemy['entity_id']}",
                            "play_card",
                            f"打出 {card.get('name', card.get('id'))} -> {enemy.get('name', enemy['entity_id'])}",
                            {"card_index": card["index"], "target": enemy["entity_id"]},
                        )
                    )
        else:
            result.append(Candidate(f"card:{card['index']}", "play_card", f"打出 {card.get('name', card.get('id'))}", {"card_index": card["index"]}))
    for potion in raw.get("player", {}).get("potions", []):
        if "slot" not in potion or not potion.get("can_use_in_combat", True):
            continue
        if potion.get("target_type") in {"AnyEnemy", "Enemy"}:
            for enemy in enemies:
                result.append(Candidate(f"potion:{potion['slot']}:{enemy['entity_id']}", "use_potion", f"使用 {potion.get('name', '药水')} -> {enemy.get('name')}", {"slot": potion["slot"], "target": enemy["entity_id"]}))
        else:
            result.append(Candidate(f"potion:{potion['slot']}", "use_potion", f"使用 {potion.get('name', '药水')}", {"slot": potion["slot"]}))
    result.append(Candidate("combat:end_turn", "end_turn", "结束回合"))
    return result
