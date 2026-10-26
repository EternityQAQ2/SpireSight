from __future__ import annotations

import unittest

from sts2_advisor.combat import (
    candidate_metrics,
    combat_summary,
    enemy_incoming,
    incoming_damage,
)
from sts2_advisor.domain import Candidate, DecisionContext
from sts2_advisor.model import model_context_payload


def enemy(
    name: str,
    hp: int,
    *intents: dict,
    thorns: int = 0,
    block: int = 0,
    entity_id: str | None = None,
) -> dict:
    status = (
        [{"id": "THORNS_POWER", "name": "荆棘", "amount": thorns, "type": "Buff"}]
        if thorns
        else []
    )
    return {
        "entity_id": entity_id or f"{name}_0",
        "name": name,
        "hp": hp,
        "max_hp": hp,
        "block": block,
        "status": status,
        "intents": list(intents),
    }


def attack(description: str, label: str = "") -> dict:
    return {"type": "Attack", "label": label, "title": "攻势", "description": description}


def make_context(
    *,
    hp: int = 40,
    block: int = 0,
    energy: int = 3,
    enemies: list[dict] | None = None,
    hand: list[dict] | None = None,
    decision_type: str = "combat_next_action",
) -> DecisionContext:
    hand = hand or []
    for index, card in enumerate(hand):
        card.setdefault("index", index)
        card.setdefault("can_play", True)
    candidates = [
        Candidate(
            f"card:{card['index']}",
            "play_card",
            f"打出 {card['name']}",
            {"card_index": card["index"]},
        )
        for card in hand
    ]
    candidates.append(Candidate("combat:end_turn", "end_turn", "结束回合"))
    return DecisionContext(
        schema_version="1",
        decision_id="sha256:test",
        decision_type=decision_type,
        state_type="monster",
        run={"act": 1, "floor": 24, "ascension": 0},
        player={
            "character": "铁甲战士",
            "hp": hp,
            "max_hp": 85,
            "block": block,
            "energy": energy,
            "max_energy": 3,
            "relics": [],
            "hand": hand,
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
        },
        situation={"battle": {"round": 2, "turn": "player", "is_play_phase": True,
                              "enemies": enemies or []}},
        candidates=candidates,
    )


class EnemyIncomingTest(unittest.TestCase):
    def test_multi_hit_is_multiplied_by_hit_count(self) -> None:
        self.assertEqual(
            enemy_incoming(enemy("多尼斯异鸟", 31, attack("这个敌人将要攻击造成6点伤害3次。"))),
            (18, 3),
        )

    def test_single_hit_uses_description(self) -> None:
        self.assertEqual(
            enemy_incoming(enemy("棘刺蟾蜍", 89, attack("这个敌人将要攻击造成23点伤害。", "23"))),
            (23, 1),
        )

    def test_non_attack_intents_are_ignored(self) -> None:
        self.assertEqual(
            enemy_incoming(enemy("盛碗虫（卵）", 22, attack("造成7点伤害。"),
                                 {"type": "Defend", "label": "", "description": "获得格挡。"})),
            (7, 1),
        )

    def test_incoming_damage_sums_all_attackers(self) -> None:
        situation = {
            "battle": {
                "enemies": [
                    enemy("石", 48, attack("造成15点伤害。")),
                    enemy("卵", 22, attack("造成7点伤害。")),
                ]
            }
        }
        self.assertEqual(incoming_damage(situation), 22)


class CombatSummaryTest(unittest.TestCase):
    def test_reports_unblocked_damage_and_lethality(self) -> None:
        context = make_context(
            hp=17,
            block=0,
            enemies=[enemy("棘刺蟾蜍", 89, attack("造成23点伤害。"), thorns=5)],
        )
        summary = combat_summary(context)
        assert summary is not None
        self.assertEqual(summary["incoming_damage"], 23)
        self.assertEqual(summary["incoming_hits"], 1)
        self.assertEqual(summary["unblocked_damage"], 23)
        self.assertEqual(summary["hp_after_unblocked"], -6)
        self.assertTrue(summary["lethal_if_unblocked"])
        self.assertEqual(summary["attackers"], ["棘刺蟾蜍"])
        self.assertEqual(summary["enemies"][0]["thorns"], 5)

    def test_existing_block_reduces_unblocked_damage(self) -> None:
        context = make_context(
            hp=14,
            block=9,
            enemies=[enemy("多尼斯异鸟", 31, attack("造成6点伤害3次。"))],
        )
        summary = combat_summary(context)
        assert summary is not None
        self.assertEqual(summary["incoming_damage"], 18)
        self.assertEqual(summary["hp_after_unblocked"], 5)
        self.assertFalse(summary["lethal_if_unblocked"])


class CandidateMetricsTest(unittest.TestCase):
    def test_multi_hit_card_counts_thorns_per_hit(self) -> None:
        card = {"name": "飞剑回旋镖+", "type": "Attack", "cost": "1",
                "description": "随机对敌人造成7点伤害4次。", "index": 0}
        metrics = candidate_metrics(
            Candidate("card:0", "play_card", "打出 飞剑回旋镖+", {"card_index": 0}),
            card,
            [enemy("棘刺蟾蜍", 89, thorns=5)],
        )
        self.assertEqual(metrics["dmg"], 28)
        self.assertEqual(metrics["hits"], 4)
        self.assertEqual(metrics["self_damage"], 20)

    def test_aoe_hits_each_enemy_once(self) -> None:
        card = {"name": "闪电霹雳", "type": "Attack", "cost": "1",
                "description": "对所有敌人造成8点伤害，给予1层易伤。", "index": 0}
        metrics = candidate_metrics(
            Candidate("card:0", "play_card", "打出 闪电霹雳", {"card_index": 0}),
            card,
            [enemy("棘刺蟾蜍", 89, thorns=5), enemy("飞蝇菌子", 49)],
        )
        self.assertEqual(metrics["dmg"], 8)
        self.assertEqual(metrics["hits"], 2)
        self.assertEqual(metrics["self_damage"], 5)

    def test_targeted_attack_uses_target_thorns(self) -> None:
        card = {"name": "打击", "type": "Attack", "cost": "1",
                "description": "造成6点伤害。", "index": 0}
        metrics = candidate_metrics(
            Candidate("card:0:FLYCONID_0", "play_card", "打出 打击",
                      {"card_index": 0, "target": "FLYCONID_0"}),
            card,
            [enemy("棘刺蟾蜍", 89, thorns=5, entity_id="SPINY_TOAD_0"),
             enemy("飞蝇菌子", 49, entity_id="FLYCONID_0")],
        )
        self.assertEqual(metrics["dmg"], 6)
        self.assertNotIn("self_damage", metrics)

    def test_skill_block_is_parsed(self) -> None:
        card = {"name": "坚毅+", "type": "Skill", "cost": "1",
                "description": "获得9点格挡。消耗1张牌。", "index": 0}
        metrics = candidate_metrics(
            Candidate("card:0", "play_card", "打出 坚毅+", {"card_index": 0}),
            card,
            [enemy("棘刺蟾蜍", 89, thorns=5)],
        )
        self.assertEqual(metrics, {"block": 9})


class PromptPayloadTest(unittest.TestCase):
    def test_combat_payload_carries_threat_and_card_metrics(self) -> None:
        hand = [{"name": "飞剑回旋镖+", "type": "Attack", "cost": "1",
                 "description": "随机对敌人造成7点伤害4次。"}]
        context = make_context(
            hp=17,
            enemies=[enemy("棘刺蟾蜍", 89, attack("造成23点伤害。"), thorns=5)],
            hand=hand,
        )
        payload = model_context_payload(context)
        self.assertEqual(payload["combat"]["incoming_damage"], 23)
        self.assertTrue(payload["combat"]["lethal_if_unblocked"])
        card = next(item for item in payload["candidates"] if item["id"] == "card:0")
        self.assertEqual(card["self_damage"], 20)

    def test_non_combat_payload_is_unchanged(self) -> None:
        context = make_context(decision_type="event")
        payload = model_context_payload(context)
        self.assertNotIn("combat", payload)


if __name__ == "__main__":
    unittest.main()
