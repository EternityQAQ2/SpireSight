from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from sts2_advisor.contracts import DecisionContext
from sts2_advisor.decision.prompts import build_prompt
from sts2_advisor.game.state import build_decision_context
from sts2_advisor.integrations.knowledge.sqlite import SqliteFactsRetriever
from sts2_advisor.integrations.knowledge.factory import create_retriever
from sts2_advisor.integrations.knowledge.common import metadata_matches
from tools.build_wiki_db import SCHEMA_SQL


def make_context(decision_type: str, character: str = "铁甲战士") -> DecisionContext:
    return DecisionContext(
        "1.0", "test", decision_type, decision_type, {},
        {"character": character}, {}, [],
    )


class MetadataFilterTests(unittest.TestCase):
    def test_chroma_multi_type_metadata_only_matches_listed_types(self) -> None:
        # ingest 多类型时写入 decision_type="all"，过滤不得把它当通配符
        metadata = {"decision_type": "all", "decision_types": "card_reward,shop",
                    "character": "铁甲战士"}
        self.assertTrue(metadata_matches(metadata, make_context("card_reward")))
        self.assertTrue(metadata_matches(metadata, make_context("shop")))
        self.assertFalse(metadata_matches(metadata, make_context("map")))
        self.assertFalse(metadata_matches(metadata, make_context("combat_next_action")))

    def test_chroma_character_filter(self) -> None:
        metadata = {"decision_types": "card_reward", "character": "故障机器人"}
        self.assertTrue(metadata_matches(metadata, make_context("card_reward", "故障机器人")))
        self.assertFalse(metadata_matches(metadata, make_context("card_reward", "铁甲战士")))
        self.assertFalse(
            metadata_matches(metadata, make_context("map", "故障机器人")))

    def test_jsonl_list_metadata_still_matches(self) -> None:
        metadata = {"decision_types": ["card_reward", "shop"], "character": "all"}
        self.assertTrue(metadata_matches(metadata, make_context("shop")))
        self.assertFalse(metadata_matches(metadata, make_context("event")))

    def test_empty_or_all_metadata_matches_everything(self) -> None:
        self.assertTrue(metadata_matches({}, make_context("map")))
        self.assertTrue(metadata_matches({"decision_types": "all"}, make_context("event")))


class SqliteRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.db = self.root / "sts2_facts.sqlite3"
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executescript(SCHEMA_SQL)
            connection.executemany(
                "INSERT INTO cards(id,name,description) VALUES (?,?,?)",
                [
                    ("strike_ironclad", "打击", "造成6点伤害。"),
                    ("strike_silent", "打击", "另一角色的打击。"),
                    ("strike_ironclad_upgrade", "打击+", "造成9点伤害。"),
                    ("unrelated", "无关卡牌", "不应检索到。"),
                    ("pile_only", "牌堆专用", "不应因牌堆而检索。"),
                ],
            )
            connection.execute(
                "INSERT INTO relics(id,name,description) VALUES (?,?,?)",
                ("burning_blood", "燃烧之血", "战斗结束回复6点生命。"),
            )
            connection.execute(
                "INSERT INTO interactions(id,description,description_wiki) VALUES (1,?,?)",
                ("燃烧之血在战斗结束时结算。", "raw"),
            )
            connection.execute("INSERT INTO interaction_pages VALUES (1,'燃烧之血')")
        self.raw = json.loads((Path(__file__).parent / "fixtures/combat_state.json").read_text(encoding="utf-8"))
        self.context = build_decision_context(self.raw)
        self.knowledge = self.root / "knowledge.jsonl"
        self.knowledge.write_text(json.dumps({
            "evidence_id": "survival-guide", "text": "优先避免生命损失。", "source_title": "战斗攻略",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        # 本文件只验证 sqlite/json 组合逻辑，空种子文件隔离 seed 注入（另见 test_seed_injection）
        self.seeds = self.root / "seeds.jsonl"
        self.seeds.write_text("", encoding="utf-8")

    def test_retrieves_matching_ids_and_interactions_into_llm_prompt(self):
        evidence = SqliteFactsRetriever(self.db).retrieve(self.context, 20)
        by_id = {item.evidence_id: item for item in evidence}
        self.assertIn("sqlite:relics:burning_blood", by_id)
        self.assertIn("sqlite:interactions:1", by_id)
        # 战斗中手牌卡牌事实由牌组图鉴提供，检索不再重复查询。
        self.assertNotIn("sqlite:cards:strike_ironclad", by_id)
        self.assertNotIn("sqlite:cards:strike_silent", by_id)
        self.assertNotIn("sqlite:cards:unrelated", by_id)
        self.assertNotIn("sqlite:cards:strike_ironclad_upgrade", by_id)
        prompt = build_prompt(self.context, evidence)
        self.assertIn("燃烧之血在战斗结束时结算。", prompt)
        self.assertIn("sqlite:interactions:1", prompt)
        # 手牌效果仍进入提示词——由图鉴提供。
        self.assertIn("造成6点伤害。", prompt)

    def test_combat_skips_hand_card_facts_but_keeps_relic_facts(self):
        evidence = SqliteFactsRetriever(self.db).retrieve(self.context, 20)
        ids = {item.evidence_id for item in evidence}
        self.assertIn("sqlite:relics:burning_blood", ids)
        self.assertFalse(any(evidence_id.startswith("sqlite:cards:") for evidence_id in ids))

    def test_upgraded_card_uses_upgraded_facts(self):
        raw = {
            "state_type": "card_reward",
            "run": {"act": 1, "floor": 4},
            "player": {},
            "card_reward": {
                "cards": [{"index": 0, "name": "打击", "is_upgraded": True}],
                "can_skip": True,
            },
        }
        context = build_decision_context(raw)
        assert context is not None
        evidence = SqliteFactsRetriever(self.db).retrieve(context, 20)
        ids = {item.evidence_id for item in evidence}
        self.assertIn("sqlite:cards:strike_ironclad_upgrade", ids)
        self.assertNotIn("sqlite:cards:strike_ironclad", ids)

    def test_ignores_draw_pile_when_collecting_current_facts(self):
        self.raw["player"]["draw_pile"] = [{
            "id": "pile_only", "name": "牌堆专用", "cost": "1", "description": "不应因牌堆而检索。",
        }]
        context = build_decision_context(self.raw)
        assert context is not None
        ids = {item.evidence_id for item in SqliteFactsRetriever(self.db).retrieve(context, 20)}
        self.assertNotIn("sqlite:cards:pile_only", ids)

    def test_long_card_name_does_not_match_unrelated_short_power_name(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO interactions(id,description,description_wiki) VALUES (2,'不动翻倍格挡','raw')")
            connection.execute("INSERT INTO interaction_pages VALUES (2,'不动')")
            connection.execute("INSERT INTO interactions(id,description,description_wiki) VALUES (3,'岿然不动相关交互','raw')")
            connection.execute("INSERT INTO interaction_pages VALUES (3,'岿然不动')")
        self.raw['player']['draw_pile'] = [
            {'name': '岿然不动+', 'cost': '2', 'description': '获得40点格挡。 消耗。'},
        ]
        context = build_decision_context(self.raw)
        ids = {e.evidence_id for e in SqliteFactsRetriever(self.db).retrieve(context, 20)}
        self.assertNotIn('sqlite:interactions:2', ids)
        self.assertIn('sqlite:interactions:3', ids)

    def test_explicit_power_and_description_reference_still_match_interactions(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO interactions(id,description,description_wiki) VALUES (2,'不动翻倍格挡','raw')")
            connection.execute("INSERT INTO interaction_pages VALUES (2,'不动')")
        for status in [
            {'id': 'UNMOVING', 'name': '不动', 'amount': 1},
            {'id': 'OTHER', 'name': '另一状态', 'description': '在回合开始获得不动。'},
        ]:
            with self.subTest(status=status):
                self.raw['player']['status'] = [status]
                context = build_decision_context(self.raw)
                ids = {e.evidence_id for e in SqliteFactsRetriever(self.db).retrieve(context, 20)}
                self.assertIn('sqlite:interactions:2', ids)

    def test_auto_combines_sqlite_facts_with_existing_guides_and_respects_top_k(self):
        with patch("sts2_advisor.integrations.knowledge.factory.ChromaRetriever", side_effect=RuntimeError("not installed")):
            retriever = create_retriever("auto", self.root / "chroma", self.knowledge,
                                         facts_path=self.db, seeds_path=self.seeds)
        evidence = retriever.retrieve(self.context, 3)
        self.assertEqual(len(evidence), 3)
        self.assertIn("survival-guide", {item.evidence_id for item in evidence})
        self.assertTrue(any(item.evidence_id.startswith("sqlite:") for item in evidence))
        self.assertEqual(retriever.retrieve(self.context, 0), [])

    def test_missing_database_falls_back_without_creating_it(self):
        missing = self.root / "missing.sqlite3"
        with patch("sts2_advisor.integrations.knowledge.factory.ChromaRetriever", side_effect=RuntimeError("not installed")):
            retriever = create_retriever("auto", self.root / "chroma", self.knowledge,
                                         facts_path=missing, seeds_path=self.seeds)
        self.assertEqual([item.evidence_id for item in retriever.retrieve(self.context)], ["survival-guide"])
        self.assertFalse(missing.exists())

    def test_explicit_json_and_none_modes_keep_existing_behavior(self):
        for mode, expected in [("none", []), ("json", ["survival-guide"])]:
            with self.subTest(mode=mode):
                retriever = create_retriever(mode, self.root / "chroma", self.knowledge,
                                             facts_path=self.db, seeds_path=self.seeds)
                self.assertEqual([item.evidence_id for item in retriever.retrieve(self.context)], expected)

    def test_active_power_and_enemy_outrank_pickup_only_relics(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO powers(id,name,description) VALUES ('vulnerable','易伤','受到攻击伤害增加50%。')")
            connection.execute("INSERT INTO monsters(id,name,powers) VALUES ('nibbit','小啃兽','当前敌人的机制。')")
            for index in range(4):
                connection.execute('INSERT INTO relics(id,name,description) VALUES (?,?,?)',
                                   (f'pickup{index}', f'开局遗物{index}', '拾起时，获得金币。'))
        self.raw['player']['relics'] = [{'id': f'PICKUP{i}', 'name': f'开局遗物{i}'} for i in range(4)]
        self.raw['player']['status'] = [{'id': 'VULNERABLE', 'name': '易伤', 'amount': 1}]
        ctx = build_decision_context(self.raw)
        ids = [e.evidence_id for e in SqliteFactsRetriever(self.db).retrieve(ctx, 2)]
        self.assertEqual(ids, ['sqlite:powers:vulnerable', 'sqlite:monsters:nibbit'])

    def test_reward_card_fact_outranks_generic_relic_interaction(self):
        raw = {'state_type': 'card_reward', 'run': {}, 'player': self.raw['player'],
               'card_reward': {'cards': [{'id': 'strike_ironclad', 'name': '打击', 'index': 0}]}}
        ctx = build_decision_context(raw)
        ids = [e.evidence_id for e in SqliteFactsRetriever(self.db).retrieve(ctx, 1)]
        self.assertEqual(ids, ['sqlite:cards:strike_ironclad'])

    def test_active_interaction_not_lost_to_low_id_interactions_before_ranking(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO powers(id,name,description) VALUES ('vulnerable','易伤','受到攻击伤害增加50%。')")
            connection.execute("INSERT INTO interactions(id,description,description_wiki) VALUES (2,'无关牌的说明','raw')")
            connection.execute("INSERT INTO interaction_pages VALUES (2,'其他牌')")
            connection.execute("INSERT INTO interactions(id,description,description_wiki) VALUES (50,'当前易伤的交互','raw')")
            connection.execute("INSERT INTO interaction_pages VALUES (50,'易伤')")
        self.raw['player']['draw_pile'] = [{'name': '其他牌'}]
        self.raw['player']['status'] = [{'id': 'VULNERABLE', 'name': '易伤', 'amount': 1}]
        ctx = build_decision_context(self.raw)
        ids = [e.evidence_id for e in SqliteFactsRetriever(self.db).retrieve(ctx, 2)]
        self.assertEqual(ids, ['sqlite:powers:vulnerable', 'sqlite:interactions:50'])

    def test_shop_card_id_and_name_are_collected_as_current_candidates(self):
        from dataclasses import replace
        ctx = replace(self.context, player={**self.context.player, 'hand': []},
                      decision_type='shop', state_type='shop', situation={
            'shop': {'items': [{'card_id': 'strike_ironclad', 'card_name': '打击', 'category': 'card'}]},
        })
        ids = [e.evidence_id for e in SqliteFactsRetriever(self.db).retrieve(ctx, 1)]
        self.assertEqual(ids, ['sqlite:cards:strike_ironclad'])

    def test_unused_potion_effect_is_not_ranked_as_an_active_power(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO powers(id,name,description) VALUES ('flex_potion','肌肉药水','药水带来的状态。')")
            connection.execute("INSERT INTO powers(id,name,description) VALUES ('vulnerable','易伤','增加伤害。')")
            connection.execute("INSERT INTO monsters(id,name) VALUES ('nibbit','小啃兽')")
        self.raw['player']['potions'] = [{'id': 'FLEX_POTION', 'name': '肌肉药水'}]
        self.raw['player']['status'] = [{'id': 'VULNERABLE', 'name': '易伤', 'amount': 1}]
        ctx = build_decision_context(self.raw)
        ids = [e.evidence_id for e in SqliteFactsRetriever(self.db).retrieve(ctx, 2)]
        self.assertEqual(ids, ['sqlite:powers:vulnerable', 'sqlite:monsters:nibbit'])


if __name__ == "__main__":
    unittest.main()
