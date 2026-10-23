from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from sts2_advisor.model import build_prompt
from sts2_advisor.normalize import build_decision_context
from sts2_advisor.retrieval import SqliteFactsRetriever, create_retriever
from tools.build_wiki_db import SCHEMA_SQL


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

    def test_retrieves_matching_ids_and_interactions_into_llm_prompt(self):
        evidence = SqliteFactsRetriever(self.db).retrieve(self.context, 20)
        by_id = {item.evidence_id: item for item in evidence}
        self.assertIn("sqlite:cards:strike_ironclad", by_id)
        self.assertIn("sqlite:relics:burning_blood", by_id)
        self.assertIn("sqlite:interactions:1", by_id)
        self.assertNotIn("sqlite:cards:strike_silent", by_id)
        self.assertNotIn("sqlite:cards:unrelated", by_id)
        self.assertNotIn("sqlite:cards:strike_ironclad_upgrade", by_id)
        prompt = build_prompt(self.context, evidence)
        self.assertIn("燃烧之血在战斗结束时结算。", prompt)
        self.assertIn("sqlite:interactions:1", prompt)
        self.assertIn("造成6点伤害。", prompt)

    def test_upgraded_card_uses_upgraded_facts(self):
        self.raw["player"]["hand"][0]["is_upgraded"] = True
        self.raw["player"]["hand"][0]["name"] = "Strike"
        context = build_decision_context(self.raw)
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

    def test_auto_combines_sqlite_facts_with_existing_guides_and_respects_top_k(self):
        with patch("sts2_advisor.retrieval.ChromaRetriever", side_effect=RuntimeError("not installed")):
            retriever = create_retriever("auto", self.root / "chroma", self.knowledge, facts_path=self.db)
        evidence = retriever.retrieve(self.context, 3)
        self.assertEqual(len(evidence), 3)
        self.assertIn("survival-guide", {item.evidence_id for item in evidence})
        self.assertTrue(any(item.evidence_id.startswith("sqlite:") for item in evidence))
        self.assertEqual(retriever.retrieve(self.context, 0), [])

    def test_missing_database_falls_back_without_creating_it(self):
        missing = self.root / "missing.sqlite3"
        with patch("sts2_advisor.retrieval.ChromaRetriever", side_effect=RuntimeError("not installed")):
            retriever = create_retriever("auto", self.root / "chroma", self.knowledge, facts_path=missing)
        self.assertEqual([item.evidence_id for item in retriever.retrieve(self.context)], ["survival-guide"])
        self.assertFalse(missing.exists())

    def test_explicit_json_and_none_modes_keep_existing_behavior(self):
        for mode, expected in [("none", []), ("json", ["survival-guide"])]:
            with self.subTest(mode=mode):
                retriever = create_retriever(mode, self.root / "chroma", self.knowledge, facts_path=self.db)
                self.assertEqual([item.evidence_id for item in retriever.retrieve(self.context)], expected)


if __name__ == "__main__":
    unittest.main()
