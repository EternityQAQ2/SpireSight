from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sts2_advisor.domain import DecisionContext, Evidence
from sts2_advisor.retrieval import (
    SeedInjectingRetriever,
    create_retriever,
    load_seeds,
)

SEED_RECORD = {
    "evidence_id": "builtin-combat-survival-001",
    "text": "当敌人本回合将造成伤害时，比较预计伤害、现有格挡与生命值。"
            "若无法在本回合可靠击杀，优先避免不必要的生命损失。",
    "source_title": "项目内置基础战斗原则",
    "source_url": "",
    "metadata": {"character": "all", "decision_types": ["combat_next_action"],
                 "license": "project", "review_status": "seed"},
}


def make_context(decision_type: str, character: str = "铁甲战士") -> DecisionContext:
    return DecisionContext("1.0", "test", decision_type, decision_type, {},
                           {"character": character}, {}, [])


class StubRetriever:
    def __init__(self, items: list[Evidence]) -> None:
        self.items = items

    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        return self.items


class SeedInjectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.seeds = self.root / "seeds.jsonl"
        self.seeds.write_text(json.dumps(SEED_RECORD, ensure_ascii=False) + "\n", encoding="utf-8")
        # 语料为空：复现 rebuild 清空 knowledge.jsonl 后语料侧不提供任何知识的事故现场
        self.knowledge = self.root / "knowledge.jsonl"
        self.knowledge.write_text("", encoding="utf-8")

    def retrieve(self, mode: str, context: DecisionContext):
        with patch("sts2_advisor.retrieval.ChromaRetriever",
                   side_effect=RuntimeError("not installed")):
            retriever = create_retriever(mode, self.root / "chroma", self.knowledge,
                                         facts_path=self.root / "missing.sqlite3",
                                         seeds_path=self.seeds)
        return retriever.retrieve(context)

    def test_seed_survives_corpus_loss_in_json_and_auto_modes(self):
        # 2026-10-25 事故回归：种子规则曾随 knowledge.jsonl rebuild 一并丢失，
        # 此后战斗证据再无生存基线。注入不依赖语料库，json/chroma/auto 均须生效。
        for mode in ("json", "auto"):
            with self.subTest(mode=mode):
                evidence = self.retrieve(mode, make_context("combat_next_action"))
                self.assertEqual(evidence[0].evidence_id, "builtin-combat-survival-001")

    def test_mode_none_is_an_explicit_opt_out(self):
        self.assertEqual(self.retrieve("none", make_context("combat_next_action")), [])

    def test_seed_routed_by_decision_type(self):
        self.assertEqual(self.retrieve("json", make_context("card_reward")), [])

    def test_seed_prepended_without_consuming_top_k_and_dedupes(self):
        seeds = load_seeds(self.seeds)
        combat = make_context("combat_next_action")

        # 语料侧空手而归：seed 仍注入
        result = SeedInjectingRetriever(StubRetriever([]), seeds).retrieve(combat, 6)
        self.assertEqual([item.evidence_id for item in result], ["builtin-combat-survival-001"])

        # 语料侧满额：seed 前置追加，不挤占原有 top_k 结果
        fillers = [Evidence(f"fact:{index}", "事实", "事实") for index in range(6)]
        result = SeedInjectingRetriever(StubRetriever(fillers), seeds).retrieve(combat, 6)
        self.assertEqual(len(result), 7)
        self.assertEqual(result[0].evidence_id, "builtin-combat-survival-001")

        # 检索结果已含同 id：去重，不重复注入
        duplicated = [Evidence("builtin-combat-survival-001", "重复", "来源")] + fillers
        result = SeedInjectingRetriever(StubRetriever(duplicated), seeds).retrieve(combat, 6)
        ids = [item.evidence_id for item in result]
        self.assertEqual(ids.count("builtin-combat-survival-001"), 1)
        self.assertEqual(len(result), 7)

    def test_load_seeds_rejects_invalid_lines(self):
        bad = self.root / "bad.jsonl"
        bad.write_text('{"evidence_id": "x"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_seeds(bad)

    def test_shipped_seeds_file_contains_survival_rule_and_default_wiring_uses_it(self):
        # data/seeds.jsonl 属于仓库数据：规则被误删/改名/未随仓库分发时在此失败
        default_path = Path(__file__).resolve().parents[1] / "data" / "seeds.jsonl"
        self.assertIn("builtin-combat-survival-001",
                      {seed.evidence_id for seed in load_seeds(default_path)})
        with patch("sts2_advisor.retrieval.ChromaRetriever", side_effect=RuntimeError("not installed")):
            retriever = create_retriever("json", self.root / "chroma", self.knowledge,
                                         facts_path=self.root / "missing.sqlite3")
        evidence = retriever.retrieve(make_context("combat_next_action"))
        self.assertIn("builtin-combat-survival-001", {item.evidence_id for item in evidence})


if __name__ == "__main__":
    unittest.main()
