from __future__ import annotations

import unittest
from unittest.mock import patch

from sts2_advisor.domain import DecisionContext
from sts2_advisor.retrieval import ChromaRetriever, metadata_matches


def combat_context(enemies: list[str], decision_type: str = "combat_next_action") -> DecisionContext:
    return DecisionContext(
        "1.0",
        "test",
        decision_type,
        "boss",
        {},
        {"character": "铁甲战士"},
        {"battle": {"enemies": [{"name": name} for name in enemies]}},
        [],
    )


class EnemyMetadataFilterTests(unittest.TestCase):
    def test_boss_guide_matches_only_its_enemy(self) -> None:
        metadata = {"decision_types": "combat_next_action", "enemies": "墨影幻灵"}
        self.assertTrue(metadata_matches(metadata, combat_context(["墨影幻灵"])))
        self.assertFalse(metadata_matches(metadata, combat_context(["毛绒伏地虫"])))
        self.assertFalse(metadata_matches(metadata, combat_context(["同族神官", "同族信徒"])))

    def test_multi_entity_boss_matches_any_alias(self) -> None:
        metadata = {
            "decision_types": "combat_next_action",
            "enemies": "同族小队,同族神官,同族信徒",
        }
        self.assertTrue(metadata_matches(metadata, combat_context(["同族神官", "同族信徒"])))
        self.assertTrue(metadata_matches(metadata, combat_context(["同族信徒"])))
        self.assertFalse(metadata_matches(metadata, combat_context(["仪式兽"])))

    def test_enemy_scoped_guide_never_leaks_without_a_battle(self) -> None:
        metadata = {"decision_types": "combat_next_action", "enemies": "女王"}
        no_battle = DecisionContext(
            "1.0", "test", "combat_next_action", "boss", {},
            {"character": "铁甲战士"}, {}, [],
        )
        self.assertFalse(metadata_matches(metadata, no_battle))

    def test_guides_without_enemy_metadata_are_unaffected(self) -> None:
        metadata = {"decision_types": "card_reward", "character": "铁甲战士"}
        self.assertTrue(metadata_matches(metadata, combat_context([], "card_reward")))

    def test_jsonl_list_format_enemy_metadata(self) -> None:
        metadata = {
            "decision_types": ["combat_next_action"],
            "enemies": ["帝皇蟹", "碾碎爪", "火箭"],
        }
        self.assertTrue(metadata_matches(metadata, combat_context(["碾碎爪", "火箭"])))
        self.assertFalse(metadata_matches(metadata, combat_context(["仪式兽"])))


class ChromaCandidatePoolTests(unittest.TestCase):
    def _retriever(self, metadatas: list[dict]) -> tuple[ChromaRetriever, dict]:
        calls: dict = {}

        class FakeCollection:
            def count(self) -> int:
                return len(metadatas)

            def query(self, query_texts, n_results):
                calls["n_results"] = n_results
                return {
                    "ids": [[f"guide-{i}" for i in range(len(metadatas))]],
                    "documents": [[f"doc {i}" for i in range(len(metadatas))]],
                    "metadatas": [metadatas],
                }

        retriever = ChromaRetriever.__new__(ChromaRetriever)
        retriever.collection = FakeCollection()
        return retriever, calls

    def test_queries_a_wider_pool_then_filters_by_enemy(self) -> None:
        metadatas = [
            {"decision_types": "combat_next_action", "enemies": "实验体", "source_title": "实验体"},
            {"decision_types": "combat_next_action", "enemies": "同族小队", "source_title": "同族小队"},
            {"decision_types": "combat_next_action", "enemies": "墨影幻灵", "source_title": "墨影幻灵"},
        ]
        retriever, calls = self._retriever(metadatas)
        with patch("sts2_advisor.retrieval.record_event"):
            evidence = retriever.retrieve(combat_context(["墨影幻灵"]), top_k=5)
        self.assertEqual(calls["n_results"], 15)
        self.assertEqual([item.source_title for item in evidence], ["墨影幻灵"])

    def test_result_is_capped_at_top_k(self) -> None:
        metadatas = [
            {"decision_types": "combat_next_action", "enemies": "墨影幻灵", "source_title": "墨影幻灵"}
            for _ in range(8)
        ]
        retriever, _ = self._retriever(metadatas)
        with patch("sts2_advisor.retrieval.record_event"):
            evidence = retriever.retrieve(combat_context(["墨影幻灵"]), top_k=3)
        self.assertEqual(len(evidence), 3)

    def test_normal_combat_returns_no_boss_guides(self) -> None:
        metadatas = [
            {"decision_types": "combat_next_action", "enemies": "实验体", "source_title": "实验体"},
            {"decision_types": "combat_next_action", "enemies": "同族小队", "source_title": "同族小队"},
        ]
        retriever, _ = self._retriever(metadatas)
        with patch("sts2_advisor.retrieval.record_event"):
            evidence = retriever.retrieve(combat_context(["毛绒伏地虫"]), top_k=6)
        self.assertEqual(evidence, [])


if __name__ == "__main__":
    unittest.main()
