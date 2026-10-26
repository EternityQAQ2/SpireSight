from __future__ import annotations

import importlib.util
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ingest_guides", ROOT / "tools" / "ingest_guides.py"
)
ingest_guides = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ingest_guides)


class DecisionTypeRoutingTest(unittest.TestCase):
    def test_boss_category_routes_to_combat(self) -> None:
        self.assertEqual(
            ingest_guides.resolve_decision_types("bosses/第一幕/瀑布巨兽.md", "bosses", []),
            ["combat_next_action"],
        )

    def test_builds_category_keeps_card_reward_and_shop(self) -> None:
        self.assertEqual(
            ingest_guides.resolve_decision_types(
                "builds/铁甲战士流派/力量.md", "builds", []
            ),
            ["card_reward", "shop"],
        )

    def test_unknown_category_falls_back_to_default(self) -> None:
        self.assertEqual(
            ingest_guides.resolve_decision_types("events/某事件.md", "events", []),
            list(ingest_guides.DEFAULT_DECISION_TYPES),
        )

    def test_file_override_beats_category(self) -> None:
        with unittest.mock.patch.dict(
            ingest_guides.DECISION_TYPES_BY_FILE,
            {"bosses/通用备战清单.md": ("map", "rest_site", "shop", "card_reward")},
        ):
            self.assertEqual(
                ingest_guides.resolve_decision_types(
                    "bosses/通用备战清单.md", "bosses", []
                ),
                ["map", "rest_site", "shop", "card_reward"],
            )

    def test_cli_override_beats_file(self) -> None:
        with unittest.mock.patch.dict(
            ingest_guides.DECISION_TYPES_BY_FILE,
            {"bosses/瀑布巨兽.md": ("map",)},
        ):
            self.assertEqual(
                ingest_guides.resolve_decision_types(
                    "bosses/瀑布巨兽.md", "bosses", ["combat_next_action"]
                ),
                ["combat_next_action"],
            )


class PlanGuidesTest(unittest.TestCase):
    def test_act_folder_keeps_character_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "bosses" / "第一幕" / "瀑布巨兽.md"
            target.parent.mkdir(parents=True)
            target.write_text(
                "# 瀑布巨兽\n\n[来源](https://sts2.huijiwiki.com/wiki/瀑布巨兽)\n",
                encoding="utf-8",
            )
            plans = ingest_guides.plan_guides(root)
            self.assertEqual(len(plans), 1)
            plan = plans[0]
            self.assertEqual(plan["rel_path"], "bosses/第一幕/瀑布巨兽.md")
            self.assertEqual(plan["character"], "all")
            self.assertEqual(plan["title"], "瀑布巨兽")
            self.assertEqual(plan["license"], "CC BY-NC-SA 灰机wiki")

    def test_stream_folder_strips_suffix_from_character(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "builds" / "铁甲战士流派" / "力量.md"
            target.parent.mkdir(parents=True)
            target.write_text("# 力量\n", encoding="utf-8")
            plan = ingest_guides.plan_guides(root)[0]
            self.assertEqual(plan["character"], "铁甲战士")
            self.assertEqual(plan["title"], "铁甲战士：力量")

    def test_stale_file_overrides_are_reported(self) -> None:
        plans = [{"rel_path": "bosses/瀑布巨兽.md"}]
        with unittest.mock.patch.dict(
            ingest_guides.DECISION_TYPES_BY_FILE,
            {"bosses/已删除.md": ("combat_next_action",), "bosses/瀑布巨兽.md": ("combat_next_action",)},
        ):
            self.assertEqual(
                ingest_guides.stale_file_overrides(plans), ["bosses/已删除.md"]
            )


class EnemyRoutingTest(unittest.TestCase):
    def test_single_entity_boss_uses_file_stem(self) -> None:
        self.assertEqual(
            ingest_guides.resolve_enemies("bosses/第一幕/瀑布巨兽.md", "bosses", "瀑布巨兽"),
            ["瀑布巨兽"],
        )

    def test_multi_entity_boss_gets_aliases(self) -> None:
        self.assertEqual(
            ingest_guides.resolve_enemies("bosses/第一幕/同族小队.md", "bosses", "同族小队"),
            ["同族小队", "同族神官", "同族信徒"],
        )
        self.assertEqual(
            ingest_guides.resolve_enemies("bosses/第二幕/帝皇蟹.md", "bosses", "帝皇蟹"),
            ["帝皇蟹", "碾碎爪", "火箭"],
        )

    def test_builds_have_no_enemy_scope(self) -> None:
        self.assertEqual(
            ingest_guides.resolve_enemies("builds/铁甲战士流派/力量.md", "builds", "铁甲战士：力量"),
            [],
        )

    def test_plan_guides_carries_enemy_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "bosses" / "第三幕" / "女王.md"
            target.parent.mkdir(parents=True)
            target.write_text(
                "# 女王\n\n[来源](https://sts2.huijiwiki.com/wiki/女王)\n", encoding="utf-8"
            )
            plan = ingest_guides.plan_guides(root)[0]
            self.assertEqual(plan["enemies"], ["女王", "火炬头聚合体"])


if __name__ == "__main__":
    unittest.main()
