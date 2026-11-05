from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

from sts2_advisor.decision.tools import DecisionTools
from sts2_advisor.game.state import build_decision_context
from sts2_advisor.integrations.models.remote import OpenAIAdvisor


class DecisionToolTests(unittest.TestCase):
    def setUp(self):
        raw = json.loads((Path(__file__).parent / "fixtures/combat_state.json").read_text(encoding="utf-8"))
        self.ctx = build_decision_context(raw)
        self.args = json.dumps({"decision_id": self.ctx.decision_id, "candidate_ids": ["card:2", "combat:end_turn"]})

    def test_readonly_projection_and_bad_arguments(self):
        tool = DecisionTools(self.ctx)
        before = copy.deepcopy(self.ctx.to_dict())
        result = tool.call("evaluate_combat_plan", self.args)
        self.assertTrue(result["exact"])
        self.assertEqual(result["projection"]["block"], 5)
        self.assertEqual(result["trace"][0]["energy_left"], 2)
        self.assertEqual(self.ctx.to_dict(), before)
        for name, args in [("execute_game", self.args), ("evaluate_combat_plan", "[]"),
                           ("evaluate_combat_plan", self.args.replace(self.ctx.decision_id, "old")),
                           ("evaluate_combat_plan", self.args.replace("card:2", "card:999"))]:
            self.assertFalse(tool.call(name, args)["ok"])

    def test_noncombat_relics_do_not_disable_forecast(self):
        from dataclasses import replace
        player = copy.deepcopy(self.ctx.player)
        player.setdefault("relics", []).extend([
            {"id": "STONE_HUMIDIFIER"}, {"id": "BOOK_OF_FIVE_RINGS"}])
        ctx = replace(self.ctx, player=player)
        self.assertTrue(DecisionTools(ctx).evaluate(["card:2"])['exact'])

    def test_recorded_scroll_boxes_and_slippery(self):
        rows = json.loads((Path(__file__).parent / "fixtures/function_calling_log_cases.json").read_text(encoding="utf-8"))
        opening = build_decision_context(rows["opening"])
        self.assertIsNotNone(opening)
        self.assertIn("SCROLL_BOXES", [r["id"] for r in opening.player["relics"]])
        attack = next(c for c in opening.candidates if c.action_type == "play_card")
        self.assertTrue(DecisionTools(opening).evaluate([attack.candidate_id])["exact"])
        inklets = build_decision_context(rows["slippery"])
        self.assertIsNotNone(inklets)
        first = next(c for c in inklets.candidates if c.action_type == "play_card" and c.params.get("card_index") == 0 and c.params.get("target") == "INKLET_0")
        projection = DecisionTools(inklets).evaluate([first.candidate_id])
        self.assertTrue(projection["exact"])
        self.assertEqual(projection["projection"]["damage"], 1)

    def test_rage_counts_cards_and_not_hits_or_its_own_cast(self):
        from dataclasses import replace
        from sts2_advisor.contracts import Candidate
        player = copy.deepcopy(self.ctx.player)
        player["hand"] = [
            {"index": 0, "id": "RAGE", "type": "Skill", "cost": "0",
             "description": "打出此牌后，你在这个回合内每打出一张攻击牌，获得3点格挡。"},
            {"index": 1, "id": "TWIN_STRIKE", "type": "Attack", "cost": "1",
             "description": "造成5点伤害两次。"}]
        target = self.ctx.situation["battle"]["enemies"][0]["entity_id"]
        ctx = replace(self.ctx, player=player, candidates=[
            Candidate("rage", "play_card", "狂怒", {"card_index": 0}),
            Candidate("twin", "play_card", "双重", {"card_index": 1, "target": target})])
        tool = DecisionTools(ctx)
        self.assertEqual(tool.evaluate(["rage"])["projection"]["block"], 0)
        result = tool.evaluate(["rage", "twin"])
        self.assertEqual(result["projection"]["block"], 3)
        self.assertEqual(result["projection"]["damage"], 10)

    def backend(self, create, **options):
        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = NS(completions=NS(create=create))
        with patch.dict(sys.modules, {"openai": NS(OpenAI=FakeOpenAI)}):
            return OpenAIAdvisor("test", api_key="test", base_url="https://example.test", **options)

    def response(self, tool=False):
        if tool:
            message = NS(content=None, reasoning_content="tool reasoning", tool_calls=[NS(
                id="call-1", function=NS(name="evaluate_combat_plan", arguments=self.args))])
        else:
            message = NS(content=json.dumps({"candidate_id": "card:2", "reason": "比较后防御", "confidence": 0.7}))
        return NS(choices=[NS(message=message, finish_reason="tool_calls" if tool else "stop")])

    def test_tool_result_and_reasoning_are_returned_before_final_decision(self):
        requests = []
        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            return self.response(tool=len(requests) == 1)
        advice = self.backend(create).advise(self.ctx, [])
        self.assertEqual(advice.candidate_id, "card:2")
        self.assertEqual(len(requests), 2)
        self.assertIn("tools", requests[0])
        messages = requests[1]["messages"]
        self.assertEqual(messages[-2]["reasoning_content"], "tool reasoning")
        self.assertEqual(messages[-1]["tool_call_id"], "call-1")
        self.assertTrue(json.loads(messages[-1]["content"])["exact"])

    def test_tool_result_overrules_false_kill_claim(self):
        def create(**kwargs):
            if len(kwargs["messages"]) == 2:
                return self.response(tool=True)
            return NS(choices=[NS(message=NS(content=json.dumps({
                "candidate_id": "card:2", "reason": "此牌可击杀敌人", "confidence": 0.9,
            })), finish_reason="stop")])
        advice = self.backend(create).advise(self.ctx, [])
        self.assertNotIn("可击杀", advice.reason)
        self.assertLessEqual(advice.confidence, 0.4)

    def test_endless_tool_calls_are_bounded(self):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return self.response(tool=True)
        with self.assertRaisesRegex(RuntimeError, "工具调用"):
            self.backend(create).advise(self.ctx, [])
        self.assertEqual(len(requests), 4)
        self.assertEqual(requests[-1]["tool_choice"], "none")

    def test_opt_out_omits_tools(self):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return self.response()
        self.backend(create, function_calling=False).advise(self.ctx, [])
        self.assertNotIn("tools", requests[0])


if __name__ == "__main__":
    unittest.main()
