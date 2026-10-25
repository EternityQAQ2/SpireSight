from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sts2_advisor.cli import choose_backend_interactively, environment_flag, load_environment
from sts2_advisor.domain import Evidence
from sts2_advisor.logging_config import close_logging, configure_logging
from sts2_advisor.model import (
    OpenAIAdvisor,
    SYSTEM_PROMPT,
    build_prompt,
    cache_token_counts,
    clear_codex_cache,
    model_context_payload,
)
from sts2_advisor.normalize import build_decision_context


FIXTURE = Path(__file__).parent / "fixtures" / "combat_state.json"


class RemoteModelTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_codex_cache()
    def test_environment_flag(self) -> None:
        with patch.dict(os.environ, {"FEATURE": "true"}, clear=True):
            self.assertTrue(environment_flag("FEATURE", False))
            self.assertFalse(environment_flag("MISSING", False))
        with patch.dict(os.environ, {"FEATURE": "invalid"}, clear=True):
            with self.assertRaisesRegex(ValueError, "true 或 false"):
                environment_flag("FEATURE", False)

    def test_reads_deepseek_and_openai_cache_usage(self) -> None:
        self.assertEqual(
            cache_token_counts({"prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 128}),
            (64, 128),
        )
        self.assertEqual(
            cache_token_counts({"prompt_tokens_details": {"cached_tokens": 96}}),
            (96, None),
        )

    def test_interactive_mode_selection(self) -> None:
        self.assertEqual(choose_backend_interactively(lambda _: "1", lambda _: None), "transformers")
        self.assertEqual(choose_backend_interactively(lambda _: "2", lambda _: None), "openai")

    def test_dotenv_does_not_override_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=remote-model\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "process-key"}, clear=True):
                load_environment(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "process-key")
                self.assertEqual(os.environ["OPENAI_MODEL"], "remote-model")

    def test_openai_backend_returns_structured_advice(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        context = build_decision_context(raw)
        assert context is not None
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                payload = {
                    "decision_id": context.decision_id,
                    "candidate_id": context.candidates[0].candidate_id,
                    "reason": "测试建议",
                    "confidence": 0.8,
                    "alternative_ids": [],
                    "evidence_ids": [],
                    "warnings": [],
                }
                message = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            backend = OpenAIAdvisor(
                "remote-model", api_key="secret", base_url="https://example.test/v1",
                thinking="disabled",
            )
            advice = backend.advise(context, [])

        self.assertEqual(advice.candidate_id, context.candidates[0].candidate_id)
        self.assertEqual(calls[0]["model"], "remote-model")
        self.assertEqual(calls[0]["temperature"], 0)
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[0]["extra_body"], {"thinking": {"type": "disabled"}})

    def test_prompt_puts_deck_codex_in_stable_prefix_and_shrinks_dynamic_region(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        raw["player"]["draw_pile"].append(
            {"name": "灼热攻击", "cost": "1", "description": "牌堆独有的冗长说明"}
        )
        context = build_decision_context(raw)
        assert context is not None
        evidence = Evidence(
            "e1", "核心事实", "来源", "https://should-not-be-sent.test",
            {"license": "metadata-should-not-be-sent"},
        )
        prompt = build_prompt(context, [evidence])
        self.assertIn("核心事实", prompt)
        self.assertNotIn("should-not-be-sent", prompt)
        self.assertEqual(prompt.count("牌堆独有的冗长说明"), 1)
        self.assertLess(prompt.index('"evidence"'), prompt.index('"decision_id"'))

        payload = json.loads(prompt.removeprefix("输入 JSON："))
        self.assertEqual(
            list(payload)[:4],
            ["stable", "run", "state_type", "decision_type"],
        )
        self.assertNotIn("run", payload["stable"])

        deck = payload["stable"]["deck"]
        codex_keys = {entry["k"] for entry in deck}
        # 图鉴键为卡牌名（牌堆 API 没有 id，名称是两侧唯一的共同标识）。
        self.assertEqual(codex_keys, {"打击", "痛击", "防御", "灼热攻击"})
        strike = next(entry for entry in deck if entry["k"] == "打击")
        self.assertEqual(strike["description"], "造成6点伤害。")

        hand = payload["player"]["hand"]
        self.assertTrue(hand)
        for entry in hand:
            self.assertNotIn("description", entry)
            self.assertIn("k", entry)
            self.assertIn(entry["k"], codex_keys)
        draw_pile = payload["player"]["draw_pile"]
        self.assertTrue(all(isinstance(value, int) for value in draw_pile.values()))
        self.assertTrue(set(draw_pile) <= codex_keys)
        self.assertEqual(draw_pile, {"防御": 2, "灼热攻击": 1})

    def test_codex_prefix_survives_cards_moving_between_hand_and_piles(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        moved = json.loads(FIXTURE.read_text(encoding="utf-8"))
        card = moved["player"]["hand"].pop(0)
        card.pop("index", None)
        moved["player"]["discard_pile"].append(card)
        moved["player"]["hp"] = moved["player"]["hp"] - 3

        first = build_decision_context(raw)
        second = build_decision_context(moved)
        assert first is not None and second is not None
        payload_first = model_context_payload(first)
        payload_second = model_context_payload(second)

        # 打出的牌从手牌进入弃牌堆：图鉴与整个静态前缀必须逐字节不变，
        # 分歧点只允许出现在 player 动态区（hp 变化）之后。
        self.assertEqual(payload_first["stable"], payload_second["stable"])
        text_first = json.dumps(payload_first, ensure_ascii=False, separators=(",", ":"))
        text_second = json.dumps(payload_second, ensure_ascii=False, separators=(",", ":"))
        shared = next(i for i, (a, b) in enumerate(zip(text_first, text_second)) if a != b)
        self.assertGreater(shared, text_first.index('"player"'), text_first[shared - 40 : shared + 40])

    def test_codex_freezes_entries_and_appends_new_cards_at_the_tail(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        first = build_decision_context(raw)
        assert first is not None
        deck_before = model_context_payload(first)["stable"]["deck"]

        # 后续回合：已有卡从手牌进弃牌堆（字段来源变化），新卡生成进手牌。
        later = json.loads(FIXTURE.read_text(encoding="utf-8"))
        card = later["player"]["hand"].pop(0)
        card.pop("index", None)
        later["player"]["discard_pile"].append(card)
        later["player"]["hand"].append(
            {"id": "GENERATED", "name": "新生卡", "cost": "0", "type": "Skill",
             "description": "战中新卡。", "can_play": True}
        )
        second = build_decision_context(later)
        assert second is not None
        deck_after = model_context_payload(second)["stable"]["deck"]

        self.assertEqual([entry["k"] for entry in deck_after][: len(deck_before)],
                         [entry["k"] for entry in deck_before])
        self.assertEqual(deck_after[-1]["k"], "新生卡")
        # 首见即冻结：进堆后不再因牌堆条目缺 type 而改写已有条目。
        self.assertIn("type", next(entry for entry in deck_after if entry["k"] == "打击"))

    def test_system_prompt_keeps_json_contract_without_redundant_formatting_rules(self) -> None:
        self.assertIn("返回 JSON 对象", SYSTEM_PROMPT)
        self.assertIn('"plan_candidate_ids"', SYSTEM_PROMPT)
        self.assertIn("图鉴", SYSTEM_PROMPT)
        self.assertNotIn("不要使用 Markdown", SYSTEM_PROMPT)
        self.assertNotIn("输出紧凑 JSON", SYSTEM_PROMPT)

    def test_empty_remote_content_reports_reasoning_only_response(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        context = build_decision_context(raw)
        assert context is not None

        class FakeOpenAI:
            def __init__(self, **kwargs):
                message = SimpleNamespace(content="", reasoning_content="仍在推理")
                choice = SimpleNamespace(message=message, finish_reason="length")
                response = SimpleNamespace(choices=[choice], usage=None)
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **request: response)
                )

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            backend = OpenAIAdvisor("remote-model", api_key="secret")
            with self.assertRaisesRegex(ValueError, "reasoning_content"):
                backend.advise(context, [])

    def test_remote_response_and_extracted_text_are_logged(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        context = build_decision_context(raw)
        assert context is not None
        payload = {
            "decision_id": context.decision_id,
            "candidate_id": context.candidates[0].candidate_id,
            "reason": "原始文本测试",
            "confidence": 0.8,
        }

        class FakeOpenAI:
            def __init__(self, **kwargs):
                message = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
                choice = SimpleNamespace(message=message, finish_reason="stop")
                response = SimpleNamespace(choices=[choice], id="response-test", usage=None)
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **request: response)
                )

        with tempfile.TemporaryDirectory() as directory:
            log_path = configure_logging(Path(directory), retention_days=3)
            try:
                with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                    OpenAIAdvisor("remote-model", api_key="secret").advise(context, [])
                for handler in logging.getLogger("sts2_advisor").handlers:
                    handler.flush()
                log = log_path.read_text(encoding="utf-8")
            finally:
                close_logging()

        self.assertIn("raw_json=", log)
        self.assertIn("response-test", log)
        self.assertIn("remote_text", log)
        self.assertIn("原始文本测试", log)

    def test_logging_removes_files_older_than_retention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            old = log_dir / "advisor-old.log"
            old.write_text("old", encoding="utf-8")
            timestamp = (datetime.now(timezone.utc) - timedelta(days=4)).timestamp()
            os.utime(old, (timestamp, timestamp))
            current = configure_logging(log_dir, retention_days=3)
            self.assertFalse(old.exists())
            self.assertTrue(current.exists())
            close_logging()


if __name__ == "__main__":
    unittest.main()
