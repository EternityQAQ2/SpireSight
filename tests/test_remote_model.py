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

from sts2_advisor.cli import choose_backend_interactively, load_environment
from sts2_advisor.logging_config import close_logging, configure_logging
from sts2_advisor.model import OpenAIAdvisor, cache_token_counts
from sts2_advisor.normalize import build_decision_context


FIXTURE = Path(__file__).parent / "fixtures" / "combat_state.json"


class RemoteModelTests(unittest.TestCase):
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
            backend = OpenAIAdvisor("remote-model", api_key="secret", base_url="https://example.test/v1")
            advice = backend.advise(context, [])

        self.assertEqual(advice.candidate_id, context.candidates[0].candidate_id)
        self.assertEqual(calls[0]["model"], "remote-model")
        self.assertEqual(calls[0]["temperature"], 0)

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
