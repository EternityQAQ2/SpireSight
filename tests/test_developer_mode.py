from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sts2_advisor.developer import (
    DeveloperTrace,
    activate,
    deactivate,
    dev_block,
    dev_request,
    dev_step,
    dev_usage,
    normalize_usage,
    runtime_environment,
)
from sts2_advisor.contracts import Evidence
from sts2_advisor.decision.prompts import build_prompt
from sts2_advisor.integrations.models.remote import OpenAIAdvisor
from sts2_advisor.game.state import StateBuilder
from sts2_advisor.recording.session import SessionRecorder
from sts2_advisor.application.advisor import AdvisorService
from test_actions import COMBAT, action_api


class StaticRetriever:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence

    def retrieve(self, context, top_k: int = 6) -> list[Evidence]:
        return list(self.evidence)


class DeveloperTraceTests(unittest.TestCase):
    def test_full_prompt_and_token_usage_are_written_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            trace = DeveloperTrace(Path(directory), stream=stream)
            trace.start({"后端": "openai", "模型": "test-model"})
            token = activate(trace)
            try:
                with dev_step("sha256:test", state_type="monster"):
                    dev_block(
                        "提示词", "完整提示词",
                        {"system": "SYSTEM-PROMPT", "user": "USER-PROMPT"},
                        backend="openai",
                    )
                    dev_request("openai", "test-model", attempt=1)
                    dev_usage(
                        "模型",
                        {
                            "prompt_tokens": 1000,
                            "completion_tokens": 20,
                            "total_tokens": 1020,
                            "prompt_tokens_details": {"cached_tokens": 800},
                            "completion_tokens_details": {"reasoning_tokens": 8},
                        },
                        backend="openai", model="test-model", attempt=1,
                    )
            finally:
                deactivate(token)
            trace.finish("completed")
            text = stream.getvalue()
            written = trace.path.read_text(encoding="utf-8")

        self.assertEqual(text, written)
        self.assertEqual(text.count("SYSTEM-PROMPT"), 1)
        self.assertEqual(text.count("USER-PROMPT"), 1)
        self.assertIn("prompt=1000 completion=20 total=1020", text)
        self.assertIn("cached=800", text)
        self.assertIn("命中率=80.0%", text)
        self.assertIn("reasoning=8", text)
        self.assertIn("平均每次请求: 1020.0 token", text)
        self.assertIn("总 token: 1020", text)

    def test_normalize_usage_accepts_deepseek_and_openai_cache_fields(self) -> None:
        deepseek = normalize_usage(
            {"prompt_tokens": 10, "completion_tokens": 2,
             "prompt_cache_hit_tokens": 6, "prompt_cache_miss_tokens": 4}
        )
        assert deepseek is not None
        self.assertEqual((deepseek.cached_tokens, deepseek.cache_miss_tokens), (6, 4))
        self.assertEqual(deepseek.total_tokens, 12)

        openai = normalize_usage({"prompt_tokens_details": {"cached_tokens": 3}})
        assert openai is not None
        self.assertEqual(openai.cached_tokens, 3)
        self.assertIsNone(normalize_usage(None))


class DeveloperModeServiceTests(unittest.TestCase):
    def test_service_trace_contains_environment_evidence_prompt_reasoning_and_output(self) -> None:
        raw = COMBAT
        builder = StateBuilder()
        context = builder(raw)
        assert context is not None
        evidence = Evidence("sqlite:cards:STRIKE", "打击造成6点伤害。", "灰机 Wiki：打击")
        expected_prompt = build_prompt(context, [evidence])
        payload = {
            "decision_id": context.decision_id,
            "candidate_id": context.candidates[0].candidate_id,
            "reason": "开发者模式测试建议",
            "confidence": 0.8,
            "alternative_ids": [],
            "evidence_ids": ["sqlite:cards:STRIKE"],
            "warnings": [],
            "plan_candidate_ids": [context.candidates[0].candidate_id],
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False),
                    reasoning_content="先比较伤害与格挡。",
                ),
                finish_reason="stop",
            )],
            usage={
                "prompt_tokens": 900,
                "completion_tokens": 30,
                "total_tokens": 930,
                "prompt_cache_hit_tokens": 700,
                "prompt_cache_miss_tokens": 200,
            },
        )

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **request: response)
                )

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                backend = OpenAIAdvisor("dev-model", api_key="secret", thinking="disabled")
            stream = io.StringIO()
            trace = DeveloperTrace(Path(directory), stream=stream)
            trace.start(runtime_environment(后端="openai", 模型="dev-model"))
            token = activate(trace)
            try:
                with action_api() as (client, _):
                    recorder = SessionRecorder(Path(directory), enabled=False)
                    service = AdvisorService(
                        client, StaticRetriever([evidence]), backend, recorder,
                        backend_name="openai", state_builder=builder, output=lambda _: None,
                    )
                    self.assertTrue(service.run_once())
            finally:
                deactivate(token)
            trace.finish("completed")
            text = stream.getvalue()

        self.assertIn("[环境信息]", text)
        self.assertIn("后端: openai", text)
        self.assertIn("检索 · 命中 1 条证据", text)
        self.assertIn("打击造成6点伤害。", text)
        self.assertIn("完整提示词（openai · dev-model）", text)
        self.assertIn(expected_prompt, text)
        self.assertEqual(text.count("先比较伤害与格挡。"), 1)
        self.assertIn("开发者模式测试建议", text)
        self.assertIn("prompt=900 completion=30 total=930", text)
        self.assertIn("cached=700", text)
        self.assertIn("最终建议 · 交付结果", text)
        self.assertIn("决策步骤: 1", text)
        self.assertIn("模型请求: 1", text)

    def test_missing_usage_still_counts_the_request(self) -> None:
        raw = COMBAT
        builder = StateBuilder()
        context = builder(raw)
        assert context is not None
        payload = {
            "decision_id": context.decision_id,
            "candidate_id": context.candidates[0].candidate_id,
            "reason": "无 usage 响应",
            "confidence": 0.8,
        }

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                response = SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)),
                        finish_reason="stop",
                    )],
                    usage=None,
                )
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **request: response)
                )

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                backend = OpenAIAdvisor("dev-model", api_key="secret")
            stream = io.StringIO()
            trace = DeveloperTrace(Path(directory), stream=stream)
            trace.start({"后端": "openai"})
            token = activate(trace)
            try:
                with action_api() as (client, _):
                    recorder = SessionRecorder(Path(directory), enabled=False)
                    service = AdvisorService(
                        client, StaticRetriever([]), backend, recorder,
                        backend_name="openai", state_builder=builder, output=lambda _: None,
                    )
                    service.run_once()
            finally:
                deactivate(token)
            trace.finish("completed")
            text = stream.getvalue()

        self.assertIn("服务未返回 usage", text)
        self.assertIn("模型请求: 1", text)
        self.assertIn("总 token: 0", text)


if __name__ == "__main__":
    unittest.main()
