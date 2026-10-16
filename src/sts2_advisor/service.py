from __future__ import annotations

import sys
import time
from typing import Callable

from .api import Sts2ApiError, Sts2RestClient
from .domain import Advice
from .model import AdvisorBackend, RuleAdvisor
from .normalize import build_decision_context
from .recording import SessionRecorder
from .render import render_console
from .retrieval import Retriever
from .validation import AdviceValidationError, validate_advice


class AdvisorService:
    def __init__(
        self,
        client: Sts2RestClient,
        retriever: Retriever,
        backend: AdvisorBackend,
        recorder: SessionRecorder,
        *,
        backend_name: str,
        fallback_backend: AdvisorBackend | None = None,
        top_k: int = 6,
        output: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.backend = backend
        self.backend_name = backend_name
        self.fallback_backend = fallback_backend
        self.recorder = recorder
        self.top_k = top_k
        self.output = output
        self.last_decision_id: str | None = None

    def run_once(self, *, force: bool = False) -> bool:
        raw_state = self.client.get_singleplayer_state()
        context = build_decision_context(raw_state)
        if context is None:
            state_type = raw_state.get("state_type", "unknown")
            self.output(f"当前状态 {state_type!r} 没有可建议的决策，保持等待。")
            return False
        if not force and context.decision_id == self.last_decision_id:
            return False

        evidence = self.retriever.retrieve(context, self.top_k)
        backend_name = self.backend_name
        try:
            advice = validate_advice(self.backend.advise(context, evidence), context, evidence)
        except (AdviceValidationError, ValueError, RuntimeError) as exc:
            if self.fallback_backend is None:
                raise
            backend_name = "rule-fallback"
            self.output(f"模型输出不可用，改用规则后端：{exc}")
            advice = validate_advice(self.fallback_backend.advise(context, evidence), context, evidence)

        self.output(render_console(advice, context, evidence))
        self._record_safely(raw_state, context, evidence, advice, backend_name)
        self.last_decision_id = context.decision_id
        return True

    def watch(self, poll_interval: float, stable_reads: int = 2) -> None:
        self.output("开始监听 STS2MCP。按 Ctrl+C 停止；本程序不会向游戏发送 POST 请求。")
        observed_id: str | None = None
        observed_count = 0
        error_message: str | None = None
        while True:
            try:
                raw_state = self.client.get_singleplayer_state()
                context = build_decision_context(raw_state)
                current_id = context.decision_id if context else None
                if current_id == observed_id:
                    observed_count += 1
                else:
                    observed_id = current_id
                    observed_count = 1
                if context and observed_count >= stable_reads and current_id != self.last_decision_id:
                    self._process(raw_state, context)
                error_message = None
            except Sts2ApiError as exc:
                if str(exc) != error_message:
                    print(f"API 错误：{exc}", file=sys.stderr)
                    error_message = str(exc)
            time.sleep(poll_interval)

    def _process(self, raw_state: dict, context) -> Advice:
        evidence = self.retriever.retrieve(context, self.top_k)
        backend_name = self.backend_name
        try:
            advice = validate_advice(self.backend.advise(context, evidence), context, evidence)
        except (AdviceValidationError, ValueError, RuntimeError) as exc:
            if self.fallback_backend is None:
                raise
            backend_name = "rule-fallback"
            self.output(f"模型输出不可用，改用规则后端：{exc}")
            advice = validate_advice(self.fallback_backend.advise(context, evidence), context, evidence)
        self.output("\n" + render_console(advice, context, evidence) + "\n")
        self._record_safely(raw_state, context, evidence, advice, backend_name)
        self.last_decision_id = context.decision_id
        return advice

    def _record_safely(self, raw_state, context, evidence, advice, backend_name: str) -> None:
        try:
            self.recorder.record(raw_state, context, evidence, advice, backend_name)
        except OSError as exc:
            self.output(f"警告：无法写入会话记录，建议流程继续运行：{exc}")
