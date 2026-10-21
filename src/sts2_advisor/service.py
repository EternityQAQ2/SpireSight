from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from typing import Callable

from .action import ActionExecutor
from .api import Sts2ApiError, Sts2RestClient
from .domain import Advice
from .model import AdvisorBackend, RuleAdvisor
from .logging_config import get_logger
from .normalize import build_decision_context
from .planning import ActivePlan
from .recording import SessionRecorder
from .record import record_event, trace_scope
from .render import render_console
from .retrieval import Retriever
from .validation import AdviceValidationError, validate_advice


logger = get_logger("service")


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
        auto_act: bool = False,
        output: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.backend = backend
        self.backend_name = backend_name
        self.fallback_backend = fallback_backend
        self.recorder = recorder
        self.trajectory = getattr(recorder, "trajectory", None)
        self.top_k = top_k
        self.action_executor = ActionExecutor(client, output=output) if auto_act else None
        self.output = output
        self.last_decision_id: str | None = None
        self.active_plan: ActivePlan | None = None
        self._previous_step_id: str | None = None
        self._plan_origin_step_id: str | None = None
        self._current_step_id: str | None = None

    def _read_state(self) -> dict:
        with trace_scope(self.trajectory):
            try:
                raw_state = self.client.get_singleplayer_state()
            except Sts2ApiError as exc:
                record_event("environment.error", message=str(exc), after_step_id=self._previous_step_id)
                raise
            record_event("environment.observation", raw_state=raw_state,
                         after_step_id=self._previous_step_id)
            return raw_state

    def run_once(self, *, force: bool = False) -> bool:
        raw_state = self._read_state()
        context = build_decision_context(raw_state)
        if context is None:
            state_type = raw_state.get("state_type", "unknown")
            self.output(f"当前状态 {state_type!r} 没有可建议的决策，保持等待。")
            return False
        if not force and context.decision_id == self.last_decision_id:
            return False

        self._process(raw_state, context)
        return True

    def watch(self, poll_interval: float, stable_reads: int = 2) -> None:
        mode = "自动执行推荐动作" if self.action_executor else "只读建议"
        self.output(f"开始监听 STS2MCP（{mode}）。按 Ctrl+C 停止。")
        observed_id: str | None = None
        observed_count = 0
        error_message: str | None = None
        while True:
            try:
                raw_state = self._read_state()
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
        scope = self.trajectory.step(raw_state, context) if self.trajectory else nullcontext(None)
        with scope as step_id:
            self._current_step_id = step_id
            self._previous_step_id = step_id
            return self._process_step(raw_state, context)

    def _process_step(self, raw_state: dict, context) -> Advice:
        if self.active_plan is not None:
            cached = self.active_plan.advance(context)
            if cached is not None:
                evidence = self.active_plan.evidence
                advice = validate_advice(cached, context, evidence)
                record_event("plan.reused", origin_step_id=self._plan_origin_step_id,
                             advice=advice.to_dict(), evidence=[item.to_dict() for item in evidence])
                logger.info(
                    "plan_cache_hit decision_id=%s remaining_steps=%s",
                    context.decision_id, len(advice.plan_candidate_ids),
                )
                self.output("复用上一轮短计划，本次未请求模型。")
                return self._deliver(
                    raw_state, context, evidence, advice, "plan-cache"
                )
            logger.info("plan_cache_invalidated decision_id=%s", context.decision_id)
            self.active_plan = None
            record_event("plan.invalidated", origin_step_id=self._plan_origin_step_id)
            self._plan_origin_step_id = None

        record_event("retrieval.request", context=context.to_dict(), top_k=self.top_k,
                     retriever=type(self.retriever).__name__)
        evidence = self.retriever.retrieve(context, self.top_k)
        record_event("retrieval.result", evidence=[item.to_dict() for item in evidence])
        backend_name = self.backend_name
        try:
            proposed = self.backend.advise(context, evidence)
            record_event("decision.proposal", backend=backend_name, advice=proposed.to_dict())
            advice = validate_advice(proposed, context, evidence)
        except (AdviceValidationError, ValueError, RuntimeError) as exc:
            record_event("decision.validation_error", error_type=type(exc).__name__, message=str(exc))
            logger.exception(
                "advisor_fallback backend=%s decision_id=%s error_type=%s",
                self.backend_name, context.decision_id, type(exc).__name__,
            )
            if self.fallback_backend is None:
                raise
            backend_name = "rule-fallback"
            record_event("decision.fallback", from_backend=self.backend_name, to_backend=backend_name,
                         error_type=type(exc).__name__, message=str(exc))
            self.output(f"模型输出不可用，改用规则后端：{exc}")
            advice = validate_advice(self.fallback_backend.advise(context, evidence), context, evidence)
        self.active_plan = ActivePlan.from_advice(advice, context, evidence)
        if self.active_plan is not None:
            self._plan_origin_step_id = self._current_step_id
            record_event("plan.created", advice=advice.to_dict())
        return self._deliver(raw_state, context, evidence, advice, backend_name)

    def _deliver(self, raw_state, context, evidence, advice, backend_name: str) -> Advice:
        rendered = render_console(advice, context, evidence)
        record_event("decision.output", backend=backend_name, advice=advice.to_dict(), rendered=rendered)
        self.output("\n" + rendered + "\n")
        self._record_safely(raw_state, context, evidence, advice, backend_name)
        if self.action_executor:
            try:
                self.action_executor.execute(advice, context)
            finally:
                self.last_decision_id = self.action_executor.last_decision_id
        else:
            record_event("action.skipped", reason="read_only")
            self.last_decision_id = context.decision_id
        return advice

    def _record_safely(self, raw_state, context, evidence, advice, backend_name: str) -> None:
        try:
            self.recorder.record(raw_state, context, evidence, advice, backend_name)
        except OSError as exc:
            self.output(f"警告：无法写入会话记录，建议流程继续运行：{exc}")
