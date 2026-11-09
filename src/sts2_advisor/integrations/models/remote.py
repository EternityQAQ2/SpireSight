"""OpenAI-compatible transport and response normalization; no game-rule implementation."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from sts2_advisor.contracts import Advice, DecisionContext, Evidence
from sts2_advisor.decision.prompts import SYSTEM_PROMPT, build_prompt
from sts2_advisor.decision.context import CodexCache
from sts2_advisor.decision.tools import DecisionTools, TOOL_SCHEMAS
from sts2_advisor.game.combat.calculator import calculate
from sts2_advisor.decision.parsing import extract_json_object, advice_from_payload
from sts2_advisor.recording.logging import get_logger
from sts2_advisor.recording.trajectory import record_event, recorded_call

logger = get_logger("model")
KILL_CLAIM = re.compile(r"(?<!不)(?<!未)(?<!无)(?<!法)(?:秒杀|斩杀|可击杀)")


class OpenAIAdvisor:
    """Remote OpenAI or OpenAI-compatible chat completion backend."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
        json_mode: bool = True,
        thinking: str | None = None,
        function_calling: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY，请在项目根目录 .env 中配置。")
        if not model_name:
            raise ValueError("缺少 OPENAI_MODEL，请在项目根目录 .env 中配置。")

        from openai import OpenAI

        kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.codex_cache = CodexCache()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.thinking = thinking
        self.function_calling = function_calling

    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice:
        prompt = build_prompt(context, evidence, codex_cache=self.codex_cache)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tools_enabled = self.function_calling and context.decision_type == "combat_next_action"
        if tools_enabled:
            messages[0]["content"] += (
                "\n可调用 calculate_attack 核对单张攻击牌，传入预期目标剩余生命；目标修正由工具读取。"
                "手牌显示伤害已包含力量，不要重复加算。仅按工具返回的当前单步结果推断，未知不等于零。"
                "出现不一致时根据工具结果重新决策一次；最后返回要求的建议 JSON。"
            )
        logger.info(
            "remote_request model=%s decision_id=%s prompt_chars=%s evidence_count=%s "
            "json_mode=%s thinking=%s max_tokens=%s",
            self.model_name, context.decision_id, len(prompt), len(evidence),
            self.json_mode, self.thinking, self.max_tokens,
        )
        try:
            dispatcher = DecisionTools(context)
            calls_used = 0
            tool_results = []
            for round_index in range(2):
                options = {"tools": TOOL_SCHEMAS, "tool_choice": "auto" if round_index == 0 and calls_used < 8 else "none"} if tools_enabled else {}
                content, reasoning, finish_reason, assistant = self._complete(
                    messages, self.max_tokens, context.decision_id, **options)
                tool_calls = assistant.get("tool_calls") or []
                if not tool_calls:
                    if not content and finish_reason == "length" and self.max_tokens < 4096:
                        content, reasoning, finish_reason, assistant = self._complete(
                            messages, min(max(self.max_tokens * 2, 2048), 4096),
                            context.decision_id, **options)
                        tool_calls = assistant.get("tool_calls") or []
                    if not tool_calls:
                        if tools_enabled and round_index == 0 and content:
                            provisional = advice_from_payload(extract_json_object(content), context)
                            if KILL_CLAIM.search(provisional.reason):
                                candidate = next((c for c in context.candidates if c.candidate_id == provisional.candidate_id), None)
                                enemy = next((e for e in (context.situation.get("battle") or {}).get("enemies", [])
                                              if candidate and e.get("entity_id") == candidate.params.get("target")), None)
                                if enemy:
                                    check = calculate(context, provisional.candidate_id, 0)
                                    record_event("decision.attack_check", decision_id=context.decision_id, result=check)
                                    tool_results.append(check)
                                    if check.get("status") == "calculated" and not check.get("matches"):
                                        messages.append({"role": "assistant", "content": content})
                                        messages.append({"role": "user", "content": "你的击杀判断与计算结果矛盾。请重新选择一次并返回最终 JSON：" + json.dumps(check, ensure_ascii=False)})
                                        continue
                        break
                if not tools_enabled or round_index == 1 or len(tool_calls) > 8:
                    raise ValueError("模型超出工具调用轮数或批量上限")
                messages.append(assistant)
                for call in tool_calls:
                    function = call.get("function") or {}
                    result = ({"ok": False, "error": "工具预算已耗尽，请返回最终建议"} if calls_used >= 8 else
                              dispatcher.call(function.get("name", ""), function.get("arguments", "")))
                    calls_used += 1
                    tool_results.append(result)
                    record_event("model.tool_result", decision_id=context.decision_id,
                                 tool_call_id=call["id"], name=function.get("name"), arguments=function.get("arguments"), result=result)
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False)})
        except Exception as exc:
            logger.exception(
                "remote_request_failed model=%s decision_id=%s error_type=%s",
                self.model_name, context.decision_id, type(exc).__name__,
            )
            raise RuntimeError(f"远程模型请求失败：{exc}") from exc
        if not content:
            logger.warning(
                "remote_empty_content model=%s decision_id=%s reasoning_preview=%r",
                self.model_name, context.decision_id, reasoning[:500],
            )
            detail = ""
            if reasoning:
                detail = "；服务只返回了 reasoning_content，可能是输出上限过小或模型接口不兼容"
            raise ValueError(f"远程模型返回了空内容{detail}，请查看运行日志")
        advice = advice_from_payload(extract_json_object(content), context)
        if tool_results and all("status" not in r for r in tool_results):
            chosen = dispatcher.evaluate(advice.plan_candidate_ids or [advice.candidate_id])
            claim = re.search(r"秒杀|斩杀|可击杀|仅\d+点伤害|只受\d+点伤害", advice.reason)
            contradicted = bool(chosen.get("exact") and chosen.get("projection") and
                                chosen["projection"]["kills"] == 0 and
                                re.search(r"秒杀|斩杀|可击杀", advice.reason))
            if claim and (not chosen.get("exact") or contradicted):
                advice = replace(advice, reason="试算结果不支持原定斩杀判断；先执行一步并重新观测。" if contradicted else
                                 "当前机制无法完整试算；先执行一步并重新观测。",
                                 confidence=min(advice.confidence, 0.4),
                                 warnings=advice.warnings + ["计划试算不支持或与斩杀断言矛盾，原理由中的确定性判断已撤回"])
        selected = next((r for r in tool_results if r.get("candidate_id") == advice.candidate_id and r.get("status") == "calculated"), None)
        if tools_enabled and KILL_CLAIM.search(advice.reason) and not selected:
            candidate = next((c for c in context.candidates if c.candidate_id == advice.candidate_id), None)
            enemy = next((e for e in (context.situation.get("battle") or {}).get("enemies", [])
                          if candidate and e.get("entity_id") == candidate.params.get("target")), None)
            if enemy:
                selected = calculate(context, advice.candidate_id, 0)
                record_event("decision.attack_check", decision_id=context.decision_id, result=selected)
        if selected and selected.get("status") == "calculated" and not selected["target_defeated"] and KILL_CLAIM.search(advice.reason):
            advice = replace(advice, reason="计算结果显示目标仍有生命，先执行一步并重新观测。",
                             confidence=min(advice.confidence, 0.4),
                             warnings=advice.warnings + ["攻击计算结果与击杀断言矛盾"])
        return advice

    def _complete(
        self, messages: list[dict], max_tokens: int, decision_id: str,
        *, tools: list[dict] | None = None, tool_choice: str = "auto",
    ) -> tuple[str, str, str | None, dict]:
        request = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if tools:
            request.update(tools=tools, tool_choice=tool_choice)
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}
        if self.thinking in {"enabled", "disabled"}:
            request["extra_body"] = {"thinking": {"type": self.thinking}}
        with recorded_call("model", backend="openai", request=request) as call_id:
            response = self.client.chat.completions.create(**request)
            record_event("model.raw_response", call_id=call_id,
                         response=json.loads(serialize_openai_response(response)))
            choice = response.choices[0]
            message = choice.message
            content = message_text(getattr(message, "content", None))
            reasoning = message_text(
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
            )
            usage = getattr(response, "usage", None)
            usage_data = usage.model_dump() if hasattr(usage, "model_dump") else usage
            cache_hit_tokens, cache_miss_tokens = cache_token_counts(usage_data)
            finish_reason = getattr(choice, "finish_reason", None)
            raw_response = serialize_openai_response(response)
            record_event("model.response", call_id=call_id, raw_response=json.loads(raw_response),
                         content=content, reasoning=reasoning, finish_reason=finish_reason,
                         usage=usage_data, cache_hit_tokens=cache_hit_tokens,
                         cache_miss_tokens=cache_miss_tokens)
            logger.info(
                "remote_response model=%s decision_id=%s finish_reason=%s usage=%s "
                "max_tokens=%s cache_hit_tokens=%s cache_miss_tokens=%s "
                "content_chars=%s reasoning_chars=%s raw_json=%s",
                self.model_name,
                decision_id,
                finish_reason,
                usage_data,
                max_tokens,
                cache_hit_tokens,
                cache_miss_tokens,
                len(content),
                len(reasoning),
                raw_response,
            )
            logger.info(
                "remote_text decision_id=%s content=%s reasoning=%s",
                decision_id,
                json.dumps(content, ensure_ascii=False),
                json.dumps(reasoning, ensure_ascii=False),
            )
            assistant = {"role": "assistant", "content": content or None}
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                assistant["tool_calls"] = [
                    {"id": call.id, "type": "function", "function": {
                        "name": call.function.name, "arguments": call.function.arguments}}
                    for call in tool_calls]
                if reasoning:
                    assistant["reasoning_content"] = reasoning
            return content, reasoning, finish_reason, assistant


def cache_token_counts(usage) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    details = usage.get("prompt_tokens_details") or {}
    hit = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
    miss = usage.get("prompt_cache_miss_tokens")
    return hit, miss


def serialize_openai_response(response) -> str:
    """Serialize SDK responses and lightweight OpenAI-compatible objects."""
    try:
        if hasattr(response, "model_dump_json"):
            return response.model_dump_json()
        if hasattr(response, "model_dump"):
            value = response.model_dump()
        elif isinstance(response, dict):
            value = response
        else:
            value = vars(response)
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(str(response), ensure_ascii=False)


def message_text(value) -> str:
    """Normalize text returned by OpenAI and compatible providers."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or getattr(item, "content", "")))
        return "".join(parts).strip()
    return str(value).strip()
