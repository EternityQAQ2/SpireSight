"""OpenAI-compatible transport and response normalization; no game-rule implementation."""
from __future__ import annotations

import json
from sts2_advisor.contracts import Advice, DecisionContext, Evidence
from sts2_advisor.decision.prompts import build_system_prompt, build_prompt
from sts2_advisor.decision.context import CodexCache
from sts2_advisor.decision.tools import DecisionTools, TOOL_SCHEMAS
from sts2_advisor.decision.parsing import extract_json_object, advice_from_payload
from sts2_advisor.recording.logging import get_logger
from sts2_advisor.recording.trajectory import record_event, recorded_call

logger = get_logger("model")


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
        adaptive_thinking: bool = False,
        reasoning_effort: str | None = None,
        function_calling: bool = True,
        precompute_attacks: bool = False,
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
        self.adaptive_thinking = adaptive_thinking
        self.reasoning_effort = reasoning_effort or ('low' if base_url and 'deepseek.com' in base_url.lower() else None)
        if self.reasoning_effort not in {None, 'low', 'high', 'max'}:
            raise ValueError('reasoning_effort 必须为 low、high 或 max')
        self.function_calling = function_calling
        self.precompute_attacks = precompute_attacks

    def advise_with_feedback(self, context: DecisionContext, evidence: list[Evidence], feedback: str) -> Advice:
        return self.advise(context, evidence, feedback=feedback)

    def advise(self, context: DecisionContext, evidence: list[Evidence], *, feedback: str = '') -> Advice:
        enemies = (context.situation.get('battle') or {}).get('enemies', [])
        difficult = (context.decision_type == 'combat_next_action' and
                     (context.state_type in {'elite', 'boss'} or
                      sum(1 for enemy in enemies if (enemy.get('hp') or 0) > 0) > 1))
        thinking = ('enabled' if difficult else 'disabled') if self.adaptive_thinking else self.thinking
        max_tokens = self.max_tokens
        inline = self.precompute_attacks and context.decision_type == 'combat_next_action'
        prompt_options = {'precompute_attacks': True} if inline else {}
        prompt = build_prompt(context, evidence, codex_cache=self.codex_cache, **prompt_options)
        if feedback:
            prompt += '\n上次计划无法执行：' + feedback + '。请重新选择合法计划。'
        messages = [
            {"role": "system", "content": build_system_prompt(**prompt_options)},
            {"role": "user", "content": prompt},
        ]
        dispatcher = DecisionTools(context)
        tools_requested = self.function_calling and context.decision_type == 'combat_next_action' and not inline
        availability = dispatcher.attack_tool_availability() if tools_requested else None
        tools_enabled = tools_requested and availability['available']
        if tools_requested:
            record_event('model.tool_availability', decision_id=context.decision_id, **availability)
        if tools_enabled:
            messages[0]["content"] += (
                "\n可调用 calculate_attack 计算单张攻击牌，只指定局面和候选动作；数值和目标修正由工具读取。"
                "手牌显示伤害已包含力量，不要重复加算。仅按工具返回的当前单步结果推断，未知不等于零。"
                "由你结合工具结果选择行动；最后返回要求的建议 JSON。"
            )
        elif tools_requested:
            messages[0]['content'] += (
                '\n本局面的 calculate_attack 无法核对任何当前合法攻击，本次不提供该工具。'
                '这表示计算器不支持，未知不等于零，也不代表这些攻击不能打。'
                '请依据当前观测决策，不宣称未知数值已经工具验证。限制：'
                + '；'.join(availability['limitations']))
        logger.info(
            "remote_request model=%s decision_id=%s prompt_chars=%s evidence_count=%s "
            "json_mode=%s thinking=%s max_tokens=%s reasoning_effort=%s",
            self.model_name, context.decision_id, len(prompt), len(evidence),
            self.json_mode, thinking, max_tokens, self.reasoning_effort if thinking == 'enabled' else None,
        )
        try:
            calls_used = 0
            for round_index in range(2):
                options = {"tools": TOOL_SCHEMAS, "tool_choice": "auto" if round_index == 0 and calls_used < 8 else "none"} if tools_enabled else {}
                options['thinking'] = thinking
                content, reasoning, finish_reason, assistant = self._complete(
                    messages, max_tokens, context.decision_id, **options)
                tool_calls = assistant.get("tool_calls") or []
                if not tool_calls:
                    break
                if not tools_enabled or round_index == 1 or len(tool_calls) > 8:
                    raise ValueError("模型超出工具调用轮数或批量上限")
                messages.append(assistant)
                for call in tool_calls:
                    function = call.get("function") or {}
                    result = ({"ok": False, "error": "工具预算已耗尽，请返回最终建议"} if calls_used >= 8 else
                              dispatcher.call(function.get("name", ""), function.get("arguments", "")))
                    calls_used += 1
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
        return advice_from_payload(extract_json_object(content), context)

    def _complete(
        self, messages: list[dict], max_tokens: int, decision_id: str,
        *, tools: list[dict] | None = None, tool_choice: str = "auto",
        thinking: str | None = None,
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
        if thinking in {"enabled", "disabled"}:
            request["extra_body"] = {"thinking": {"type": thinking}}
        if thinking == 'enabled' and self.reasoning_effort:
            request['reasoning_effort'] = self.reasoning_effort
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
