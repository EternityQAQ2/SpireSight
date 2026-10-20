from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from .domain import Advice, DecisionContext, Evidence
from .logging_config import get_logger


logger = get_logger("model")


class AdvisorBackend(Protocol):
    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice: ...


def select_device(torch, requested: str = "auto") -> str:
    """Resolve an inference device without requiring PyTorch for the rule backend."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"不支持的设备：{requested}")
    if requested == "cpu":
        return "cpu"
    mps = getattr(torch.backends, "mps", None)
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": mps is not None and mps.is_available(),
    }
    if requested == "auto":
        return next((device for device in ("cuda", "mps") if available[device]), "cpu")
    if not available[requested]:
        raise RuntimeError(f"设备 {requested} 不可用；请检查 PyTorch 与硬件，或使用 --device cpu。")
    return requested


class RuleAdvisor:
    """Deterministic fallback used for pipeline validation and model failures."""

    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice:
        candidate = choose_rule_candidate(context)
        alternatives = [item.candidate_id for item in context.candidates if item != candidate][:2]
        return Advice(
            decision_id=context.decision_id,
            candidate_id=candidate.candidate_id,
            reason=rule_reason(context, candidate.label),
            confidence=0.45,
            alternative_ids=alternatives,
            evidence_ids=[item.evidence_id for item in evidence[:2]],
            warnings=["规则后端仅用于验证流程，不代表高水平策略。"],
            plan_candidate_ids=[candidate.candidate_id],
        )


class TransformersAdvisor:
    def __init__(
        self,
        model_name: str,
        cache_root: Path | None = None,
        max_new_tokens: int = 384,
        device: str = "auto",
    ) -> None:
        if cache_root:
            cache_root = cache_root.expanduser()
            os.environ.setdefault("HF_HOME", str(cache_root))
            os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
            os.environ.setdefault("HF_XET_CACHE", str(cache_root / "xet"))

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = select_device(torch, device)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        cache_dir = os.environ.get("HF_HUB_CACHE")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=cache_dir,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice:
        prompt = build_prompt(context, evidence)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            rendered = SYSTEM_PROMPT + "\n\n" + prompt
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        payload = extract_json_object(text)
        return advice_from_payload(payload, context)


class OpenAIAdvisor:
    """Remote OpenAI or OpenAI-compatible chat completion backend."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 384,
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
        self.model_name = model_name
        self.max_tokens = max_tokens

    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(context, evidence)},
        ]
        try:
            content, reasoning, finish_reason = self._complete(
                messages, self.max_tokens, context.decision_id
            )
            if not content and finish_reason == "length" and self.max_tokens < 4096:
                retry_tokens = min(max(self.max_tokens * 2, 2048), 4096)
                logger.warning(
                    "remote_retry_empty_content model=%s decision_id=%s max_tokens=%s",
                    self.model_name, context.decision_id, retry_tokens,
                )
                content, reasoning, finish_reason = self._complete(
                    messages, retry_tokens, context.decision_id
                )
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
        self, messages: list[dict[str, str]], max_tokens: int, decision_id: str,
    ) -> tuple[str, str, str | None]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
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
        return content, reasoning, finish_reason


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


SYSTEM_PROMPT = """你是《杀戮尖塔2》的决策建议器。输入始终包含当前游戏状态、合法候选动作和检索证据。

约束：
1. candidate_id、alternative_ids 和 plan_candidate_ids 只能使用 candidates 中原样提供的 ID，不得创造动作、索引或目标。
2. candidate_id 必须等于 plan_candidate_ids 的第一项。plan_candidate_ids 是从当前局面开始依次执行的短计划，最多 8 项。
3. 战斗中，在当前手牌与能量足以确定后续动作时，一次规划多张牌以及必要的结束回合。可能抽牌、随机生成牌、改变手牌、进入选择界面或目标可能提前死亡时，在该不确定动作处结束计划。
4. 商店中可按当前金币规划多个购买，但必须考虑购买后的剩余金币；奖励列表可连续领取确定性奖励，遇到打开选牌界面的奖励时结束计划。
5. 地图、事件、选牌、休息点等结果会改变后续选择的场景只规划当前一步。
6. 只依据提供的状态与证据；证据不足时降低 confidence，不得虚构卡牌、遗物或敌人效果。
7. reason 不超过 120 个汉字，只解释关键取舍，不输出隐藏思维过程。

只输出一个 JSON 对象，不要使用 Markdown。固定字段如下：
{"decision_id":"输入中的decision_id","candidate_id":"当前动作ID","reason":"简短理由","confidence":0.0,"alternative_ids":[],"evidence_ids":[],"warnings":[],"plan_candidate_ids":["当前动作ID","后续动作ID"]}
所有字段必须存在；没有备选、证据、警告或后续动作时使用空数组，plan_candidate_ids 至少包含 candidate_id。"""


def build_prompt(context: DecisionContext, evidence: list[Evidence]) -> str:
    return (
        "当前局面：\n"
        + json.dumps(context.to_dict(), ensure_ascii=False, separators=(",", ":"))
        + "\n\n检索证据：\n"
        + json.dumps([item.to_dict() for item in evidence], ensure_ascii=False, separators=(",", ":"))
    )


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(f"模型未返回 JSON: {stripped[:300]}")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("模型返回值不是 JSON 对象")
    return payload


def advice_from_payload(payload: dict, context: DecisionContext) -> Advice:
    return Advice(
        decision_id=str(payload.get("decision_id", context.decision_id)),
        candidate_id=str(payload["candidate_id"]),
        reason=str(payload.get("reason", "")),
        confidence=float(payload.get("confidence", 0.5)),
        alternative_ids=[str(value) for value in payload.get("alternative_ids", [])],
        evidence_ids=[str(value) for value in payload.get("evidence_ids", [])],
        warnings=[str(value) for value in payload.get("warnings", [])],
        plan_candidate_ids=[str(value) for value in payload.get("plan_candidate_ids", [])],
    )


def choose_rule_candidate(context: DecisionContext):
    if context.decision_type == "combat_next_action":
        player = context.player
        incoming = incoming_damage(context.situation)
        uncovered = max(0, incoming - int(player.get("block", 0)))
        if uncovered:
            defense = next(
                (candidate for candidate in context.candidates if "防御" in candidate.label or "格挡" in candidate.label),
                None,
            )
            if defense:
                return defense
        attack = next((candidate for candidate in context.candidates if candidate.action_type == "play_card"), None)
        if attack:
            return attack
    non_exit = next(
        (
            candidate
            for candidate in context.candidates
            if candidate.action_type not in {"proceed", "end_turn", "skip_card_reward", "skip_relic_selection"}
        ),
        None,
    )
    return non_exit or context.candidates[0]


def incoming_damage(situation: dict) -> int:
    total = 0
    for enemy in situation.get("battle", {}).get("enemies", []):
        for intent in enemy.get("intents", []):
            if intent.get("type") == "Attack":
                numbers = re.findall(r"\d+", str(intent.get("label", "")))
                if numbers:
                    total += int(numbers[0])
    return total


def rule_reason(context: DecisionContext, label: str) -> str:
    if context.decision_type == "combat_next_action":
        damage = incoming_damage(context.situation)
        return f"当前敌方预计造成 {damage} 点伤害；规则后端选择：{label}。"
    return f"规则后端从当前合法候选中选择：{label}。"
