from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from .domain import Advice, DecisionContext, Evidence
from .logging_config import get_logger
from .record import record_event, recorded_call, recording_enabled


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
        self.model_name = model_name
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
        with recorded_call("model", backend="transformers", request={
            "model": self.model_name, "messages": messages, "device": self.device,
            "max_new_tokens": self.max_new_tokens, "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }) as call_id:
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
            if recording_enabled():
                generation_config = getattr(self.model, "generation_config", None)
                record_event("model.input", call_id=call_id, rendered_prompt=rendered,
                             inputs={key: value.tolist() for key, value in inputs.items()},
                             generation_config=generation_config.to_dict() if generation_config else None)
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
            record_event("model.response", call_id=call_id, content=text,
                         generated_token_ids=generated.tolist() if recording_enabled() else None)
            payload = extract_json_object(text)
            record_event("model.parsed", call_id=call_id, payload=payload)
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
        max_tokens: int = 1024,
        json_mode: bool = True,
        thinking: str | None = None,
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
        self.json_mode = json_mode
        self.thinking = thinking

    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice:
        prompt = build_prompt(context, evidence)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        logger.info(
            "remote_request model=%s decision_id=%s prompt_chars=%s evidence_count=%s "
            "json_mode=%s thinking=%s max_tokens=%s",
            self.model_name, context.decision_id, len(prompt), len(evidence),
            self.json_mode, self.thinking, self.max_tokens,
        )
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
        request = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
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


SYSTEM_PROMPT = """你是《杀戮尖塔2》决策建议器。输入含状态、合法动作 candidates 和证据 evidence。
stable.deck 是卡牌图鉴：手牌与牌堆只给出图鉴键 k（末尾 + 表示已升级）及数量，卡牌效果一律查图鉴；图鉴之外的卡（商店、奖励等）在其界面数据中自带说明。
规则：
1. 所有动作 ID 必须逐字来自 candidates；candidate_id 是 plan_candidate_ids 首项，计划共 1 至 8 步。
2. 战斗中能量与手牌允许时应规划 2 至 8 步连续动作，并以 combat:end_turn 收尾；下一步会抽牌、生成或消耗手牌、结果随机或目标可能提前死亡时必须停止，之后的局面会重新决策。
3. 商店可按余额连续购买；奖励可连续领取确定项，打开选牌时停止。
4. 地图、事件、选牌、休息点只规划当前一步。选择类界面（card_select/hand_select）中已选卡不会出现在 candidates；选择已满足要求时直接输出 confirm 候选，candidate_id 与 alternative_ids 都不得引用已选卡。
5. 只依据输入；证据不足则降低 confidence，不虚构效果。reason 不超过 80 个汉字，不输出思维过程。
返回 JSON 对象，必须且只能包含：
{"decision_id":"输入值","candidate_id":"动作ID","reason":"理由","confidence":0.0,"alternative_ids":[],"evidence_ids":[],"warnings":[],"plan_candidate_ids":["动作ID"]}
confidence 范围 0 至 1；无值的数组用 []。"""


def build_prompt(context: DecisionContext, evidence: list[Evidence]) -> str:
    payload = model_context_payload(context)
    payload["evidence"] = [
        {"id": item.evidence_id, "title": item.source_title, "text": item.text}
        for item in evidence
    ]
    payload["decision_id"] = context.decision_id
    return "输入 JSON：" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def model_context_payload(context: DecisionContext) -> dict:
    """Build a smaller prompt with stable fields first for provider prefix caching."""
    player = context.player
    relics = player.get("relics", [])
    hand_cards = player.get("hand", [])
    pile_cards = {
        pile: player.get(pile, [])
        for pile in ("draw_pile", "discard_pile", "exhaust_pile")
    }
    name_to_id = card_name_to_id_map(hand_cards, *pile_cards.values())
    stable = {
        "character": player.get("character"),
        "max_hp": player.get("max_hp"),
        "max_energy": player.get("max_energy"),
        "relics": [
            compact_fields(item, ("id", "name", "description")) for item in relics
        ],
    }
    deck = deck_codex(hand_cards, pile_cards, name_to_id)
    if deck:
        stable["deck"] = deck
    dynamic_player = compact_fields(
        player, ("hp", "block", "energy", "gold", "status", "potions", "max_potion_slots")
    )
    dynamic_player["relic_counters"] = [
        {"id": item.get("id"), "counter": item.get("counter")}
        for item in relics if item.get("counter") is not None
    ]
    dynamic_player["hand"] = [
        compact_hand_card(card, name_to_id) for card in hand_cards
    ]
    for pile, cards in pile_cards.items():
        dynamic_player[pile] = pile_counts(cards, name_to_id)
    return {
        "stable": stable,
        "run": context.run,
        "state_type": context.state_type,
        "decision_type": context.decision_type,
        "player": dynamic_player,
        "situation": context.situation,
        "candidates": [
            {"id": item.candidate_id, "label": item.label} for item in context.candidates
        ],
    }


def card_name_to_id_map(*card_groups: list[dict]) -> dict[str, str]:
    """Map card names to ids so pile entries without id share the hand card's key."""
    mapping: dict[str, str] = {}
    for cards in card_groups:
        for card in cards:
            if card.get("id") and card.get("name"):
                mapping.setdefault(str(card["name"]), str(card["id"]))
    return mapping


def card_key(card: dict, name_to_id: dict[str, str] | None = None) -> str:
    card_id = card.get("id")
    if not card_id:
        card_id = (name_to_id or {}).get(str(card.get("name") or "")) or card.get("name")
    if not card_id:
        return ""
    return str(card_id) + ("+" if card.get("is_upgraded") else "")


def deck_codex(
    hand_cards: list[dict], pile_cards: dict[str, list[dict]], name_to_id: dict[str, str],
) -> list[dict]:
    """One entry per distinct card across hand and piles, sorted for prefix caching.

    图鉴只在牌组集合变化（拿卡/移除/生成新卡）时才更新，战斗内的出牌与抽牌
    只移动卡牌，键集合不变，因此它位于静态前缀内，可被服务端前缀缓存命中。
    """
    codex: dict[str, dict] = {}
    sources = [hand_cards, *pile_cards.values()]
    for cards in sources:
        for card in cards:
            key = card_key(card, name_to_id)
            if not key or key in codex:
                continue
            entry = {
                "k": key,
                "name": card.get("name"),
                "type": card.get("type"),
                "cost": card.get("cost"),
                "description": card.get("description", ""),
            }
            codex[key] = {field: value for field, value in entry.items() if value is not None}
    return [codex[key] for key in sorted(codex)]


def compact_hand_card(card: dict, name_to_id: dict[str, str] | None = None) -> dict:
    entry = {
        "k": card_key(card, name_to_id),
        "cost": card.get("cost"),
        "can": card.get("can_play", True),
    }
    if not card.get("can_play", True) and card.get("unplayable_reason"):
        entry["why"] = card.get("unplayable_reason")
    return entry


def pile_counts(cards: list[dict], name_to_id: dict[str, str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in cards:
        key = card_key(card, name_to_id)
        if key:
            counts[key] = counts.get(key, 0) + int(card.get("count", 1))
    return counts


def compact_fields(value: dict, fields: tuple[str, ...]) -> dict:
    return {field: value[field] for field in fields if field in value}


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
    # 选择类界面出现 confirm 候选说明当前选择已可提交；回退必须直接确认，
    # 否则会盲目再选一张卡，把多选界面一路推向全选（附魔事件曾因此卡死）。
    confirm = next(
        (
            candidate
            for candidate in context.candidates
            if candidate.action_type in {"confirm_selection", "combat_confirm_selection"}
        ),
        None,
    )
    if confirm is not None:
        return confirm
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
