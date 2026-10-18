from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from .domain import Advice, DecisionContext, Evidence


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
        return Advice(
            decision_id=str(payload.get("decision_id", context.decision_id)),
            candidate_id=str(payload["candidate_id"]),
            reason=str(payload.get("reason", "")),
            confidence=float(payload.get("confidence", 0.5)),
            alternative_ids=[str(value) for value in payload.get("alternative_ids", [])],
            evidence_ids=[str(value) for value in payload.get("evidence_ids", [])],
            warnings=[str(value) for value in payload.get("warnings", [])],
        )


SYSTEM_PROMPT = """你是《杀戮尖塔2》的只读决策建议器。
只能从 candidates 中选择一个 candidate_id，不得创造动作、索引或目标。
只依据提供的游戏状态与证据；证据不足时降低 confidence，不要假装确定。
一次只建议当前下一步。不要输出思维过程，只给简短、可验证的理由。
必须仅输出一个 JSON 对象，不要使用 Markdown。"""


def build_prompt(context: DecisionContext, evidence: list[Evidence]) -> str:
    schema = {
        "decision_id": context.decision_id,
        "candidate_id": "必须来自 candidates",
        "reason": "不超过120字",
        "confidence": 0.0,
        "alternative_ids": [],
        "evidence_ids": [],
        "warnings": [],
    }
    return (
        "当前局面：\n"
        + json.dumps(context.to_dict(), ensure_ascii=False, separators=(",", ":"))
        + "\n\n检索证据：\n"
        + json.dumps([item.to_dict() for item in evidence], ensure_ascii=False, separators=(",", ":"))
        + "\n\n输出格式：\n"
        + json.dumps(schema, ensure_ascii=False)
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
