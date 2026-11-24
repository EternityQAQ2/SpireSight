"""Optional local Transformers backend; heavyweight dependencies load only on construction."""
from __future__ import annotations

import os
import time
from pathlib import Path
from sts2_advisor.contracts import Advice, DecisionContext, Evidence
from sts2_advisor.developer import dev_block, dev_request, dev_usage
from sts2_advisor.decision.prompts import SYSTEM_PROMPT, build_prompt
from sts2_advisor.decision.context import CodexCache
from sts2_advisor.decision.parsing import extract_json_object, advice_from_payload
from sts2_advisor.recording.trajectory import record_event, recorded_call, recording_enabled


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
        self.codex_cache = CodexCache()
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
        prompt = build_prompt(context, evidence, codex_cache=self.codex_cache)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        dev_block(
            "提示词", f"完整提示词（transformers · {self.model_name}）",
            {"system": SYSTEM_PROMPT, "user": prompt},
            backend="transformers", model=self.model_name,
            max_new_tokens=self.max_new_tokens, device=self.device,
            prompt_chars=len(prompt),
            chat_template="apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
        )
        dev_request("transformers", self.model_name, attempt=1,
                    max_new_tokens=self.max_new_tokens, device=self.device)
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
            input_tokens = int(inputs["input_ids"].numel())
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            started = time.monotonic()
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            duration_ms = (time.monotonic() - started) * 1000
            generated = output[0][inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            record_event("model.response", call_id=call_id, content=text,
                         generated_token_ids=generated.tolist() if recording_enabled() else None)
            dev_block(
                "模型输出", "transformers 生成结果",
                {"content": text} if text else {},
                call_id=call_id, finish_reason="eos_or_max_new_tokens",
                duration_ms=round(duration_ms, 1),
                prompt_tokens=input_tokens, generated_tokens=int(generated.numel()),
            )
            dev_usage(
                "模型", {"prompt_tokens": input_tokens,
                         "completion_tokens": int(generated.numel())},
                backend="transformers", model=self.model_name, estimated=True,
                duration_ms=duration_ms, call_id=call_id,
            )
            payload = extract_json_object(text)
            record_event("model.parsed", call_id=call_id, payload=payload)
            return advice_from_payload(payload, context)
