"""Compare two DeepSeek thinking settings on recorded decisions; no game I/O."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from sts2_advisor.cli import load_environment
from sts2_advisor.contracts import Candidate, DecisionContext, Evidence
from sts2_advisor.integrations.models.remote import OpenAIAdvisor


def run_case(row: dict, mode: str, max_tokens: int, prompt_variant: str = 'current', *, budget=None) -> dict:
    started = time.perf_counter()
    raw = dict(row["context"])
    raw["candidates"] = [Candidate(**item) for item in raw["candidates"]]
    context = DecisionContext(**raw)
    evidence = [Evidence(**item) for item in row["evidence"]]
    backend = OpenAIAdvisor(os.environ["OPENAI_MODEL"], api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        thinking='disabled' if mode == 'adaptive' else mode,
        adaptive_thinking=mode == 'adaptive', max_tokens=max_tokens)
    backend.client.max_retries = 0
    create = backend.client.chat.completions.create
    request_count = 0
    events = []
    request_options = []
    initial_input = []

    def limited_create(**kwargs):
        nonlocal request_count
        if request_count >= 2:
            raise RuntimeError("已达到每个局面每种模式最多 2 次请求")
        if budget is not None:
            budget.reserve(kwargs)
        request_count += 1
        if not initial_input:
            # Exact inputs make prompt comparisons reproducible; no API credentials.
            initial_input.extend(dict(message) for message in kwargs['messages'])
        request_options.append({'thinking': (kwargs.get('extra_body') or {}).get('thinking'),
                                'max_tokens': kwargs.get('max_tokens'),
                                'input_chars': sum(len(str(m.get('content') or '')) for m in kwargs['messages'])})
        response = create(**kwargs)
        if budget is not None:
            usage = getattr(response, 'usage', None)
            budget.record_usage(usage.model_dump() if hasattr(usage, 'model_dump') else usage)
        return response

    def collect(event_type, **payload):
        if event_type in {"model.response", "model.tool_result", "decision.attack_check"}:
            events.append({"type": event_type, "data": payload})

    result = {"mode": mode, "prompt_variant": prompt_variant, "decision_id": context.decision_id,
              "evidence_ids": [e.evidence_id for e in evidence]}
    try:
        from sts2_advisor.decision.prompts import build_prompt, build_system_prompt
        if prompt_variant == 'compact':
            from evals.prompt_experiment import build_compact_prompt, build_compact_system_prompt
        def selected_prompt(*args, **kwargs):
            if prompt_variant == 'compact':
                return build_compact_prompt(*args, **kwargs)
            prompt = build_prompt(*args, **kwargs)
            return '输入 JSON：' + prompt.split('输入 JSON：', 1)[1] if prompt_variant == 'baseline' else prompt
        with patch.object(backend.client.chat.completions, "create", limited_create), \
             patch("sts2_advisor.integrations.models.remote.record_event", collect), \
             patch("sts2_advisor.integrations.models.remote.build_prompt", selected_prompt), \
             patch("sts2_advisor.integrations.models.remote.build_system_prompt",
                   build_compact_system_prompt if prompt_variant == 'compact' else build_system_prompt):
            result["advice"] = backend.advise(context, evidence).to_dict()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["requests"] = request_count
    result["initial_input"] = initial_input
    result["request_options"] = request_options
    result["usage"] = [e["data"].get("usage") for e in events if e["type"] == "model.response"]
    result["model_responses"] = [{**{k: e["data"].get(k) for k in ("content", "finish_reason")},
                                 "reasoning_chars": len(e["data"].get("reasoning") or "")}
                                for e in events if e["type"] == "model.response"]
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["tools"] = [e["data"] for e in events if e["type"] == "model.tool_result"]
    result["automatic_checks"] = [e["data"].get("result") for e in events if e["type"] == "decision.attack_check"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("recordings/session-20261105-201253-679615.jsonl"))
    parser.add_argument("--indices", type=int, nargs="+", default=[69, 150])
    parser.add_argument("--modes", choices=("disabled", "enabled", "adaptive"), nargs="+", default=["disabled", "enabled"])
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--variants", choices=("baseline", "current", "compact"), nargs="+", default=["current"])
    parser.add_argument("--output", type=Path, default=Path("tmp/thinking-comparison.json"))
    args = parser.parse_args()
    load_environment(Path(".env"))
    with args.session.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    results = []
    for index in args.indices:
        for mode in args.modes:
            for variant in args.variants:
                print(f"局面 {index}，思考模式 {mode}，提示 {variant}…", flush=True)
                results.append({"index": index, **run_case(rows[index], mode, args.max_tokens, variant)})
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
