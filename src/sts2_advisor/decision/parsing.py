"""Parse provider-independent structured advice."""
from __future__ import annotations

import json
import re
from sts2_advisor.contracts import Advice, DecisionContext


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
        selection_analysis=payload.get("selection_analysis") or {},
    )
