from __future__ import annotations

from dataclasses import replace

from .domain import Advice, DecisionContext, Evidence


class AdviceValidationError(ValueError):
    pass


def validate_advice(
    advice: Advice,
    context: DecisionContext,
    evidence: list[Evidence],
) -> Advice:
    if advice.decision_id != context.decision_id:
        raise AdviceValidationError("模型返回了过期或错误的 decision_id")

    candidate_ids = {candidate.candidate_id for candidate in context.candidates}
    if advice.candidate_id not in candidate_ids:
        raise AdviceValidationError(f"模型选择了非法候选: {advice.candidate_id}")

    invalid_alternatives = set(advice.alternative_ids) - candidate_ids
    if invalid_alternatives:
        raise AdviceValidationError(f"模型返回了非法备选: {sorted(invalid_alternatives)}")

    evidence_ids = {item.evidence_id for item in evidence}
    invalid_evidence = set(advice.evidence_ids) - evidence_ids
    if invalid_evidence:
        raise AdviceValidationError(f"模型引用了未检索到的证据: {sorted(invalid_evidence)}")

    reason = advice.reason.strip()
    if not reason:
        raise AdviceValidationError("模型未给出理由")
    if len(reason) > 240:
        reason = reason[:237] + "..."

    return replace(
        advice,
        reason=reason,
        confidence=max(0.0, min(1.0, advice.confidence)),
        alternative_ids=list(dict.fromkeys(advice.alternative_ids))[:2],
        evidence_ids=list(dict.fromkeys(advice.evidence_ids))[:6],
        warnings=list(dict.fromkeys(advice.warnings))[:4],
    )

