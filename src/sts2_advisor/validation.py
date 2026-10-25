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

    # 备选只用于展示，出现非法项时丢弃并警告，而不是作废整个合法建议；
    # 曾因模型把已选卡写进 alternative_ids 触发回退，导致选择界面被推成全选。
    dropped_alternatives = set(advice.alternative_ids) - candidate_ids

    evidence_ids = {item.evidence_id for item in evidence}
    invalid_evidence = set(advice.evidence_ids) - evidence_ids
    if invalid_evidence:
        raise AdviceValidationError(f"模型引用了未检索到的证据: {sorted(invalid_evidence)}")

    reason = advice.reason.strip()
    if not reason:
        raise AdviceValidationError("模型未给出理由")
    if len(reason) > 240:
        reason = reason[:237] + "..."

    plan_candidate_ids = list(dict.fromkeys(advice.plan_candidate_ids))
    if not plan_candidate_ids:
        plan_candidate_ids = [advice.candidate_id]
    elif plan_candidate_ids[0] != advice.candidate_id:
        raise AdviceValidationError("计划第一步与当前建议不一致")
    invalid_plan = set(plan_candidate_ids) - candidate_ids
    if invalid_plan:
        raise AdviceValidationError(f"模型计划包含非法候选: {sorted(invalid_plan)}")

    candidate_labels = {
        candidate.candidate_id: candidate.label for candidate in context.candidates
    }
    selected_label = candidate_labels[advice.candidate_id]
    alternative_ids: list[str] = []
    alternative_labels: set[str] = set()
    for candidate_id in advice.alternative_ids:
        if candidate_id in dropped_alternatives:
            continue
        label = candidate_labels[candidate_id]
        if candidate_id == advice.candidate_id or label == selected_label or label in alternative_labels:
            continue
        alternative_ids.append(candidate_id)
        alternative_labels.add(label)
        if len(alternative_ids) == 2:
            break

    warnings = list(dict.fromkeys(advice.warnings))
    if dropped_alternatives:
        warning = "模型返回了非法备选，已忽略：" + ",".join(sorted(dropped_alternatives))
        warnings = [warning] + warnings

    return replace(
        advice,
        reason=reason,
        confidence=max(0.0, min(1.0, advice.confidence)),
        alternative_ids=alternative_ids,
        evidence_ids=list(dict.fromkeys(advice.evidence_ids))[:6],
        warnings=warnings[:4],
        plan_candidate_ids=plan_candidate_ids[:8],
    )

