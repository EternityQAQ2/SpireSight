from __future__ import annotations

from .domain import Advice, DecisionContext, Evidence


def render_console(advice: Advice, context: DecisionContext, evidence: list[Evidence]) -> str:
    candidates = {candidate.candidate_id: candidate for candidate in context.candidates}
    selected = candidates[advice.candidate_id]
    run = context.run
    location = f"A{run.get('act', '?')} F{run.get('floor', '?')}"
    lines = [
        f"[{location} | {context.decision_type} | 置信度 {advice.confidence:.0%}]",
        f"建议：{selected.label}",
        f"理由：{advice.reason}",
    ]
    alternatives = [candidates[item].label for item in advice.alternative_ids if item in candidates]
    if alternatives:
        lines.append("备选：" + "；".join(alternatives))
    plan = [candidates[item].label for item in advice.plan_candidate_ids if item in candidates]
    if len(plan) > 1:
        lines.append("计划：" + " -> ".join(plan))
    evidence_by_id = {item.evidence_id: item for item in evidence}
    sources = [evidence_by_id[item].source_title for item in advice.evidence_ids if item in evidence_by_id]
    if sources:
        lines.append("依据：" + "；".join(sources))
    for warning in advice.warnings:
        lines.append("注意：" + warning)
    lines.append(f"动作标识：{selected.candidate_id}")
    return "\n".join(lines)

