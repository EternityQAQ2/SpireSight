from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    action_type: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    legal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionContext:
    schema_version: str
    decision_id: str
    decision_type: str
    state_type: str
    run: dict[str, Any]
    player: dict[str, Any]
    situation: dict[str, Any]
    candidates: list[Candidate]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return result


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    source_title: str
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Advice:
    decision_id: str
    candidate_id: str
    reason: str
    confidence: float
    alternative_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan_candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

