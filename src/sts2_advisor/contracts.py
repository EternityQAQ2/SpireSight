from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


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
    selection_analysis: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanCheck:
    candidate_ids: list[str]
    error: str | None = None
    boundary: str | None = None


class AdvisorBackend(Protocol):
    def advise(self, context: DecisionContext, evidence: list[Evidence]) -> Advice: ...


class Retriever(Protocol):
    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]: ...


class GameClient(Protocol):
    def get_singleplayer_state(self) -> dict[str, Any]: ...


class ActionExecutorPort(Protocol):
    last_decision_id: str | None

    def execute(self, advice: Advice, context: DecisionContext) -> dict | None: ...


class DecisionRecorder(Protocol):
    def record(self, raw_state: dict, context: DecisionContext, evidence: list[Evidence],
               advice: Advice, backend: str) -> None: ...


class Sts2ApiError(RuntimeError):
    """Game observation or action transport failed."""


class Sts2ActionRejectedError(Sts2ApiError):
    """The game explicitly rejected the action; it was not executed."""
