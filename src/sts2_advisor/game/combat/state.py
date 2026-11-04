"""Typed internal simulation state; raw game API dictionaries stop at its boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class EnemyState:
    entity_id: str
    hp: int
    block: int
    vulnerable: int
    artifact: int
    incoming: int
    thorns: int
    minion: bool = False


@dataclass(slots=True)
class CombatState:
    hp: int
    block: int
    energy: int
    strength: int
    strength_delta: int
    kusari: int | None
    enemies: list[EnemyState]
    used: set[int] = field(default_factory=set)
    rage: int = 0
    pen_nib: int | None = None
    gremlin_horn: bool = False
    boundary: bool = False


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    damage: int
    block: int
    incoming: int
    hp_loss: int
    hp_after: int
    energy_left: int
    kills: int
    enemies_remaining: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
