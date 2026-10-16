from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .domain import Advice, DecisionContext, Evidence


class SessionRecorder:
    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = directory / f"session-{stamp}.jsonl"

    def record(
        self,
        raw_state: dict,
        context: DecisionContext,
        evidence: list[Evidence],
        advice: Advice,
        backend: str,
    ) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "raw_state": raw_state,
            "context": context.to_dict(),
            "evidence": [item.to_dict() for item in evidence],
            "advice": advice.to_dict(),
            "training_consent": False,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

