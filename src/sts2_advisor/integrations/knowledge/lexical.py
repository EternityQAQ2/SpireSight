"""Dependency-free JSONL lexical retrieval and explicit retrieval opt-out."""
from __future__ import annotations

import json
import math
from pathlib import Path
from sts2_advisor.contracts import DecisionContext, Evidence
from sts2_advisor.recording.trajectory import record_event
from sts2_advisor.integrations.knowledge.common import parse_evidence_lines, tokenize, metadata_matches


class NullRetriever:
    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        return []


class JsonKnowledgeRetriever:
    """Dependency-free lexical fallback for JSONL knowledge records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[Evidence] = parse_evidence_lines(path, "知识库")

    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        query_terms = tokenize(json.dumps(context.to_dict(), ensure_ascii=False))
        record_event("retrieval.query", provider="json", path=str(self.path), top_k=top_k,
                     query_terms=sorted(query_terms), decision_type=context.decision_type,
                     character=context.player.get("character"))
        scored: list[tuple[float, Evidence]] = []
        for evidence in self.records:
            if not metadata_matches(evidence.metadata, context):
                continue
            terms = tokenize(evidence.text + " " + evidence.source_title)
            overlap = query_terms & terms
            score = 0.1 + sum(1.0 + math.log1p(len(term)) for term in overlap)
            scored.append((score, evidence))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        record_event("retrieval.ranked", provider="json", score_kind="lexical_overlap",
                     hits=[{"score": score, "evidence": evidence.to_dict()} for score, evidence in scored[:top_k]])
        return [evidence for _, evidence in scored[:top_k]]
