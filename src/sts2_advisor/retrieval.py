from __future__ import annotations
'''
 Chroma / JSONL 知识检索
'''
import json
import math
import re
from pathlib import Path
from typing import Any, Protocol

from .domain import DecisionContext, Evidence


class Retriever(Protocol):
    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]: ...


class NullRetriever:
    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        return []


class JsonKnowledgeRetriever:
    """Dependency-free lexical fallback for JSONL knowledge records."""

    def __init__(self, path: Path) -> None:
        self.records: list[Evidence] = []
        if not path.exists():
            return
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                self.records.append(
                    Evidence(
                        evidence_id=str(record["evidence_id"]),
                        text=str(record["text"]),
                        source_title=str(record.get("source_title", record["evidence_id"])),
                        source_url=str(record.get("source_url", "")),
                        metadata=dict(record.get("metadata") or {}),
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"知识库 {path} 第 {line_number} 行无效: {exc}") from exc

    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        query_terms = tokenize(json.dumps(context.to_dict(), ensure_ascii=False))
        scored: list[tuple[float, Evidence]] = []
        for evidence in self.records:
            if not metadata_matches(evidence.metadata, context):
                continue
            terms = tokenize(evidence.text + " " + evidence.source_title)
            overlap = query_terms & terms
            score = 0.1 + sum(1.0 + math.log1p(len(term)) for term in overlap)
            scored.append((score, evidence))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [evidence for _, evidence in scored[:top_k]]


class ChromaRetriever:
    def __init__(self, path: Path, collection_name: str = "strategy_chunks") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("未安装 chromadb；请安装 rag 可选依赖或使用 JSONL 回退知识库") from exc
        self.collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            name=collection_name
        )

    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        if self.collection.count() == 0:
            return []
        query = build_query(context)
        result = self.collection.query(query_texts=[query], n_results=top_k)
        evidences: list[Evidence] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        for evidence_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            decision_types = str(metadata.get("decision_types", "")).split(",")
            decision_type = str(metadata.get("decision_type", ""))
            if decision_type not in {"", "all", context.decision_type} and context.decision_type not in decision_types:
                continue
            evidences.append(
                Evidence(
                    evidence_id=str(evidence_id),
                    text=str(text),
                    source_title=str(metadata.get("source_title", evidence_id)),
                    source_url=str(metadata.get("source_url", "")),
                    metadata=dict(metadata),
                )
            )
        return evidences


def create_retriever(mode: str, chroma_path: Path, fallback_path: Path) -> Retriever:
    if mode == "none":
        return NullRetriever()
    if mode == "json":
        return JsonKnowledgeRetriever(fallback_path)
    if mode == "chroma":
        return ChromaRetriever(chroma_path)
    if mode == "auto":
        try:
            retriever = ChromaRetriever(chroma_path)
            if retriever.collection.count() > 0:
                return retriever
        except RuntimeError:
            pass
        return JsonKnowledgeRetriever(fallback_path)
    raise ValueError(f"未知检索模式: {mode}")


def build_query(context: DecisionContext) -> str:
    player = context.player
    situation = context.situation
    return " ".join(
        [
            context.decision_type,
            str(player.get("character", "")),
            f"act {context.run.get('act', '')}",
            json.dumps(situation, ensure_ascii=False),
        ]
    )


def metadata_matches(metadata: dict[str, Any], context: DecisionContext) -> bool:
    decision_types = metadata.get("decision_types")
    if isinstance(decision_types, list) and context.decision_type not in decision_types:
        return False
    decision_type = metadata.get("decision_type")
    if decision_type and decision_type != context.decision_type:
        return False
    character = metadata.get("character")
    if character and character not in {context.player.get("character"), "all", "*"}:
        return False
    return True


def tokenize(text: str) -> set[str]:
    latin = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]{1,4}", text)
    return set(latin + chinese)
