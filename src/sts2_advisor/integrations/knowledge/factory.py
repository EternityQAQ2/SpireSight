"""Select and compose retrieval adapters at the application boundary."""
from __future__ import annotations

from pathlib import Path
from sts2_advisor.contracts import Retriever
from sts2_advisor.integrations.knowledge.lexical import NullRetriever, JsonKnowledgeRetriever
from sts2_advisor.integrations.knowledge.seeds import load_seeds, SeedInjectingRetriever
from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
from sts2_advisor.integrations.knowledge.sqlite import SqliteFactsRetriever


def create_retriever(
    mode: str, chroma_path: Path, fallback_path: Path, *, facts_path: Path | None = None,
    seeds_path: Path | None = None,
) -> Retriever:
    if mode == "none":
        return NullRetriever()
    seeds = load_seeds(
        seeds_path if seeds_path is not None
        else Path(__file__).resolve().parents[4] / "data/seeds.jsonl"
    )
    if mode == "json":
        return SeedInjectingRetriever(JsonKnowledgeRetriever(fallback_path), seeds)
    if mode == "chroma":
        return SeedInjectingRetriever(ChromaRetriever(chroma_path), seeds)
    if mode == "auto":
        retriever: Retriever | None = None
        try:
            chroma = ChromaRetriever(chroma_path)
            if chroma.collection.count() > 0:
                retriever = chroma
        except RuntimeError:
            pass
        if retriever is None:
            retriever = JsonKnowledgeRetriever(fallback_path)
        path = facts_path if facts_path is not None else (
            Path(__file__).resolve().parents[4] / "data/sts2_facts.sqlite3"
        )
        combined = SqliteFactsRetriever(path, retriever) if path.is_file() else retriever
        return SeedInjectingRetriever(combined, seeds)
    raise ValueError(f"未知检索模式: {mode}")
