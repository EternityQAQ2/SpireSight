"""Optional Chroma guide retrieval adapter."""
from __future__ import annotations

from pathlib import Path
from sts2_advisor.contracts import DecisionContext, Evidence
from sts2_advisor.developer import dev_section
from sts2_advisor.recording.trajectory import record_event
from sts2_advisor.integrations.knowledge.embeddings import chroma_embedding_function
from sts2_advisor.integrations.knowledge.common import build_query, metadata_matches


class ChromaRetriever:
    def __init__(self, path: Path, collection_name: str = "strategy_chunks") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("未安装 chromadb；请安装 rag 可选依赖或使用 JSONL 回退知识库") from exc
        self.collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_embedding_function(),
        )

    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        collection_count = self.collection.count()
        if collection_count == 0:
            return []
        query = build_query(context)
        candidate_k = max(top_k * 3, 12)
        record_event("retrieval.query", provider="chroma", query=query, top_k=top_k,
                     candidate_k=candidate_k)
        dev_section("检索查询", "chroma 语义检索", fields={
            "query": query, "top_k": top_k, "candidate_k": candidate_k,
            "collection_count": collection_count,
        })
        result = self.collection.query(query_texts=[query], n_results=candidate_k)
        record_event("retrieval.raw_result", provider="chroma", result=result)
        evidences: list[Evidence] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        dev_section("检索排序", "chroma 原始距离", fields={
            "hits": [
                {"evidence_id": evidence_id, "distance": distance}
                for evidence_id, distance in zip(ids, distances)
            ],
        })
        for evidence_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            if not metadata_matches(metadata, context):
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
        return evidences[:top_k]
