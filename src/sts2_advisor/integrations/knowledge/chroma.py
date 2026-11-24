"""Optional Chroma guide retrieval adapter."""
from __future__ import annotations

from pathlib import Path
from sts2_advisor.contracts import DecisionContext, Evidence
from sts2_advisor.developer import dev_section
from sts2_advisor.recording.trajectory import record_event
from sts2_advisor.integrations.knowledge.embeddings import chroma_embedding_function
from sts2_advisor.integrations.knowledge.common import build_query, metadata_matches
from sts2_advisor.integrations.knowledge.examples import EXAMPLE_KIND, matching_mechanisms, select_examples


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
        # A small overfetch can contain only other characters/bosses/build
        # advice. Retry locally with a bounded pool rather than let an example
        # silently replace a SQLite fact because no guide slot existed.
        raw_metas = (result.get('metadatas') or [[]])[0]
        expanded_k = min(max(48, top_k * 6), self.collection.count())
        if expanded_k > candidate_k and not any(
                metadata_matches(m or {}, context) and (m or {}).get('kind') != EXAMPLE_KIND
                for m in raw_metas):
            result = self.collection.query(query_texts=[query], n_results=expanded_k)
            record_event('retrieval.raw_result', provider='chroma_expanded', result=result)
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
        mechanisms = matching_mechanisms(context)
        if mechanisms:
            # Same collection and embedding model; no new store or startup writes.
            examples = self.collection.query(query_texts=[query], n_results=1, where={
                '$and': [{'kind': EXAMPLE_KIND}, {'review_status': 'verified_arithmetic'},
                         {'mechanism': {'$in': mechanisms}},
                         {'character': {'$in': ['all', str(context.player.get('character', ''))]}}]})
            record_event('retrieval.raw_result', provider='chroma_examples', result=examples)
            for ident, text, metadata in zip((examples.get('ids') or [[]])[0],
                    (examples.get('documents') or [[]])[0], (examples.get('metadatas') or [[]])[0]):
                metadata = metadata or {}
                evidences.insert(0, Evidence(str(ident), str(text), str(metadata.get('source_title', ident)),
                                             str(metadata.get('source_url', '')), dict(metadata)))
        return select_examples(context, evidences, top_k)
