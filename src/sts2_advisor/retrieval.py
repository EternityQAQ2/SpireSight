from __future__ import annotations
'''
 Chroma / JSONL 攻略与 SQLite 事实检索
'''
import json
import math
import re
import sqlite3
from contextlib import closing
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


class SqliteFactsRetriever:
    """按局面中的实体 ID/名称只读查询事实与效果交互，并保留已有攻略。"""

    TABLES = (
        "cards", "relics", "potions", "powers", "monsters", "events",
        "enchantments", "afflictions", "keywords", "intents",
    )

    def __init__(self, path: Path, fallback: Retriever | None = None) -> None:
        self.path = path
        self.fallback = fallback or NullRetriever()

    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        if top_k <= 0:
            return []
        ids: set[str] = set()
        names: set[str] = set()

        def collect(value):
            if isinstance(value, dict):
                entity_id = (
                    value.get("id") or value.get("event_id")
                    or re.sub(r"_\d+$", "", str(value.get("entity_id", "")))
                )
                name = value.get("name") or value.get("event_name")
                upgraded = value.get("is_upgraded", False)
                if isinstance(entity_id, str) and entity_id:
                    entity_id = entity_id.lower()
                    if upgraded and not entity_id.endswith("_upgrade"):
                        entity_id += "_upgrade"
                    ids.add(entity_id)
                if isinstance(name, str) and name:
                    names.add(name + "+" if upgraded and not name.endswith("+") else name)
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(context.player)
        collect(context.situation)
        identifiers, labels = sorted(ids), sorted(names)
        id_slots = ",".join("?" for _ in identifiers) or "NULL"
        name_slots = ",".join("?" for _ in labels) or "NULL"
        facts: list[Evidence] = []
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            for table in self.TABLES:
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE id COLLATE NOCASE IN ({id_slots}) "
                    f"OR name IN ({name_slots}) ORDER BY id",
                    identifiers + labels,
                ).fetchall()
                # 同名卡牌可能属于不同角色，已有准确 ID 时优先使用对应记录。
                exact_names = {row["name"] for row in rows if row["id"].lower() in ids}
                for row in rows:
                    if row["id"].lower() not in ids and row["name"] in exact_names:
                        continue
                    names.add(row["name"])
                    text = json.dumps({
                        key: row[key] for key in row.keys()
                        if key != "description_wiki" and row[key] is not None
                    }, ensure_ascii=False)
                    facts.append(Evidence(
                        f"sqlite:{table}:{row['id']}", text, f"灰机 Wiki：{row['name']}",
                        metadata={"table": table},
                    ))
            # 交互说明的关联页面可出现在名称、卡牌说明或状态说明中。
            query = json.dumps(context.to_dict(), ensure_ascii=False) + " " + " ".join(sorted(names))
            rows = connection.execute(
                "SELECT DISTINCT i.id, i.description FROM interactions i "
                "JOIN interaction_pages p ON p.interaction_id = i.id "
                "WHERE instr(?, p.page_name) > 0 ORDER BY i.id LIMIT ?", (query, top_k),
            ).fetchall()
            interactions = [Evidence(
                f"sqlite:interactions:{row['id']}", row["description"], "灰机 Wiki：效果交互",
                metadata={"table": "interactions"},
            ) for row in rows]

        facts = interactions + facts
        guides = self.fallback.retrieve(context, top_k)
        guide_count = min(len(guides), top_k // 2) if facts else len(guides)
        return (guides[:guide_count] + facts + guides[guide_count:])[:top_k]


def create_retriever(
    mode: str, chroma_path: Path, fallback_path: Path, *, facts_path: Path | None = None,
) -> Retriever:
    if mode == "none":
        return NullRetriever()
    if mode == "json":
        return JsonKnowledgeRetriever(fallback_path)
    if mode == "chroma":
        return ChromaRetriever(chroma_path)
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
            Path(__file__).resolve().parents[2] / "data/sts2_facts.sqlite3"
        )
        return SqliteFactsRetriever(path, retriever) if path.is_file() else retriever
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
