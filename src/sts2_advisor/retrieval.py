from __future__ import annotations
'''
 Chroma / JSONL 攻略与 SQLite 事实检索
'''
import json
import math
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol

from .domain import DecisionContext, Evidence
from .record import record_event


# 语料以中文攻略为主，默认嵌入模型用中文 BGE；换模型只需设置同名环境变量。
# 注意：更换模型后必须删除 data/chroma 并重新入库，否则查询与文档向量不一致。
EMBEDDING_MODEL = os.environ.get("STS2_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def chroma_embedding_function():
    """ingest 与查询共用的 sentence-transformers 嵌入函数（懒加载，rag 可选依赖）。"""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


class Retriever(Protocol):
    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]: ...


class NullRetriever:
    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        return []


class JsonKnowledgeRetriever:
    """Dependency-free lexical fallback for JSONL knowledge records."""

    def __init__(self, path: Path) -> None:
        self.path = path
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
        if self.collection.count() == 0:
            return []
        query = build_query(context)
        record_event("retrieval.query", provider="chroma", query=query, top_k=top_k)
        result = self.collection.query(query_texts=[query], n_results=top_k)
        record_event("retrieval.raw_result", provider="chroma", result=result)
        evidences: list[Evidence] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
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

        # 牌堆中的卡当前不能直接操作；用它们扩展 SQLite 查询只会把无关卡牌
        # 事实塞进战斗提示词。检索只覆盖本回合可见、会影响当前决策的实体。
        visible = {
            "relics": context.player.get("relics", []),
            "potions": context.player.get("potions", []),
            "status": context.player.get("status", []),
        }
        # 战斗中手牌效果由提示词的牌组图鉴提供（游戏自带描述），这里不再查询，
        # 避免与图鉴重复占用证据槽位；其余界面（选牌、商店等）仍按手牌查询。
        if context.decision_type != "combat_next_action":
            visible["hand"] = context.player.get("hand", [])
        collect(visible)
        collect(context.situation)
        identifiers, labels = sorted(ids), sorted(names)
        record_event("retrieval.query", provider="sqlite", path=str(self.path), top_k=top_k,
                     identifiers=identifiers, names=labels, tables=list(self.TABLES))
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
                record_event("retrieval.rows", provider="sqlite", table=table,
                             rows=[dict(row) for row in rows])
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
            record_event("retrieval.rows", provider="sqlite", table="interactions", query=query,
                         rows=[dict(row) for row in rows])
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
    """JSONL（列表）与 Chroma（逗号字符串）元数据共用的过滤规则。

    decision_types 为空或含 "all" 表示不限；否则必须精确包含当前决策类型。
    旧实现把单值 decision_type="all" 当通配符，导致标注 card_reward+shop 的
    攻略泄漏进地图/战斗等所有决策；现在以 decision_types 列表为唯一裁决。
    """
    decision_types = metadata.get("decision_types")
    if isinstance(decision_types, str):
        decision_types = [item for item in decision_types.split(",") if item]
    if isinstance(decision_types, list) and decision_types:
        if "all" not in decision_types and context.decision_type not in decision_types:
            return False
    character = metadata.get("character")
    if character and character not in {context.player.get("character"), "all", "*"}:
        return False
    return True


def tokenize(text: str) -> set[str]:
    latin = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]{1,4}", text)
    return set(latin + chinese)
