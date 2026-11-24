"""Read-only entity and interaction lookup against the local SQLite facts database."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from sts2_advisor.contracts import DecisionContext, Evidence
from sts2_advisor.contracts import Retriever
from sts2_advisor.developer import dev_section
from sts2_advisor.recording.trajectory import record_event
from sts2_advisor.integrations.knowledge.lexical import NullRetriever
from sts2_advisor.integrations.knowledge.relevance import entity_name, rank_facts


def interaction_mentions(context: DecisionContext, fact_names: set[str]) -> tuple[str, list[str]]:
    """Match full entity names separately from references in descriptive text.

    A card named 岿然不动 is not evidence of the unrelated 不动 power. Mask
    observed names longest-first before looking for descriptive references;
    keep exact names (and upgraded-card base names) available for direct lookup.
    """
    names = set(fact_names)

    def collect(value):
        if isinstance(value, dict):
            name = entity_name(value)
            if name:
                names.add(name)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    value = context.to_dict()
    collect(value)
    names.update(name[:-1] for name in list(names) if name.endswith('+'))
    query = json.dumps(value, ensure_ascii=False)
    for name in sorted(names, key=len, reverse=True):
        query = query.replace(name, ' ')
    return query, sorted(names)


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
                    value.get("id") or value.get("event_id") or value.get('card_id')
                    or value.get('relic_id') or value.get('potion_id')
                    or re.sub(r"_\d+$", "", str(value.get("entity_id", "")))
                )
                name = entity_name(value)
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
        dev_section("检索查询", "sqlite 实体查询", fields={
            "path": str(self.path), "top_k": top_k,
            "identifiers": identifiers, "names": labels,
        })
        id_slots = ",".join("?" for _ in identifiers) or "NULL"
        name_slots = ",".join("?" for _ in labels) or "NULL"
        facts: list[Evidence] = []
        table_counts: dict[str, int] = {}
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            for table in self.TABLES:
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE id COLLATE NOCASE IN ({id_slots}) "
                    f"OR name IN ({name_slots}) ORDER BY id",
                    identifiers + labels,
                ).fetchall()
                table_counts[table] = len(rows)
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
                        metadata={"table": table, "name": row["name"], 'entity_id': row['id'],
                                  "description": row["description"] if "description" in row.keys() else ""},
                    ))
            # 完整实体名称精确匹配；说明中的效果引用另行匹配，避免短名撞长名。
            query, interaction_names = interaction_mentions(context, names)
            page_slots = ','.join('?' for _ in interaction_names) or 'NULL'
            rows = connection.execute(
                "SELECT i.id, i.description, p.page_name FROM interactions i "
                "JOIN interaction_pages p ON p.interaction_id = i.id "
                f"WHERE p.page_name IN ({page_slots}) OR instr(?, p.page_name) > 0 "
                "ORDER BY i.id, p.page_name", (*interaction_names, query),
            ).fetchall()
            record_event("retrieval.rows", provider="sqlite", table="interactions", query=query,
                         names=interaction_names, rows=[dict(row) for row in rows])
            grouped = {}
            for row in rows:
                item = grouped.setdefault(row['id'], {'description': row['description'], 'pages': []})
                item['pages'].append(row['page_name'])
            interactions = [Evidence(
                f"sqlite:interactions:{row['id']}", row["description"], "灰机 Wiki：效果交互",
                metadata={"table": "interactions"},
            ) for row in rows]
        dev_section("检索事实", "sqlite 命中统计", fields={
            "table_row_counts": {key: value for key, value in table_counts.items() if value},
            "interaction_count": len(interactions),
        })

        facts = rank_facts(context, interactions + facts)
        record_event('retrieval.ranked', provider='sqlite', hits=[
            {'id': e.evidence_id, 'priority': e.metadata['priority'], 'reason': e.metadata['relevance']}
            for e in facts])
        guides = self.fallback.retrieve(context, top_k)
        guide_count = min(len(guides), top_k // 2) if facts else len(guides)
        fact_count = top_k - guide_count
        return (facts[:fact_count] + guides[:guide_count] + facts[fact_count:] + guides[guide_count:])[:top_k]
