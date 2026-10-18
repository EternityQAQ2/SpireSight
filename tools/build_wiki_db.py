#!/usr/bin/env python3
"""Build the STS2 SQLite fact database from Huiji Wiki TabX JSONL exports."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(__file__).resolve().parent / "sts_raw_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sts2_facts.sqlite3"

# These tables directly support single-player decisions.
CORE_EXPORTS = {
    "Card", "Relic", "Potion", "Power", "Enchantment", "Monster", "Event",
    "Interaction", "Affliction", "Keyword", "Intent",
}
OPTIONAL_EXPORTS = {"Modifier"}  # Daily challenge only.


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE cards (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, color TEXT, rarity TEXT,
    card_type TEXT, cost TEXT, description TEXT, description_wiki TEXT,
    upgrade_id TEXT, multiplayer_only INTEGER NOT NULL DEFAULT 0,
    predecessor TEXT
);
CREATE INDEX idx_cards_name ON cards(name);
CREATE INDEX idx_cards_color ON cards(color);

CREATE TABLE relics (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, pool TEXT, tier TEXT,
    description TEXT, description_wiki TEXT, ancient INTEGER,
    predecessor TEXT
);
CREATE INDEX idx_relics_name ON relics(name);
CREATE INDEX idx_relics_pool ON relics(pool);

CREATE TABLE potions (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, color TEXT, tier TEXT,
    description TEXT, description_wiki TEXT
);
CREATE INDEX idx_potions_name ON potions(name);

CREATE TABLE powers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
    description_wiki TEXT, power_type TEXT, stack_type TEXT,
    allow_negative INTEGER
);
CREATE INDEX idx_powers_name ON powers(name);

CREATE TABLE enchantments (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
    description_wiki TEXT
);

CREATE TABLE monsters (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, min_hp INTEGER, max_hp INTEGER,
    ascension_min_hp INTEGER, ascension_max_hp INTEGER, tier TEXT,
    powers TEXT, stage TEXT
);
CREATE INDEX idx_monsters_name ON monsters(name);
CREATE INDEX idx_monsters_stage ON monsters(stage);

CREATE TABLE events (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, stage TEXT,
    preconditions TEXT, multiplayer_event INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_events_name ON events(name);

CREATE TABLE afflictions (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
    description_wiki TEXT
);

CREATE TABLE keywords (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
    description_wiki TEXT, visible INTEGER
);

CREATE TABLE intents (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, game_text TEXT,
    description TEXT, description_wiki TEXT
);

CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    description_wiki TEXT NOT NULL
);
CREATE TABLE interaction_pages (
    interaction_id INTEGER NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    page_name TEXT NOT NULL,
    PRIMARY KEY (interaction_id, page_name)
);
CREATE INDEX idx_interaction_pages_name ON interaction_pages(page_name);

CREATE TABLE modifiers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
    description_wiki TEXT, kind TEXT
);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从灰机 Wiki TabX JSONL 构建 STS2 SQLite 事实库")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL 导出目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="SQLite 输出路径")
    parser.add_argument("--include-modifiers", action="store_true", help="导入每日挑战特效")
    return parser.parse_args()


def load_export(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the wrapper JSONL and expand the embedded TabX document."""
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
                raw = wrapper.get("raw_line", wrapper)
                document = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, AttributeError) as exc:
                raise ValueError(f"{path}:{line_number} 不是有效的 TabX JSONL") from exc
            if not isinstance(document, dict):
                raise ValueError(f"{path}:{line_number} 的 TabX 内容必须是对象")
            documents.append(document)

    if len(documents) != 1:
        raise ValueError(f"{path} 应包含一个完整 TabX 文档，实际为 {len(documents)} 个")
    document = documents[0]
    fields = [field["name"] for field in document.get("schema", {}).get("fields", [])]
    rows = document.get("data", [])
    if not fields or not isinstance(rows, list):
        raise ValueError(f"{path} 缺少 schema.fields 或 data")
    records = [dict(zip(fields, row)) for row in rows if isinstance(row, list)]
    return document, records


def text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    # "null" is a real STS2 card id, so only an actual null/empty value is absent.
    return result or None


def integer(value: Any) -> int | None:
    value = text(value)
    if value is None:
        return None
    match = re.search(r"-?\d+", value)
    return int(match.group()) if match else None


def boolean(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    value = (text(value) or "").lower()
    if value in {"1", "true", "yes", "是", "有", "仅多人"}:
        return 1
    if value in {"0", "false", "no", "否", "无"}:
        return 0
    return None


def clean_wiki(value: Any) -> str | None:
    """Remove presentation markup while retaining gameplay values and terms."""
    source = text(value)
    if source is None:
        return None
    result = source
    result = re.sub(r"<!--.*?-->", "", result, flags=re.DOTALL)
    result = re.sub(r"\[\[(?:[^]|]+\|)?([^]]+)]]", r"\1", result)

    template = re.compile(r"{{([^{}]+)}}")
    for _ in range(8):
        updated = template.sub(lambda m: _template_text(m.group(1)), result)
        if updated == result:
            break
        result = updated
    result = re.sub(r"<[^>]+>", "", result)
    result = result.replace("'''", "").replace("''", "")
    result = re.sub(r"\s+", " ", result).strip()
    return result or None


def _template_text(body: str) -> str:
    parts = [part.strip() for part in body.split("|")]
    if not parts:
        return ""
    name = parts[0].lower()
    values = [part.split("=", 1)[-1].strip() for part in parts[1:] if part.strip()]
    if name in {"颜色", "color"} and len(values) >= 2:
        return values[-1]
    if name in {"状态", "power", "关键词", "keyword", "卡牌", "card", "遗物", "relic"}:
        return values[0] if values else ""
    # Unknown templates are retained as readable values rather than deleted.
    return " ".join(values)


def required(record: dict[str, Any], field: str, source: str) -> str:
    value = text(record.get(field))
    if value is None:
        raise ValueError(f"{source} 存在缺少 {field!r} 的记录")
    return value


def insert_many(connection: sqlite3.Connection, sql: str, rows: Iterable[tuple[Any, ...]]) -> int:
    materialized = list(rows)
    connection.executemany(sql, materialized)
    return len(materialized)


def import_records(connection: sqlite3.Connection, kind: str, records: list[dict[str, Any]]) -> int:
    if kind == "Card":
        return insert_many(connection, "INSERT INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), text(r.get("color")), text(r.get("rarity")),
             text(r.get("type")), text(r.get("cost")), clean_wiki(r.get("description_raw") or r.get("description")),
             text(r.get("description")), text(r.get("upgrade")), boolean(r.get("multiplayer_only")) or 0,
             text(r.get("predecessor"))) for r in records))
    if kind == "Relic":
        return insert_many(connection, "INSERT INTO relics VALUES (?,?,?,?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), text(r.get("pool")), text(r.get("tier")),
             clean_wiki(r.get("description_raw") or r.get("description")), text(r.get("description")),
             boolean(r.get("ancient")), text(r.get("predecessor"))) for r in records))
    if kind == "Potion":
        return insert_many(connection, "INSERT INTO potions VALUES (?,?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), text(r.get("color")), text(r.get("tier")),
             clean_wiki(r.get("description_raw") or r.get("description")), text(r.get("description"))) for r in records))
    if kind == "Power":
        return insert_many(connection, "INSERT INTO powers VALUES (?,?,?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), clean_wiki(r.get("description")),
             text(r.get("description")), text(r.get("type")), text(r.get("stack_type")),
             boolean(r.get("allow_negative"))) for r in records))
    if kind in {"Enchantment", "Affliction"}:
        table = "enchantments" if kind == "Enchantment" else "afflictions"
        return insert_many(connection, f"INSERT INTO {table} VALUES (?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), clean_wiki(r.get("description")),
             text(r.get("description"))) for r in records))
    if kind == "Monster":
        return insert_many(connection, "INSERT INTO monsters VALUES (?,?,?,?,?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), integer(r.get("minHP")), integer(r.get("maxHP")),
             integer(r.get("ascenderminHP")), integer(r.get("ascendermaxHP")), text(r.get("tier")),
             clean_wiki(r.get("power")), text(r.get("stage")),) for r in records))
    if kind == "Event":
        return insert_many(connection, "INSERT INTO events VALUES (?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), text(r.get("stage_act")),
             clean_wiki(r.get("preconditions")), boolean(r.get("mul_event")) or 0) for r in records))
    if kind == "Keyword":
        return insert_many(connection, "INSERT INTO keywords VALUES (?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), clean_wiki(r.get("description")),
             text(r.get("description")), boolean(r.get("show"))) for r in records))
    if kind == "Intent":
        return insert_many(connection, "INSERT INTO intents VALUES (?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), clean_wiki(r.get("game_text")),
             clean_wiki(r.get("description")), text(r.get("description"))) for r in records))
    if kind == "Interaction":
        count = 0
        for record in records:
            raw = required(record, "text", kind)
            cursor = connection.execute(
                "INSERT INTO interactions(description, description_wiki) VALUES (?, ?)",
                (clean_wiki(raw) or raw, raw),
            )
            pages = {page.strip() for page in (text(record.get("pages")) or "").split(";") if page.strip()}
            connection.executemany(
                "INSERT INTO interaction_pages(interaction_id, page_name) VALUES (?, ?)",
                ((cursor.lastrowid, page) for page in sorted(pages)),
            )
            count += 1
        return count
    if kind == "Modifier":
        return insert_many(connection, "INSERT INTO modifiers VALUES (?,?,?,?,?)", (
            (required(r, "id", kind), required(r, "name", kind), clean_wiki(r.get("description")),
             text(r.get("description")), text(r.get("kind"))) for r in records))
    raise ValueError(f"不支持的导出类型: {kind}")


def build_database(input_dir: Path, output: Path, include_modifiers: bool) -> dict[str, int]:
    selected = CORE_EXPORTS | (OPTIONAL_EXPORTS if include_modifiers else set())
    missing = sorted(kind for kind in CORE_EXPORTS if not (input_dir / f"{kind}.jsonl").exists())
    if missing:
        raise FileNotFoundError("缺少核心导出: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    counts: dict[str, int] = {}
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(SCHEMA_SQL)
            for kind in sorted(selected):
                path = input_dir / f"{kind}.jsonl"
                if not path.exists():
                    continue
                document, records = load_export(path)
                counts[kind] = import_records(connection, kind, records)
                source = document.get("sources") or document.get("description") or ""
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (f"source.{kind}", json.dumps(source, ensure_ascii=False)),
                )
            metadata = {
                "built_at": datetime.now(UTC).isoformat(),
                "source": "https://sts2.huijiwiki.com/",
                "license": "See each source TabX page; preserve attribution when redistributing.",
                "schema_version": "1",
            }
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
            connection.commit()
        finally:
            connection.close()
            connection = None
        temporary.replace(output)
    except Exception:
        if connection is not None:
            connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    return counts


def main() -> int:
    args = parse_args()
    counts = build_database(args.input.resolve(), args.output.resolve(), args.include_modifiers)
    print(f"数据库已生成: {args.output.resolve()}")
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}")
    print(f"  合计: {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
