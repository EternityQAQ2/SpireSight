"""Knowledge document parsing, query construction and metadata filtering."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from sts2_advisor.contracts import DecisionContext, Evidence


def parse_evidence_lines(path: Path, kind: str) -> list[Evidence]:
    """逐行解析 JSONL 知识记录；文件缺失视为空，坏行报错并标出行号。"""
    records: list[Evidence] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records.append(
                Evidence(
                    evidence_id=str(record["evidence_id"]),
                    text=str(record["text"]),
                    source_title=str(record.get("source_title", record["evidence_id"])),
                    source_url=str(record.get("source_url", "")),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{kind} {path} 第 {line_number} 行无效: {exc}") from exc
    return records


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
    enemies = metadata.get("enemies")
    if isinstance(enemies, str):
        enemies = [item.strip() for item in enemies.split(",") if item.strip()]
    if isinstance(enemies, list) and enemies:
        # Boss/精英攻略只对“当前战斗里实际出现的敌人”生效。否则同类战斗词汇
        # （造成伤害、回合、生命值）会让所有 Boss 攻略在普通战斗里抢占证据槽位。
        if not set(enemies) & combat_enemy_names(context):
            return False
    return True


def combat_enemy_names(context: DecisionContext) -> set[str]:
    """从紧凑局面中取出当前战斗的敌人名。"""
    battle = context.situation.get("battle")
    if not isinstance(battle, dict):
        return set()
    names: set[str] = set()
    for enemy in battle.get("enemies") or []:
        if isinstance(enemy, dict) and isinstance(enemy.get("name"), str):
            names.add(enemy["name"])
    return names


def tokenize(text: str) -> set[str]:
    latin = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]{1,4}", text)
    return set(latin + chinese)
