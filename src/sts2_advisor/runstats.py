"""每局 AI 战绩统计：从轨迹库重建 data/run_stats.json 与 docs/run-stats.md。

切局规则：角色变化或层数回退即开新局；game_over 观察收尾。
结局判定：game_over 且 hp=0 → death；game_over 且 hp>0 → victory（推断）；
无 game_over → aborted（advisor 被手动停止）。
advisor 退出时自动调用 refresh()（见 cli.py），也可经 tools/report_runs.py 手动重建。
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path("recordings/trajectories.sqlite3")
DEFAULT_JSON = Path("data/run_stats.json")
DEFAULT_MD = Path("docs/run-stats.md")

EVENT_STATS = {
    "decision.input": "decisions",
    "decision.deterministic": "deterministic",
    "plan.reused": "plan_reuses",
    "model.response": "model_calls",
}

OUTCOME_LABEL = {"death": "死亡", "victory": "胜利(推断)", "aborted": "中途停止"}


def load_observations(db: Path) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, session_id, payload_json FROM events "
        "WHERE event_type='environment.observation' ORDER BY event_id"
    ).fetchall()
    conn.close()
    observations = []
    for row in rows:
        try:
            raw = (json.loads(row["payload_json"]).get("raw_state") or {})
        except json.JSONDecodeError:
            continue
        if not raw.get("state_type"):
            continue
        run = raw.get("run") or {}
        player = raw.get("player") or {}
        observations.append(
            {
                "timestamp": row["timestamp"],
                "session_id": row["session_id"],
                "state_type": raw["state_type"],
                "character": player.get("character"),
                "hp": player.get("hp"),
                "max_hp": player.get("max_hp"),
                "act": run.get("act"),
                "floor": run.get("floor"),
                "ascension": run.get("ascension"),
            }
        )
    return observations


def load_events(db: Path) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, event_type, payload_json FROM events "
        "WHERE event_type IN ('decision.input', 'decision.deterministic', "
        "'plan.reused', 'model.response') ORDER BY event_id"
    ).fetchall()
    conn.close()
    events = []
    for row in rows:
        item = {"timestamp": row["timestamp"], "event_type": row["event_type"]}
        if row["event_type"] == "model.response":
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
            usage = payload.get("usage") or {}
            item["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
            item["hit"] = int(payload.get("cache_hit_tokens") or 0)
        events.append(item)
    return events


def split_runs(observations: list[dict]) -> list[dict]:
    runs: list[dict] = []

    def close(current: dict | None) -> None:
        if current is not None:
            runs.append(current)

    current: dict | None = None
    for obs in observations:
        if obs["state_type"] == "menu":
            continue
        character, floor = obs["character"], obs["floor"]
        if obs["state_type"] == "game_over":
            if current is not None:
                current["ended_at"] = obs["timestamp"]
                current["game_over_observed"] = True
                if obs["hp"] is not None:
                    current["final_hp"] = obs["hp"]
                close(current)
                current = None
            continue
        if character is None or floor is None:
            continue
        new_run = (
            current is None
            or character != current["character"]
            or floor < current["max_floor"]
            # 同角色换会话仍停在第 1 层：上一局在第 1 层就被放弃，另起一局
            or (floor == 1 and obs["session_id"] != current["last_session_id"])
        )
        if new_run:
            close(current)
            current = {
                "character": character,
                "started_at": obs["timestamp"],
                "ended_at": obs["timestamp"],
                "max_floor": floor,
                "max_act": obs["act"],
                "ascension": obs["ascension"] or 0,
                "final_hp": obs["hp"],
                "max_hp": obs["max_hp"],
                "game_over_observed": False,
                "session_ids": [],
                "last_session_id": obs["session_id"],
            }
        current["ended_at"] = obs["timestamp"]
        if obs["session_id"] not in current["session_ids"]:
            current["session_ids"].append(obs["session_id"])
        current["max_floor"] = max(current["max_floor"], floor)
        if obs["act"] is not None:
            current["max_act"] = max(current["max_act"] or 0, obs["act"])
        if obs["ascension"] is not None:
            current["ascension"] = max(current["ascension"], obs["ascension"])
        if obs["hp"] is not None:
            current["final_hp"] = obs["hp"]
        if obs["max_hp"]:
            current["max_hp"] = obs["max_hp"]
        current["last_session_id"] = obs["session_id"]
    close(current)
    return runs


def outcome_of(run: dict) -> str:
    if not run.get("game_over_observed"):
        return "aborted"
    return "death" if (run.get("final_hp") or 0) == 0 else "victory"


def attach_events(runs: list[dict], events: list[dict]) -> None:
    counters = [Counter() for _ in runs]
    run_index = 0
    for event in events:
        while run_index < len(runs) and event["timestamp"] > runs[run_index]["ended_at"]:
            run_index += 1
        if run_index >= len(runs):
            break
        run = runs[run_index]
        if run["started_at"] <= event["timestamp"] <= run["ended_at"]:
            counters[run_index][EVENT_STATS[event["event_type"]]] += 1
            if event["event_type"] == "model.response":
                counters[run_index]["prompt_tokens"] += event["prompt_tokens"]
                counters[run_index]["cache_hit_tokens"] += event["hit"]
    for run, counter in zip(runs, counters):
        run.pop("last_session_id", None)
        run.update({key: counter.get(key, 0) for key in EVENT_STATS.values()})
        run["prompt_tokens"] = counter.get("prompt_tokens", 0)
        run["cache_hit_tokens"] = counter.get("cache_hit_tokens", 0)
        run["cache_hit_rate"] = (
            round(counter["cache_hit_tokens"] / counter["prompt_tokens"], 3)
            if counter.get("prompt_tokens") else 0.0
        )
        run["outcome"] = outcome_of(run)


def render_markdown(runs: list[dict]) -> str:
    lines = [
        "# AI 战绩统计",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} · 来源 `recordings/trajectories.sqlite3` · advisor 退出时自动更新（也可 `python tools/report_runs.py` 手动重建）",
        "",
        "| # | 日期 | 角色 | 进阶 | 最高层 | 幕 | 结局 | 终局HP | 决策 | 模型调用 | 计划复用 | 确定性 | Prompt tokens | 命中率 |",
        "|---|------|------|------|--------|----|------|--------|------|----------|----------|--------|---------------|--------|",
    ]
    for index, run in enumerate(runs, 1):
        hp = f"{run['final_hp']}/{run['max_hp']}" if run["final_hp"] is not None else "-"
        lines.append(
            f"| {index} | {run['started_at'][:10]} | {run['character']} | {run['ascension']} "
            f"| {run['max_floor']} | {run['max_act'] or '-'} | {OUTCOME_LABEL[run['outcome']]} | {hp} "
            f"| {run['decisions']} | {run['model_calls']} | {run['plan_reuses']} | {run['deterministic']} "
            f"| {run['prompt_tokens']:,} | {run['cache_hit_rate']:.0%} |"
        )
    lines += ["", "## 按角色汇总", "",
              "| 角色 | 局数 | 最佳层数 | 平均层数 | 胜场 | 局均决策 | 局均 tokens |",
              "|------|------|----------|----------|------|----------|-------------|"]
    by_character: dict[str, list[dict]] = {}
    for run in runs:
        by_character.setdefault(run["character"], []).append(run)
    for character, items in sorted(by_character.items()):
        floors = [item["max_floor"] for item in items]
        lines.append(
            f"| {character} | {len(items)} | {max(floors)} | {sum(floors) / len(floors):.1f} "
            f"| {sum(1 for i in items if i['outcome'] == 'victory')} "
            f"| {sum(i['decisions'] for i in items) / len(items):.0f} "
            f"| {sum(i['prompt_tokens'] for i in items) / len(items):,.0f} |"
        )
    total_tokens = sum(run["prompt_tokens"] for run in runs)
    lines += [
        "",
        f"共 {len(runs)} 局 · 累计 prompt {total_tokens:,} tokens。结局判定：hp=0 死亡，hp>0 推断胜利，无 game_over 记录为中途停止。",
        "",
    ]
    return "\n".join(lines)


def refresh(
    db: Path = DEFAULT_DB, json_out: Path = DEFAULT_JSON, md_out: Path = DEFAULT_MD,
) -> int:
    """重建报表；返回局数。供 advisor 退出钩子与 tools 脚本共用。"""
    runs = split_runs(load_observations(db))
    attach_events(runs, load_events(db))

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(runs), encoding="utf-8")
    return len(runs)
