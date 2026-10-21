"""Raw, ordered trajectories. No training labels, rewards, or sample filtering.

The optional context-local sink lets model/API/action code report data without
depending on the service or changing their public call signatures.
"""
from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
import time
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .logging_config import get_logger


_current: ContextVar[tuple[TrajectoryRecorder | None, str | None]] = ContextVar(
    "trajectory", default=(None, None)
)
logger = get_logger("record")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    step_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    step_index INTEGER NOT NULL,
    decision_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    UNIQUE(session_id, step_index),
    UNIQUE(session_id, step_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    sequence INTEGER NOT NULL,
    step_id TEXT,
    timestamp TEXT NOT NULL,
    elapsed_ms REAL NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(session_id, sequence),
    FOREIGN KEY(session_id, step_id) REFERENCES steps(session_id, step_id)
);
CREATE INDEX IF NOT EXISTS events_step ON events(step_id, sequence);
CREATE INDEX IF NOT EXISTS events_type ON events(session_id, event_type, sequence);
CREATE INDEX IF NOT EXISTS steps_decision ON steps(decision_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


@contextmanager
def trace_scope(recorder: TrajectoryRecorder | None, step_id: str | None = None):
    token = _current.set((recorder, step_id))
    try:
        yield
    finally:
        _current.reset(token)


def recording_enabled() -> bool:
    recorder, _ = _current.get()
    return recorder is not None and recorder.enabled


def record_event(event_type: str, **payload) -> None:
    """Append raw data to the active trajectory; otherwise do nothing."""
    recorder, step_id = _current.get()
    if recorder is not None:
        recorder.emit(event_type, payload, step_id=step_id)


@contextmanager
def recorded_call(kind: str, **request):
    call_id = str(uuid4())
    started = time.monotonic()
    record_event(f"{kind}.request", call_id=call_id, **request)
    try:
        yield call_id
    except Exception as exc:
        record_event(f"{kind}.error", call_id=call_id, error_type=type(exc).__name__,
                     message=str(exc), duration_ms=(time.monotonic() - started) * 1000)
        raise


class TrajectoryRecorder:
    """Single-threaded session writer; every operation commits independently.

    Open a short-lived connection per write, so legacy callers need not manage
    a persistent SQLite connection. A failed sink is disabled for this session
    and must never cause a model request or game action to be retried.
    """

    def __init__(self, directory: Path, enabled: bool = True, *, metadata: dict | None = None):
        self.enabled = enabled
        self.path = directory / "trajectories.sqlite3"
        self.session_id = str(uuid4())
        self.started_at = utc_now()
        self.metadata = {
            "schema_version": 1, "python": platform.python_version(),
            "platform": platform.platform(), "architecture": platform.machine(),
            **(metadata or {}),
        }
        self._started = time.monotonic()
        self._ready = False
        self._closed = False
        self._sequence = 0
        self._step_index = 0

    def _write(self, sql: str, parameters: tuple = ()) -> None:
        if not self.enabled or self._closed:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=1)) as connection:
                connection.execute("PRAGMA foreign_keys=ON")
                if not self._ready:
                    version = connection.execute("PRAGMA user_version").fetchone()[0]
                    if version not in (0, 1):
                        raise ValueError(f"不支持的轨迹数据库版本：{version}")
                    connection.executescript(SCHEMA)
                    connection.execute("PRAGMA user_version=1")
                    connection.execute(
                        "INSERT INTO sessions VALUES (?, ?, NULL, 'running', ?)",
                        (self.session_id, self.started_at, _json(self.metadata)),
                    )
                connection.execute(sql, parameters)
                connection.commit()
                self._ready = True
        except (OSError, sqlite3.Error, ValueError) as exc:
            self.enabled = False
            logger.warning("轨迹记录已停用，游戏流程继续；本次记录可能不完整：%s", exc)

    def emit(self, event_type: str, payload: dict, *, step_id: str | None = None) -> None:
        if not self.enabled or self._closed:
            return
        self._sequence += 1
        self._write(
            "INSERT INTO events(session_id, sequence, step_id, timestamp, elapsed_ms, event_type, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.session_id, self._sequence, step_id, utc_now(),
             (time.monotonic() - self._started) * 1000, event_type, _json(payload)),
        )

    @property
    def current_step_id(self) -> str | None:
        recorder, step_id = _current.get()
        return step_id if recorder is self else None

    @contextmanager
    def step(self, raw_state: dict, context):
        step_id = str(uuid4())
        self._step_index += 1
        self._write(
            "INSERT INTO steps VALUES (?, ?, ?, ?, ?, NULL, 'running')",
            (step_id, self.session_id, self._step_index, context.decision_id, utc_now()),
        )
        status = "completed"
        with trace_scope(self, step_id):
            record_event("decision.input", raw_state=raw_state, context=context.to_dict())
            try:
                yield step_id
            except BaseException as exc:
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
                record_event("decision.error", error_type=type(exc).__name__, message=str(exc))
                raise
            finally:
                self._write("UPDATE steps SET ended_at=?, status=? WHERE step_id=?",
                            (utc_now(), status, step_id))

    def close(self, status: str = "completed") -> None:
        if self._ready and not self._closed:
            self.emit("session.end", {"status": status})
            self._write("UPDATE sessions SET ended_at=?, status=? WHERE session_id=?",
                        (utc_now(), status, self.session_id))
        self._closed = True


def iter_events(database: Path, session_id: str) -> Iterator[dict]:
    """Read-only, chronological raw events; does not send game actions."""
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(
            "SELECT e.*, s.step_index FROM events e LEFT JOIN steps s ON s.step_id=e.step_id "
            "WHERE e.session_id=? ORDER BY e.sequence", (session_id,),
        ):
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            yield event


def export_session(database: Path, session_id: str, output) -> None:
    """Export session metadata, step manifest and events as UTF-8 JSONL."""
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        session = connection.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if session is None:
            raise ValueError(f"没有找到会话：{session_id}")
        header = dict(session)
        header["metadata"] = json.loads(header.pop("metadata_json"))
        header["steps"] = [dict(row) for row in connection.execute(
            "SELECT * FROM steps WHERE session_id=? ORDER BY step_index", (session_id,),
        )]
    output.write(_json({"record_type": "session", **header}) + "\n")
    for event in iter_events(database, session_id):
        output.write(_json({"record_type": "event", **event}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查看原始轨迹会话，或按时间顺序导出 JSONL")
    parser.add_argument("--db", type=Path, default=Path("recordings/trajectories.sqlite3"))
    parser.add_argument("--session", help="会话 UUID；省略时列出所有会话")
    parser.add_argument("--output", type=Path, help="导出文件；省略时输出到终端")
    args = parser.parse_args(argv)
    if args.output and not args.session:
        parser.error("--output 需要搭配 --session")
    try:
        if args.session:
            if args.output and args.output.resolve() == args.db.resolve():
                raise ValueError("导出文件不能覆盖轨迹数据库")
            if args.output:
                with args.output.open("w", encoding="utf-8") as output:
                    export_session(args.db, args.session, output)
            else:
                export_session(args.db, args.session, sys.stdout)
        else:
            with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                for row in connection.execute("SELECT * FROM sessions ORDER BY started_at, session_id"):
                    session = dict(row)
                    session["metadata"] = json.loads(session.pop("metadata_json"))
                    print(_json(session))
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"无法读取轨迹：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
