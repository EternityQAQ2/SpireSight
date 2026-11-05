"""开发者模式：按决策阶段输出环境、证据、完整提示词、模型输出与 token 统计。

与 `record.py` 的原始轨迹互补：轨迹面向机器读取（SQLite 事件流），这里面向
人工阅读和提示词调优。两者互不依赖，`--dev` 在 `--no-record` 下同样工作。

输出约定：同一信息只出现一次——完整提示词在「提示词」阶段整段给出，
模型输出阶段不再重复提示词；证据在「检索」阶段完整给出（提示词内嵌的副本
属于模型实际输入，无法省略）。所有写入失败都只会关闭对应输出，
不会中断游戏决策。
"""
from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TextIO

SEPARATOR = "=" * 100
SECTION = "-" * 100

_current: ContextVar["DeveloperTrace | None"] = ContextVar("developer_trace", default=None)


# ---------------------------------------------------------------------------
# 上下文入口：调用方无条件调用，未启用开发者模式时全部是 no-op。
# ---------------------------------------------------------------------------


def developer_enabled() -> bool:
    return _current.get() is not None


def activate(trace: "DeveloperTrace | None"):
    return _current.set(trace)


def deactivate(token) -> None:
    _current.reset(token)


def dev_section(stage: str, title: str = "", *, blocks: dict[str, str] | None = None,
                fields: dict[str, Any] | None = None) -> None:
    trace = _current.get()
    if trace is not None:
        trace.section(stage, title, blocks=blocks, fields=fields)


def dev_block(stage: str, title: str, blocks: dict[str, str], **fields: Any) -> None:
    """输出若干整段原文（提示词、reasoning、模型 content 等）。"""
    trace = _current.get()
    if trace is not None:
        trace.section(stage, title, blocks=blocks, fields=fields)


def evidence_blocks(items: list[dict], *, title: str = "证据") -> dict[str, str]:
    """把证据渲染成可读段落，避免 JSON 把正文换行转义成 \\n。"""
    blocks: dict[str, str] = {}
    for index, item in enumerate(items, 1):
        source = item.get("source_title") or "(无标题)"
        blocks[f"{title} {index} · {source}"] = (
            f"id: {item.get('evidence_id')}\n"
            f"source_url: {item.get('source_url') or '-'}\n"
            f"metadata: {json.dumps(item.get('metadata') or {}, ensure_ascii=False)}\n"
            f"text:\n{item.get('text', '')}"
        )
    return blocks


def dev_request(backend: str, model: str = "", *, attempt: int | None = None,
                **fields: Any) -> None:
    trace = _current.get()
    if trace is not None:
        trace.record_request(backend, model, attempt=attempt, **fields)


def dev_usage(label: str, usage, *, backend: str = "", model: str = "",
              attempt: int | None = None, estimated: bool = False,
              duration_ms: float | None = None, call_id: str | None = None) -> None:
    trace = _current.get()
    if trace is not None:
        trace.record_usage(label, usage, backend=backend, model=model,
                           attempt=attempt, estimated=estimated,
                           duration_ms=duration_ms, call_id=call_id)


@contextmanager
def dev_step(decision_id: str, **meta: Any):
    trace = _current.get()
    if trace is None:
        yield
        return
    with trace.step(decision_id, **meta):
        yield


# ---------------------------------------------------------------------------
# token 统计
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_miss_tokens += other.cache_miss_tokens
        self.reasoning_tokens += other.reasoning_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        if not self.prompt_tokens:
            return None
        return self.cached_tokens / self.prompt_tokens

    def summary(self) -> str:
        parts = [
            f"prompt={self.prompt_tokens}",
            f"completion={self.completion_tokens}",
            f"total={self.total_tokens}",
        ]
        if self.cached_tokens or self.cache_miss_tokens:
            parts.append(f"cached={self.cached_tokens}")
            if self.cache_hit_rate is not None:
                parts.append(f"命中率={self.cache_hit_rate:.1%}")
            if self.cache_miss_tokens:
                parts.append(f"cache_miss={self.cache_miss_tokens}")
        if self.reasoning_tokens:
            parts.append(f"reasoning={self.reasoning_tokens}")
        return " ".join(parts)


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_usage(usage) -> TokenUsage | None:
    """兼容 OpenAI 标准字段与 DeepSeek 等兼容服务的缓存字段。"""
    if isinstance(usage, TokenUsage):
        return usage
    if not isinstance(usage, dict) or not usage:
        return None
    prompt = _int(usage.get("prompt_tokens"))
    completion = _int(usage.get("completion_tokens"))
    total = _int(usage.get("total_tokens")) or prompt + completion
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cached = _int(usage.get("prompt_cache_hit_tokens")) or _int(
        prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    )
    miss = _int(usage.get("prompt_cache_miss_tokens"))
    reasoning = _int(
        completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None
    )
    return TokenUsage(
        prompt_tokens=prompt, completion_tokens=completion,
        total_tokens=total, cached_tokens=cached, cache_miss_tokens=miss,
        reasoning_tokens=reasoning,
    )


# ---------------------------------------------------------------------------
# 环境信息
# ---------------------------------------------------------------------------


def package_versions(names: tuple[str, ...] = (
    "openai", "torch", "transformers", "chromadb", "sentence-transformers",
)) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def runtime_environment(**extra: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "Python": sys.version.split()[0],
        "平台": platform.platform(),
        "架构": platform.machine(),
        "解释器": sys.executable,
        "工作目录": str(Path.cwd()),
        "依赖版本": package_versions(),
    }
    info.update(extra)
    return info


# ---------------------------------------------------------------------------
# 跟踪器
# ---------------------------------------------------------------------------


class DeveloperTrace:
    """单线程顺序写入；每个事件独立追加，避免长会话占用文件句柄。"""

    def __init__(self, directory: Path, *, echo: bool = True, stream: TextIO | None = None,
                 path: Path | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = path or directory / f"dev-{stamp}.log"
        self.echo = echo
        self.stream = stream if stream is not None else sys.stdout
        self.started = time.monotonic()
        self.step_index = 0
        self.requests = 0
        self.usage = TokenUsage()
        self._step_usage = TokenUsage()
        self._step_requests = 0
        self._file_error: str | None = None
        self._closed = False

    # -- 写入 -----------------------------------------------------------------

    def _write(self, text: str) -> None:
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(text)
            except OSError as exc:
                self._file_error = str(exc)
                self.path = None
        if self.echo:
            try:
                self.stream.write(text)
                self.stream.flush()
            except (OSError, ValueError):
                self.echo = False

    def _line(self, text: str = "") -> None:
        self._write(text + "\n")

    def start(self, environment: dict[str, Any]) -> None:
        self._line(SEPARATOR)
        self._line("STS2 Advisor 开发者模式跟踪")
        self._line(SEPARATOR)
        self._emit_fields("环境信息", environment)
        if self._file_error:
            self._line(f"警告：开发者日志文件不可写：{self._file_error}")
        self._line()

    def section(self, stage: str, title: str = "", *, blocks: dict[str, str] | None = None,
                fields: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        header = f"[{self._now()}] {stage}"
        if title:
            header += f" · {title}"
        self._line(SECTION)
        self._line(header)
        for key, value in (fields or {}).items():
            if value is None:
                continue
            rendered = _format_value(value)
            self._line(f"{key}: {rendered}" if "\n" not in rendered else f"{key}:\n{rendered}")
        for label, content in (blocks or {}).items():
            self._line(f"----- {label} -----")
            self._line(str(content).rstrip("\n"))
            self._line(f"----- /{label} -----")

    def _emit_fields(self, title: str, fields: dict[str, Any]) -> None:
        self._line(f"[{title}]")
        for key, value in fields.items():
            rendered = _format_value(value)
            if "\n" in rendered:
                self._line(f"{key}:")
                self._line(rendered)
            else:
                self._line(f"{key}: {rendered}")

    # -- 步骤 -----------------------------------------------------------------

    @contextmanager
    def step(self, decision_id: str, **meta: Any):
        self.step_index += 1
        number = self.step_index
        started = time.monotonic()
        previous = self._step_usage
        previous_requests = self._step_requests
        self._step_usage = TokenUsage()
        self._step_requests = 0
        try:
            self.section(
                f"步骤 {number} 开始",
                fields={"decision_id": decision_id, **meta},
            )
            status = "completed"
            try:
                yield
            except BaseException as exc:  # noqa: BLE001 - 记录后原样抛出
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
                self.section(
                    "步骤异常", f"步骤 {number}", fields={
                        "error_type": type(exc).__name__, "message": str(exc),
                    },
                )
                raise
            finally:
                self.section(
                    f"步骤 {number} 结束",
                    fields={
                        "status": status,
                        "耗时ms": round((time.monotonic() - started) * 1000, 1),
                        "本步骤模型请求": self._step_requests,
                        "本步骤token": self._step_usage.summary(),
                        "会话token": self.usage.summary(),
                    },
                )
                self._line()
        finally:
            self._step_usage = previous
            self._step_requests = previous_requests

    # -- token ----------------------------------------------------------------

    def record_request(self, backend: str, model: str = "", *,
                       attempt: int | None = None, **fields: Any) -> None:
        self.requests += 1
        self._step_requests += 1
        detail = [part for part in (backend, model) if part]
        if attempt is not None:
            detail.append(f"第{attempt}次")
        self.section(
            "模型请求", "，".join(detail) or "模型",
            fields={"backend": backend, "model": model, "attempt": attempt, **fields},
        )

    def record_usage(self, label: str, usage, *, backend: str = "", model: str = "",
                     attempt: int | None = None, estimated: bool = False,
                     duration_ms: float | None = None,
                     call_id: str | None = None) -> None:
        stats = normalize_usage(usage)
        if stats is not None:
            self.usage.add(stats)
            self._step_usage.add(stats)

        detail = [part for part in (backend, model) if part]
        if attempt is not None:
            detail.append(f"第{attempt}次")
        prefix = "，".join(detail)
        header = f"[Token] {label}" + (f"（{prefix}）" if prefix else "")
        if stats is None:
            line = f"{header} 服务未返回 usage，无法统计 token。"
        else:
            line = f"{header} {stats.summary()}"
            if estimated:
                line += "（估算：tokenizer 计数）"
            if duration_ms is not None:
                line += f" 耗时={duration_ms:.0f}ms"
            if call_id:
                line += f" call_id={call_id}"
        self._line(line)

    # -- 汇总 -----------------------------------------------------------------

    def finish(self, status: str = "completed", **extra: Any) -> None:
        if self._closed:
            return
        self._closed = True
        elapsed = time.monotonic() - self.started
        self._line(SEPARATOR)
        self._line(f"[会话汇总] status={status}")
        self._line(f"决策步骤: {self.step_index}")
        self._line(f"模型请求: {self.requests}")
        self._line(f"输入 token: {self.usage.prompt_tokens}")
        if self.usage.cached_tokens or self.usage.cache_miss_tokens:
            rate = self.usage.cache_hit_rate
            rate_text = f"（{rate:.1%}）" if rate is not None else ""
            self._line(f"  其中缓存命中: {self.usage.cached_tokens}{rate_text}")
        self._line(f"输出 token: {self.usage.completion_tokens}")
        if self.usage.reasoning_tokens:
            self._line(f"  其中推理 token: {self.usage.reasoning_tokens}")
        self._line(f"总 token: {self.usage.total_tokens}")
        if self.requests:
            self._line(
                f"平均每次请求: {self.usage.total_tokens / self.requests:.1f} token"
            )
        if self.step_index:
            self._line(
                f"平均每个决策步骤: {self.usage.total_tokens / self.step_index:.1f} token"
            )
        self._line(f"会话总耗时: {elapsed:.1f}s")
        for key, value in extra.items():
            self._line(f"{key}: {_format_value(value)}")
        if self.path is not None:
            self._line(f"开发者日志: {self.path}")
        self._line(SEPARATOR)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            return repr(value)
    return str(value)
