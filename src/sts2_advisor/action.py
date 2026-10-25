"""游戏动作的发送与推荐执行，协议参照 docs/api.md。

直接调用：action("play_card", card_index=0, target="NIBBIT_0")。
自动执行：main.py --watch --auto-act --backend transformers。
修改动作协议、执行前检查或防重复策略时，统一在本模块调整。
"""
from __future__ import annotations

import json
from http.client import HTTPException
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from .api import Sts2ActionRejectedError, Sts2ApiError, Sts2RestClient
from .domain import Advice, DecisionContext
from .normalize import build_decision_context
from .record import record_event
from .validation import AdviceValidationError


# 纯界面导航类动作，重复发送最坏只得到一次无害报错，可在被明确拒绝后安全重试一次；
# 购买、出牌等改变游戏状态的动作一旦发出绝不重发。
IDEMPOTENT_RETRY_ACTION_TYPES = {"proceed", "end_turn"}


def action(
    action_type: str, *, client: Sts2RestClient | None = None, **params: Any,
) -> dict[str, Any]:
    """发送一次单人游戏动作，返回 status=ok 的响应，否则抛出 Sts2ApiError。

    client 可复用现有只读客户端的地址和超时配置；省略时连接 localhost:15526。
    参数按 API 原样展开，如 card_index、target、slot、index、x、y。
    本函数不检查局面或重试；自动执行推荐时使用 ActionExecutor。
    """
    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("action_type 必须是非空字符串")
    if "action" in params:
        raise ValueError("请通过 action_type 指定动作，params 不得覆盖 action")
    if client is None:
        client = Sts2RestClient()
    request = Request(
        f"{client.base_url}/api/v1/singleplayer",
        data=json.dumps({"action": action_type, **params}, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "sts2-advisor/0.1",
        },
    )
    action_id = str(uuid4())
    record_event("action.request", action_id=action_id, method="POST",
                 endpoint="/api/v1/singleplayer", request={"action": action_type, **params})
    try:
        with urlopen(request, timeout=client.timeout) as response:
            raw = response.read()
            status = response.status
    except HTTPError as exc:
        raw = exc.read()
        record_event("action.error", action_id=action_id, outcome="http_error", http_status=exc.code,
                     raw_body=raw.decode("utf-8", errors="replace"), message=str(exc))
        message = Sts2RestClient._error_message(raw) or str(exc)
        raise Sts2ApiError(f"STS2MCP HTTP {exc.code}: {message}") from exc
    except (OSError, HTTPException) as exc:
        reason = getattr(exc, "reason", exc)
        record_event("action.error", action_id=action_id, outcome="unknown",
                     error_type=type(exc).__name__, message=str(reason))
        raise Sts2ApiError(
            f"STS2MCP 动作请求失败 ({reason})，结果未确认，请先核对游戏状态。"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        record_event("action.error", action_id=action_id, outcome="unknown", http_status=status,
                     raw_body=raw.decode("utf-8", errors="replace"), message=str(exc))
        raise Sts2ApiError("STS2MCP 返回了无效 JSON") from exc
    record_event("action.response", action_id=action_id, http_status=status, response=payload,
                 raw_body=raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise Sts2ApiError("STS2MCP 响应必须是 JSON 对象")
    if payload.get("status") != "ok":
        message = payload.get("error") or payload.get("message") or "响应缺少有效的 status: ok"
        raise Sts2ActionRejectedError(f"STS2MCP 动作 {action_type} 未确认成功：{message}")
    return payload


class ActionExecutor:
    """把推荐候选转成动作；每个稳定决策最多尝试发送一次。"""

    def __init__(self, client: Sts2RestClient, *, output: Callable[[str], None] = print) -> None:
        self.client = client
        self.output = output
        self.last_decision_id: str | None = None
        self._retried_decision_id: str | None = None

    def execute(self, advice: Advice, context: DecisionContext) -> dict[str, Any] | None:
        """校验推荐与最新局面后发送；重复或过期局面返回 None。"""
        if advice.decision_id != context.decision_id:
            raise AdviceValidationError("不能执行过期或错误 decision_id 的推荐")
        candidate = next(
            (item for item in context.candidates if item.candidate_id == advice.candidate_id), None
        )
        if candidate is None or not candidate.legal:
            raise AdviceValidationError(f"不能执行非法候选：{advice.candidate_id}")
        if context.decision_id == self.last_decision_id:
            record_event("action.skipped", reason="duplicate", decision_id=context.decision_id)
            return None
        latest_raw = self.client.get_singleplayer_state()
        record_event("action.precheck", raw_state=latest_raw)
        latest = build_decision_context(latest_raw)
        if latest is None or latest.decision_id != context.decision_id:
            record_event("action.skipped", reason="stale", decision_id=context.decision_id,
                         latest_decision_id=latest.decision_id if latest else None)
            self.output("游戏状态已变化，跳过过期动作，等待下一次决策。")
            return None

        # 预检 GET 失败可重试；POST 已发出后，即使响应丢失也不对同一状态重发。
        # 唯一例外：服务器明确拒绝（动作未执行）且动作幂等时，允许重试一次，
        # 避免界面类故障（如关闭商店失败）让自动执行永久卡死在同一局面。
        self.last_decision_id = context.decision_id
        try:
            result = action(candidate.action_type, client=self.client, **candidate.params)
        except Sts2ActionRejectedError as exc:
            retryable = (
                candidate.action_type in IDEMPOTENT_RETRY_ACTION_TYPES
                and self._retried_decision_id != context.decision_id
            )
            record_event("action.rejected", decision_id=context.decision_id,
                         action_type=candidate.action_type, retryable=retryable, message=str(exc))
            if not retryable:
                raise
            self._retried_decision_id = context.decision_id
            self.last_decision_id = None
            self.output(f"动作 {candidate.candidate_id} 被游戏拒绝：{exc}；幂等动作将在下一次轮询重试一次。")
            return None
        self.output(f"已发送动作 {candidate.candidate_id}：{result.get('message') or 'ok'}")
        return result
