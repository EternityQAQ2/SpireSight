from __future__ import annotations
'''
只读访问 STS2MCP /singleplayer
'''
import hashlib
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .record import record_event


class Sts2ApiError(RuntimeError):
    pass


class Sts2ActionRejectedError(Sts2ApiError):
    """服务器已处理请求并明确拒绝该动作（status != ok），动作未执行。"""


class Sts2RestClient:
    def __init__(self, base_url: str = "http://localhost:15526", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._last_response_digest: dict[str, str] = {}

    def get_singleplayer_state(self) -> dict[str, Any]:
        return self._get_json("/api/v1/singleplayer")

    def get_profile(self) -> dict[str, Any]:
        return self._get_json("/api/v1/profile")

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            method="GET",
            headers={"Accept": "application/json", "User-Agent": "sts2-advisor/0.1"},
        )
        request_id = str(uuid4())
        record_event("environment.request", request_id=request_id, method="GET", endpoint=path)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                status = response.status
                # 轮询期间同一端点常返回逐字节相同的响应；重复响应不再落库，
                # 变化后的首条仍完整保留（environment.observation 侧同理去重）。
                digest = hashlib.sha256(raw).hexdigest()
                previous = self._last_response_digest.get(path)
                self._last_response_digest[path] = digest
                if digest != previous:
                    record_event("environment.response", request_id=request_id,
                                 http_status=status, raw_body=raw.decode("utf-8", errors="replace"))
        except HTTPError as exc:
            raw = exc.read()
            record_event("environment.response", request_id=request_id, http_status=exc.code,
                         raw_body=raw.decode("utf-8", errors="replace"))
            message = self._error_message(raw) or str(exc)
            raise Sts2ApiError(f"STS2MCP HTTP {exc.code}: {message}") from exc
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            record_event("environment.transport_error", request_id=request_id,
                         error_type=type(exc).__name__, message=str(reason))
            raise Sts2ApiError(
                f"无法连接 STS2MCP ({reason})。请确认游戏、模组和 localhost:15526 已启动。"
            ) from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Sts2ApiError("STS2MCP 返回了无效 JSON") from exc
        if not isinstance(payload, dict):
            raise Sts2ApiError("STS2MCP 响应必须是 JSON 对象")
        return payload

    @staticmethod
    def _error_message(raw: bytes) -> str:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw[:300].decode("utf-8", errors="replace")
        if isinstance(payload, dict):
            return str(payload.get("error") or payload.get("message") or payload)
        return str(payload)

