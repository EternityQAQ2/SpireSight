"""Read-only smoke tests for the local STS2_MCP HTTP API.

This script never sends POST requests and therefore cannot perform game actions.
Run it while Slay the Spire 2 and the STS2_MCP mod are running:

    python test_api.py
    python test_api.py --wiki-query "Perfected Strike"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://localhost:15526"
EXPECTED_STATE_TYPES = {
    "menu",
    "unknown",
    "monster",
    "elite",
    "boss",
    "hand_select",
    "rewards",
    "card_reward",
    "map",
    "event",
    "rest_site",
    "shop",
    "fake_merchant",
    "treasure",
    "card_select",
    "bundle_select",
    "relic_select",
    "crystal_sphere",
    "game_over",
    "overlay",
}


class ApiTestError(RuntimeError):
    """Raised when an endpoint violates the documented API contract."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    content_type: str
    body: bytes
    elapsed_ms: float

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            preview = self.body[:200].decode("utf-8", errors="replace")
            raise ApiTestError(f"response is not valid UTF-8 JSON: {preview!r}") from exc


@dataclass(frozen=True)
class CheckResult:
    name: str
    outcome: str
    detail: str
    elapsed_ms: float = 0.0
    response_body: str | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ApiTestError(message)


def require_object(value: Any, context: str = "response") -> dict[str, Any]:
    require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def response_error_detail(response: HttpResponse) -> str:
    try:
        payload = response.json()
    except ApiTestError:
        return response.body[:300].decode("utf-8", errors="replace").strip()

    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("message")
        if detail:
            return str(detail)
    return json.dumps(payload, ensure_ascii=False)[:300]


def format_response_body(response: HttpResponse) -> str:
    try:
        return json.dumps(response.json(), ensure_ascii=False, indent=2)
    except ApiTestError:
        return response.body.decode("utf-8", errors="replace")


class Sts2ApiClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def get(self, path: str, params: dict[str, Any] | None = None) -> HttpResponse:
        relative_path = path.lstrip("/")
        url = urljoin(self.base_url, relative_path)
        if params:
            url = f"{url}?{urlencode(params)}"

        request = Request(
            url,
            method="GET",
            headers={"Accept": "application/json", "User-Agent": "sts2-api-smoke-test/1.0"},
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                status = response.status
                content_type = response.headers.get_content_type()
        except HTTPError as exc:
            body = exc.read()
            status = exc.code
            content_type = exc.headers.get_content_type()
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ApiTestError(
                f"cannot connect to {self.base_url} ({reason}); start the game and STS2_MCP mod"
            ) from exc
        except TimeoutError as exc:
            raise ApiTestError(f"request timed out after {self.timeout:g}s") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        return HttpResponse(status, content_type, body, elapsed_ms)


def validate_game_state(payload: Any) -> str:
    data = require_object(payload)
    state_type = data.get("state_type")
    require(isinstance(state_type, str), "missing string field: state_type")
    require(
        state_type in EXPECTED_STATE_TYPES,
        f"undocumented state_type {state_type!r}; update the test if the API changed",
    )

    if state_type != "menu":
        if state_type not in {"unknown"}:
            require(isinstance(data.get("run"), dict), "non-menu response is missing run object")
            require(isinstance(data.get("player"), dict), "non-menu response is missing player object")
    else:
        require(isinstance(data.get("menu_screen"), str), "menu response is missing menu_screen")

    return f"state_type={state_type}"


def validate_profile(payload: Any) -> str:
    data = require_object(payload)
    require(isinstance(data.get("characters"), list), "missing array field: characters")
    require(isinstance(data.get("discovered_cards"), list), "missing array field: discovered_cards")
    require(isinstance(data.get("discovered_relics"), list), "missing array field: discovered_relics")
    require(isinstance(data.get("total_wins"), int), "missing integer field: total_wins")
    require(isinstance(data.get("total_losses"), int), "missing integer field: total_losses")
    return (
        f"characters={len(data['characters'])}, "
        f"cards={len(data['discovered_cards'])}, relics={len(data['discovered_relics'])}"
    )


def validate_compendium(payload: Any) -> str:
    data = require_object(payload)
    require(isinstance(data.get("profile_id"), int), "missing integer field: profile_id")
    sections = data.get("sections")
    require(isinstance(sections, dict), "missing object field: sections")
    expected_sections = {
        "card_library",
        "relic_collection",
        "potion_lab",
        "bestiary",
        "character_stats",
        "run_history",
    }
    missing = expected_sections - sections.keys()
    require(not missing, f"compendium is missing sections: {sorted(missing)}")
    return f"profile_id={data['profile_id']}, sections={len(sections)}"


def validate_profiles(payload: Any) -> str:
    data = require_object(payload)
    current_profile_id = data.get("current_profile_id")
    profiles = data.get("profiles")
    require(isinstance(current_profile_id, int), "missing integer field: current_profile_id")
    require(isinstance(profiles, list), "missing array field: profiles")
    require(len(profiles) == 3, f"expected 3 profile slots, received {len(profiles)}")

    for index, profile in enumerate(profiles):
        profile = require_object(profile, f"profiles[{index}]")
        require(profile.get("id") in {1, 2, 3}, f"profiles[{index}].id must be 1, 2, or 3")
        require(isinstance(profile.get("is_current"), bool), f"profiles[{index}].is_current must be bool")
        require(isinstance(profile.get("has_data"), bool), f"profiles[{index}].has_data must be bool")

    require(
        sum(profile["is_current"] for profile in profiles) == 1,
        "exactly one profile must be current",
    )
    return f"current_profile_id={current_profile_id}"


def validate_wiki(payload: Any) -> str:
    data = require_object(payload)
    require(data.get("status") == "ok", "wiki status must be 'ok'")
    require(isinstance(data.get("query"), str), "wiki response is missing query")
    require(isinstance(data.get("results"), list), "wiki response is missing results array")
    require(isinstance(data.get("counts"), dict), "wiki response is missing counts object")
    return f"results={len(data['results'])}"


def run_json_check(
    client: Sts2ApiClient,
    name: str,
    path: str,
    validator: Callable[[Any], str],
    *,
    params: dict[str, Any] | None = None,
    skip_statuses: set[int] | None = None,
) -> CheckResult:
    try:
        response = client.get(path, params)
        if skip_statuses and response.status in skip_statuses:
            payload = response.json()
            data = require_object(payload)
            message = data.get("error") or data.get("message") or "endpoint unavailable in current mode"
            return CheckResult(
                name,
                "SKIP",
                f"HTTP {response.status}: {message}",
                response.elapsed_ms,
                format_response_body(response),
            )

        require(
            response.status == 200,
            f"expected HTTP 200, received {response.status}: {response_error_detail(response)}",
        )
        detail = validator(response.json())
        return CheckResult(
            name,
            "PASS",
            detail,
            response.elapsed_ms,
            format_response_body(response),
        )
    except ApiTestError as exc:
        body = format_response_body(response) if "response" in locals() else None
        elapsed_ms = response.elapsed_ms if "response" in locals() else 0.0
        return CheckResult(name, "FAIL", str(exc), elapsed_ms, body)
    except Exception as exc:  # Keep smoke-test output concise for unexpected failures.
        return CheckResult(name, "FAIL", f"{type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-destructive smoke tests against the local STS2_MCP API."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-request timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--wiki-query",
        default="strike",
        help="Query used for the read-only wiki endpoint (default: strike)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2

    client = Sts2ApiClient(args.base_url, args.timeout)
    checks = [
        ("singleplayer", "/api/v1/singleplayer", validate_game_state, None, {409}),
        ("multiplayer", "/api/v1/multiplayer", validate_game_state, None, {409}),
        ("profile", "/api/v1/profile", validate_profile, None, None),
        ("compendium", "/api/v1/compendium", validate_compendium, None, None),
        ("profiles", "/api/v1/profiles", validate_profiles, None, None),
        (
            "wiki",
            "/api/v1/wiki",
            validate_wiki,
            {"query": args.wiki_query, "item_type": "all", "limit": 3},
            None,
        ),
    ]

    print(f"STS2_MCP read-only API test: {client.base_url}")
    results: list[CheckResult] = []
    for name, path, validator, params, skip_statuses in checks:
        result = run_json_check(
            client,
            name,
            path,
            validator,
            params=params,
            skip_statuses=skip_statuses,
        )
        results.append(result)
        timing = f" ({result.elapsed_ms:.0f} ms)" if result.elapsed_ms else ""
        print(f"[{result.outcome:4}] {result.name:<13} {result.detail}{timing}")
        if result.response_body is not None:
            print(result.response_body)
        print()

    passed = sum(result.outcome == "PASS" for result in results)
    skipped = sum(result.outcome == "SKIP" for result in results)
    failed = sum(result.outcome == "FAIL" for result in results)
    print(f"\nSummary: {passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
