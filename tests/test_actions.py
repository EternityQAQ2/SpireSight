from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from collections import deque
from contextlib import contextmanager
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from sts2_advisor.action import ActionExecutor, action
from sts2_advisor.api import Sts2ApiError, Sts2RestClient
from sts2_advisor.domain import Advice
from sts2_advisor.model import RuleAdvisor
from sts2_advisor.recording import SessionRecorder
from sts2_advisor.retrieval import NullRetriever
from sts2_advisor.service import AdvisorService
from sts2_advisor.normalize import build_decision_context
from sts2_advisor.validation import AdviceValidationError


FIXTURE = Path(__file__).parent / "fixtures/combat_state.json"
COMBAT = json.loads(FIXTURE.read_text(encoding="utf-8"))
MAP = {"state_type": "map", "map": {"next_options": [{"index": 3, "type": "rest"}]}}


@contextmanager
def action_api(states=None, *, post_body=b'{"status":"ok","message":"done"}', post_status=200):
    states = deque(states or [COMBAT])
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path, None))
            state = states.popleft() if len(states) > 1 else states[0]
            status, body = state if isinstance(state, tuple) else (200, state)
            self.respond(status, json.dumps(body).encode("utf-8"))

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(("POST", self.path, json.loads(body)))
            if self.headers.get_content_type() != "application/json":
                self.respond(415, b'{"error":"expected JSON"}')
                return
            self.respond(post_status, post_body)

        def respond(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield Sts2RestClient(f"http://127.0.0.1:{server.server_port}"), requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class ActionApiTests(unittest.TestCase):
    def test_posts_flat_action_parameters_to_singleplayer(self):
        cases = [
            ("play_card", {"card_index": 0, "target": "NIBBIT_0"},
             {"action": "play_card", "card_index": 0, "target": "NIBBIT_0"}),
            ("use_potion", {"slot": 0}, {"action": "use_potion", "slot": 0}),
            ("crystal_sphere_click_cell", {"x": 4, "y": 7},
             {"action": "crystal_sphere_click_cell", "x": 4, "y": 7}),
            ("end_turn", {}, {"action": "end_turn"}),
        ]
        for action_type, params, expected in cases:
            with self.subTest(action=action_type), action_api() as (client, requests):
                result = action(action_type, client=client, **params)
                self.assertEqual(result, {"status": "ok", "message": "done"})
                self.assertEqual(requests, [("POST", "/api/v1/singleplayer", expected)])

    def test_rejects_http_business_and_malformed_errors_without_retry(self):
        cases = [
            (400, b'{"status":"error","error":"invalid index"}', "invalid index"),
            (200, b'{"status":"error","error":"not play phase"}', "not play phase"),
            (200, b'not json', "无效 JSON"),
            (200, b'[]', "JSON 对象"),
            (200, b'{}', "status"),
        ]
        for status, body, message in cases:
            with self.subTest(body=body), action_api(post_status=status, post_body=body) as (client, requests):
                with self.assertRaisesRegex(Sts2ApiError, message):
                    action("end_turn", client=client)
                self.assertEqual(len(requests), 1)

    def test_rejects_empty_action_and_action_override_before_post(self):
        with action_api() as (client, requests):
            with self.assertRaises(ValueError):
                action(" ", client=client)
            with self.assertRaises(ValueError):
                action("end_turn", client=client, action="play_card")
            self.assertEqual(requests, [])

    def test_disconnected_post_is_reported_without_retry(self):
        with patch("sts2_advisor.action.urlopen", side_effect=RemoteDisconnected("response lost")) as send:
            with self.assertRaisesRegex(Sts2ApiError, "response lost"):
                action("end_turn")
            self.assertEqual(send.call_count, 1)


class ActionExecutorTests(unittest.TestCase):
    def test_executor_rejects_invalid_or_stale_advice_without_requests(self):
        context = build_decision_context(COMBAT)
        with action_api() as (client, requests):
            executor = ActionExecutor(client, output=lambda text: None)
            for decision_id, candidate_id in [
                (context.decision_id, "card:999"), ("old-decision", "card:2"),
            ]:
                with self.subTest(candidate=candidate_id, decision=decision_id):
                    with self.assertRaises(AdviceValidationError):
                        executor.execute(Advice(decision_id, candidate_id, "test", 1.0), context)
            self.assertEqual(requests, [])

    def test_executor_deduplicates_without_service(self):
        context = build_decision_context(COMBAT)
        advice = Advice(context.decision_id, "card:2", "test", 1.0)
        with action_api() as (client, requests):
            executor = ActionExecutor(client, output=lambda text: None)
            self.assertEqual(executor.execute(advice, context)["status"], "ok")
            self.assertIsNone(executor.execute(advice, context))
            self.assertEqual(requests, [
                ("GET", "/api/v1/singleplayer", None),
                ("POST", "/api/v1/singleplayer", {"action": "play_card", "card_index": 2}),
            ])


class AutomaticActionTests(unittest.TestCase):
    def service(self, client, *, auto_act=True):
        return AdvisorService(
            client, NullRetriever(), RuleAdvisor(), SessionRecorder(Path("unused"), enabled=False),
            backend_name="rule", auto_act=auto_act, output=lambda text: None,
        )

    def test_executes_recommended_candidate_once_and_waits_for_state_change(self):
        with action_api() as (client, requests):
            service = self.service(client)
            self.assertTrue(service.run_once())
            self.assertFalse(service.run_once())
            self.assertEqual(requests, [
                ("GET", "/api/v1/singleplayer", None),
                ("GET", "/api/v1/singleplayer", None),
                ("POST", "/api/v1/singleplayer", {"action": "play_card", "card_index": 2}),
                ("GET", "/api/v1/singleplayer", None),
            ])

    def test_read_only_mode_never_posts(self):
        with action_api() as (client, requests):
            self.service(client, auto_act=False).run_once()
            self.assertEqual(requests, [("GET", "/api/v1/singleplayer", None)])

    def test_skips_stale_recommendation_and_uses_new_state(self):
        with action_api([COMBAT, MAP]) as (client, requests):
            service = self.service(client)
            service.run_once()
            self.assertTrue(all(method == "GET" for method, _, _ in requests))
            service.run_once()
            self.assertEqual([body for method, _, body in requests if method == "POST"], [
                {"action": "choose_map_node", "index": 3},
            ])

    def test_watch_executes_next_decision_after_stable_reads(self):
        states = [COMBAT, COMBAT, COMBAT, MAP, MAP, MAP, {"state_type": "menu"}]
        with action_api(states) as (client, requests):
            service = self.service(client)
            with patch("sts2_advisor.service.time.sleep", side_effect=[None] * 7 + [KeyboardInterrupt]):
                with self.assertRaises(KeyboardInterrupt):
                    service.watch(0.01, stable_reads=2)
            self.assertEqual([body for method, _, body in requests if method == "POST"], [
                {"action": "play_card", "card_index": 2},
                {"action": "choose_map_node", "index": 3},
            ])

    def test_failed_post_is_not_repeated_for_unchanged_state(self):
        with action_api(post_body=b'{"status":"error","error":"not play phase"}') as (client, requests):
            service = self.service(client)
            with self.assertRaises(Sts2ApiError):
                service.run_once()
            self.assertFalse(service.run_once())
            self.assertEqual(sum(method == "POST" for method, _, _ in requests), 1)

    def test_preflight_get_failure_does_not_suppress_unexecuted_action(self):
        with action_api([COMBAT, (503, {"error": "temporarily unavailable"}), COMBAT]) as (client, requests):
            service = self.service(client)
            with self.assertRaises(Sts2ApiError):
                service.run_once()
            self.assertTrue(service.run_once())
            self.assertEqual([body for method, _, body in requests if method == "POST"], [
                {"action": "play_card", "card_index": 2},
            ])


class AutomaticActionCliTests(unittest.TestCase):
    def run_cli(self, url, *args):
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            [sys.executable, str(root / "main.py"), "--base-url", url,
             "--rag", "none", "--no-record", *args],
            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )

    def test_auto_act_posts_recommended_action(self):
        with action_api() as (client, requests):
            result = self.run_cli(client.base_url, "--auto-act")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("已发送动作", result.stdout)
        self.assertEqual(requests, [
            ("GET", "/api/v1/singleplayer", None),
            ("GET", "/api/v1/singleplayer", None),
            ("POST", "/api/v1/singleplayer", {"action": "play_card", "card_index": 2}),
        ])

    def test_auto_act_reports_rejected_action(self):
        with action_api(post_body=b'{"status":"error","error":"not play phase"}') as (client, requests):
            result = self.run_cli(client.base_url, "--auto-act")
        self.assertEqual(result.returncode, 1)
        self.assertIn("not play phase", result.stderr)
        self.assertNotIn("已发送动作", result.stdout)
        self.assertEqual(sum(item[0] == "POST" for item in requests), 1)

    def test_dump_context_with_auto_act_does_not_execute(self):
        with action_api() as (client, requests):
            result = self.run_cli(client.base_url, "--dump-context", "--auto-act")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["decision_type"], "combat_next_action")
        self.assertEqual(requests, [("GET", "/api/v1/singleplayer", None)])


if __name__ == "__main__":
    unittest.main()
