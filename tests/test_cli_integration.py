from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/combat_state.json"


@contextmanager
def game_api(body: bytes, status=200):
    requests = []
    polled = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.command, self.path))
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            if len(requests) >= 6:
                polled.set()

        def do_POST(self):
            requests.append((self.command, self.path))
            self.send_error(405)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, polled
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class CliIntegrationTests(unittest.TestCase):
    def run_cli(self, url, *args, cwd=ROOT):
        return subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--base-url", url, *args],
            cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )

    def test_rule_pipeline_records_valid_chinese_advice_from_other_directory(self):
        with tempfile.TemporaryDirectory(prefix="尖塔 mac test ") as directory:
            with game_api(FIXTURE.read_bytes()) as (url, requests, _):
                result = self.run_cli(
                    url, "--backend", "rule", "--rag", "json",
                    "--recording-dir", str(Path(directory) / "会话 记录"), cwd=directory,
                )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("建议", result.stdout)
            files = list(Path(directory).rglob("*.jsonl"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["advice"]["candidate_id"], "card:2")
            self.assertFalse(record["training_consent"])
            self.assertEqual(requests, [("GET", "/api/v1/singleplayer")])

    def test_dump_context_returns_legal_candidates(self):
        with game_api(FIXTURE.read_bytes()) as (url, requests, _):
            result = self.run_cli(url, "--dump-context")
        self.assertEqual(result.returncode, 0, result.stderr)
        context = json.loads(result.stdout)
        self.assertEqual(context["decision_type"], "combat_next_action")
        self.assertEqual(len(context["candidates"]), 4)
        self.assertEqual(requests, [("GET", "/api/v1/singleplayer")])

    def test_malformed_api_response_exits_cleanly(self):
        with game_api(b"not json") as (url, _, _):
            result = self.run_cli(url, "--no-record")
        self.assertEqual(result.returncode, 1)
        self.assertIn("无效 JSON", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_api_conflict_exits_cleanly(self):
        with game_api(b'{"error":"multiplayer active"}', status=409) as (url, _, _):
            result = self.run_cli(url, "--no-record")
        self.assertEqual(result.returncode, 1)
        self.assertIn("HTTP 409", result.stderr)

    @unittest.skipIf(os.name == "nt", "POSIX SIGINT test")
    def test_watch_deduplicates_and_stops_with_ctrl_c(self):
        with tempfile.TemporaryDirectory() as directory:
            with game_api(FIXTURE.read_bytes()) as (url, requests, polled):
                process = subprocess.Popen(
                    [sys.executable, str(ROOT / "main.py"), "--base-url", url,
                     "--watch", "--poll-interval", "0.05", "--stable-reads", "2",
                     "--rag", "json", "--recording-dir", directory],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8",
                )
                try:
                    self.assertTrue(polled.wait(10), "watch did not poll six times")
                    process.send_signal(signal.SIGINT)
                    stdout, stderr = process.communicate(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("已停止监听", stdout)
            lines = [line for path in Path(directory).glob("*.jsonl")
                     for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 1)
            self.assertTrue(all(item == ("GET", "/api/v1/singleplayer") for item in requests))


if __name__ == "__main__":
    unittest.main()
