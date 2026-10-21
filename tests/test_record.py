from __future__ import annotations

import json
import copy
import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sts2_advisor.api import Sts2ApiError
from sts2_advisor.domain import Advice
from sts2_advisor.model import OpenAIAdvisor, RuleAdvisor
from sts2_advisor.recording import SessionRecorder
from sts2_advisor.retrieval import JsonKnowledgeRetriever, NullRetriever
from sts2_advisor.service import AdvisorService
from test_actions import COMBAT, action_api
from test_cli_integration import game_api
import test_cli_integration as cli_tests


def read_events(directory):
    path = Path(directory) / "trajectories.sqlite3"
    if not path.is_file():
        raise AssertionError("raw trajectory database was not created")
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) | {"payload": json.loads(row["payload_json"])}
                for row in connection.execute("SELECT * FROM events ORDER BY event_id")]


class TrajectoryTests(unittest.TestCase):
    def service(self, directory, client, backend=None, **kwargs):
        recorder = SessionRecorder(Path(directory))
        service = AdvisorService(
            client, NullRetriever(), backend or RuleAdvisor(), recorder,
            backend_name="test", output=lambda _: None, **kwargs,
        )
        return service, recorder

    def test_readonly_records_input_evidence_choice_and_http_without_post(self):
        with tempfile.TemporaryDirectory() as directory:
            with action_api() as (client, requests):
                service, recorder = self.service(directory, client)
                self.assertTrue(service.run_once())
                events = read_events(directory)
                kinds = [row["event_type"] for row in events]
                self.assertIn("environment.observation", kinds)
                self.assertIn("decision.input", kinds)
                self.assertIn("retrieval.result", kinds)
                self.assertIn("decision.output", kinds)
                self.assertNotIn("action.request", kinds)
                self.assertEqual(len(requests), 1)
                decision = next(row for row in events if row["event_type"] == "decision.input")
                self.assertEqual(decision["payload"]["raw_state"], COMBAT)
                self.assertEqual(len(decision["payload"]["context"]["candidates"]), 4)
                recorder.close()
            with closing(sqlite3.connect(Path(directory) / "trajectories.sqlite3")) as connection:
                self.assertEqual(connection.execute("SELECT status FROM steps").fetchone()[0], "completed")
                self.assertEqual(connection.execute("SELECT status FROM sessions").fetchone()[0], "completed")

    def test_action_and_terminal_observation_remain_linked_without_extra_requests(self):
        terminal = {"state_type": "game_over", "result": {"victory": False}}
        with tempfile.TemporaryDirectory() as directory:
            with action_api([COMBAT, COMBAT, terminal]) as (client, requests):
                service, recorder = self.service(directory, client, auto_act=True)
                service.run_once()
                self.assertFalse(service.run_once())
                events = read_events(directory)
                sent = next(row for row in events if row["event_type"] == "action.request")
                response = next(row for row in events if row["event_type"] == "action.response")
                self.assertEqual(sent["step_id"], response["step_id"])
                self.assertEqual(sent["payload"]["action_id"], response["payload"]["action_id"])
                self.assertEqual(response["payload"]["response"]["status"], "ok")
                observations = [row for row in events if row["event_type"] == "environment.observation"]
                self.assertEqual(observations[-1]["payload"]["raw_state"], terminal)
                self.assertEqual(observations[-1]["payload"]["after_step_id"], sent["step_id"])
                self.assertEqual([request[0] for request in requests], ["GET", "GET", "POST", "GET"])
                recorder.close()

    def test_lost_post_response_is_recorded_as_unknown_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            with action_api() as (client, requests):
                service, recorder = self.service(directory, client, auto_act=True)
                with patch("sts2_advisor.action.urlopen", side_effect=RemoteDisconnected("lost")):
                    with self.assertRaises(Sts2ApiError):
                        service.run_once()
                self.assertFalse(service.run_once())
                errors = [row for row in read_events(directory) if row["event_type"] == "action.error"]
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]["payload"]["outcome"], "unknown")
                self.assertIn("lost", errors[0]["payload"]["message"])
                self.assertEqual([request[0] for request in requests], ["GET", "GET", "GET"])
                recorder.close()

    def test_invalid_remote_answer_and_actual_request_survive_rule_fallback(self):
        invalid_text = '{"candidate_id":"does-not-exist","reason":"原始回答","confidence":0.8}'
        response_data = {
            "id": "response-raw", "choices": [{"message": {"content": invalid_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=invalid_text), finish_reason="stop")],
            usage=None, model_dump_json=lambda: json.dumps(response_data, ensure_ascii=False),
        )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **request: response))

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                backend = OpenAIAdvisor("teacher", api_key="do-not-record-me", thinking="disabled")
            with action_api() as (client, _):
                service, recorder = self.service(directory, client, backend, fallback_backend=RuleAdvisor())
                service.run_once()
                events = read_events(directory)
                request = next(row["payload"] for row in events if row["event_type"] == "model.request")
                answer = next(row["payload"] for row in events if row["event_type"] == "model.response")
                self.assertEqual(request["request"]["model"], "teacher")
                self.assertEqual([message["role"] for message in request["request"]["messages"]], ["system", "user"])
                self.assertEqual(answer["content"], invalid_text)
                self.assertEqual(answer["raw_response"], response_data)
                self.assertIn("decision.fallback", [row["event_type"] for row in events])
                self.assertNotIn("do-not-record-me", json.dumps(events))
                recorder.close()

    def test_repeated_state_forced_twice_has_distinct_step_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with action_api() as (client, _):
                service, recorder = self.service(directory, client)
                service.run_once(force=True)
                service.run_once(force=True)
                inputs = [row for row in read_events(directory) if row["event_type"] == "decision.input"]
                self.assertEqual(len(inputs), 2)
                self.assertNotEqual(inputs[0]["step_id"], inputs[1]["step_id"])
                self.assertEqual(inputs[0]["payload"]["context"]["decision_id"], inputs[1]["payload"]["context"]["decision_id"])
                recorder.close()

    def test_disabled_recording_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with action_api() as (client, _):
                recorder = SessionRecorder(Path(directory), enabled=False)
                service = AdvisorService(client, NullRetriever(), RuleAdvisor(), recorder,
                                         backend_name="rule", output=lambda _: None)
                service.run_once()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_recording_failure_does_not_prevent_decision_or_action(self):
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "not-a-directory"
            blocked.write_text("occupied")
            with action_api() as (client, requests):
                service, _ = self.service(blocked, client, auto_act=True)
                self.assertTrue(service.run_once())
                self.assertEqual([request[0] for request in requests], ["GET", "GET", "POST"])

    def test_query_terms_scores_and_evidence_text_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Path(directory) / "knowledge.jsonl"
            knowledge.write_text(json.dumps({"evidence_id": "guide-1", "text": "monster 防御",
                                             "source_title": "攻略原文"}), encoding="utf-8")
            with action_api() as (client, _):
                service, recorder = self.service(directory, client)
                service.retriever = JsonKnowledgeRetriever(knowledge)
                service.run_once()
                events = read_events(directory)
                query = next(row["payload"] for row in events if row["event_type"] == "retrieval.query")
                ranked = next(row["payload"] for row in events if row["event_type"] == "retrieval.ranked")
                self.assertIn("monster", query["query_terms"])
                self.assertGreater(ranked["hits"][0]["score"], 0)
                self.assertEqual(ranked["hits"][0]["evidence"]["text"], "monster 防御")
                recorder.close()

    def test_malformed_environment_response_is_preserved_even_without_a_step(self):
        from sts2_advisor.api import Sts2RestClient
        with tempfile.TemporaryDirectory() as directory:
            with game_api(b"not json") as (url, _, _):
                service, recorder = self.service(directory, Sts2RestClient(url))
                with self.assertRaises(Sts2ApiError):
                    service.run_once()
                responses = [row for row in read_events(directory) if row["event_type"] == "environment.response"]
                self.assertEqual(responses[0]["payload"]["raw_body"], "not json")
                self.assertIsNone(responses[0]["step_id"])
                recorder.close("error")

    def test_plan_reuse_links_original_step_without_fake_model_or_retrieval_calls(self):
        changed = copy.deepcopy(COMBAT)
        playable = [card for card in changed["player"]["hand"] if card.get("can_play")]
        first_index = playable[0]["index"]
        changed["player"]["hand"] = [card for card in changed["player"]["hand"] if card["index"] != first_index]
        for index, card in enumerate(changed["player"]["hand"]):
            card["index"] = index
        changed["player"]["energy"] -= 1

        class PlanningBackend:
            def advise(self, context, evidence):
                cards = [item.candidate_id for item in context.candidates if item.action_type == "play_card"]
                return Advice(context.decision_id, cards[0], "两步计划", 0.8, plan_candidate_ids=cards[:2])

        with tempfile.TemporaryDirectory() as directory:
            with action_api([COMBAT, changed]) as (client, _):
                service, recorder = self.service(directory, client, PlanningBackend())
                service.run_once()
                service.run_once()
                events = read_events(directory)
                reused = next(row for row in events if row["event_type"] == "plan.reused")
                inputs = [row for row in events if row["event_type"] == "decision.input"]
                self.assertEqual(reused["payload"]["origin_step_id"], inputs[0]["step_id"])
                self.assertEqual(reused["step_id"], inputs[1]["step_id"])
                self.assertFalse(any(row["event_type"] in {"model.request", "retrieval.request"}
                                     and row["step_id"] == reused["step_id"] for row in events))
                recorder.close()

    def test_export_roundtrips_raw_events_and_keeps_sessions_separate(self):
        from sts2_advisor.record import export_session, iter_events, record_event, main
        with tempfile.TemporaryDirectory() as directory:
            with action_api() as (client, _):
                first, recorder = self.service(directory, client)
                first.run_once()
                recorder.close()
                second, another = self.service(directory, client)
                second.run_once()
                another.close()
                database = Path(directory) / "trajectories.sqlite3"
                output = io.StringIO()
                export_session(database, recorder.trajectory.session_id, output)
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(records[0]["record_type"], "session")
                self.assertEqual(records[0]["status"], "completed")
                self.assertEqual(len(records[0]["steps"]), 1)
                exported = records[1:]
                self.assertEqual([row["sequence"] for row in exported], list(range(1, len(exported) + 1)))
                self.assertEqual(list(iter_events(database, recorder.trajectory.session_id)),
                                 [{key: value for key, value in row.items() if key != "record_type"} for row in exported])
                self.assertEqual(len(list(Path(directory).glob("session-*.jsonl"))), 2)
                listing = io.StringIO()
                with redirect_stdout(listing):
                    self.assertEqual(main(["--db", str(database)]), 0)
                self.assertEqual(len(listing.getvalue().splitlines()), 2)
                destination = Path(directory) / "replay.jsonl"
                self.assertEqual(main(["--db", str(database), "--session", recorder.trajectory.session_id,
                                       "--output", str(destination)]), 0)
                self.assertEqual([json.loads(line) for line in destination.read_text().splitlines()], records)
                count = len(read_events(directory))
                record_event("should_not_leak")
                self.assertEqual(len(read_events(directory)), count)

    def test_cli_finishes_session_and_saves_run_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            with game_api(json.dumps(COMBAT).encode()) as (url, _, _):
                result = cli_tests.CliIntegrationTests().run_cli(url, "--backend", "rule", "--rag", "none",
                                                       "--recording-dir", directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(read_events(directory))
                with closing(sqlite3.connect(Path(directory) / "trajectories.sqlite3")) as connection:
                    status, metadata = connection.execute("SELECT status, metadata_json FROM sessions").fetchone()
                    self.assertEqual(status, "completed")
                    self.assertEqual(json.loads(metadata)["config"]["backend"], "rule")

    def test_each_remote_retry_keeps_its_own_request_and_original_response(self):
        answers = [
            SimpleNamespace(content="", reasoning_content="首轮原始推理"),
            SimpleNamespace(content='{"candidate_id":"card:2","reason":"第二次回答","confidence":0.8}'),
        ]
        responses = iter(SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=reason)], usage=None)
                         for message, reason in zip(answers, ["length", "stop"]))

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **request: next(responses)))

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
                backend = OpenAIAdvisor("teacher", api_key="secret")
            with action_api() as (client, _):
                service, recorder = self.service(directory, client, backend)
                service.run_once()
                events = read_events(directory)
                requests = [row["payload"] for row in events if row["event_type"] == "model.request"]
                results = [row["payload"] for row in events if row["event_type"] == "model.response"]
                self.assertEqual([row["request"]["max_tokens"] for row in requests], [1024, 2048])
                self.assertEqual([row["call_id"] for row in requests], [row["call_id"] for row in results])
                self.assertNotEqual(requests[0]["call_id"], requests[1]["call_id"])
                self.assertEqual(results[0]["reasoning"], "首轮原始推理")
                self.assertEqual(results[1]["content"], answers[1].content)
                recorder.close()

    def test_legacy_jsonl_can_be_joined_to_exact_database_step(self):
        with tempfile.TemporaryDirectory() as directory:
            with action_api() as (client, _):
                service, recorder = self.service(directory, client)
                service.run_once()
                legacy = json.loads(recorder.path.read_text())
                decision = next(row for row in read_events(directory) if row["event_type"] == "decision.input")
                self.assertEqual(legacy["step_id"], decision["step_id"])
                self.assertEqual(legacy["session_id"], decision["session_id"])
                recorder.close()


if __name__ == "__main__":
    unittest.main()
