from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from test_template_validation import REPO_ROOT, load_script


class OpenAIDecisionTests(unittest.TestCase):
    def setUp(self):
        self.gate = load_script("decision_protocol_under_test", REPO_ROOT / ".harness/gates/run_gate.py")
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_choice_uses_typed_questions_and_answers(self):
        answer = {
            "type": "choice", "name": "classification", "choice": "pending", "confidence": 0.88,
            "probabilities": [{"value": "complete", "probability": 0.1}, {"value": "pending", "probability": 0.9}],
        }
        with patch.object(self.gate, "_http_post_json", return_value={"answers": [answer]}) as post:
            result = self.gate.classify_choice("Work pending.", "Outcome?", ["complete", "pending"])
        self.assertEqual(result["top_choice"], "pending")
        self.assertEqual(result["confidence"], 0.88)
        self.assertEqual(result["probabilities"], {"complete": 0.1, "pending": 0.9})
        payload = post.call_args.args[2]
        self.assertEqual(payload["input"], "Work pending.")
        question = payload["questions"][0]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["instructions"], "Outcome?")
        self.assertEqual([item["value"] for item in question["choices"]], ["complete", "pending"])
        self.assertNotIn("options", payload)

    def test_predicates_are_batched_and_matched_by_name(self):
        conditions = [{"id": "debug", "condition": "A bug report."}, {"id": "design", "condition": "An architecture task."}]
        answers = [{"type": "predicate", "name": "design", "probability": 0.03}, {"type": "predicate", "name": "debug", "probability": 0.97}]
        with patch.object(self.gate, "_http_post_json", return_value={"answers": answers}) as post:
            result = self.gate.classify_predicates("Debug this crash.", conditions)
        self.assertEqual(result, {"debug": 0.97, "design": 0.03})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[2]["questions"], [
            {"type": "predicate", "name": item["id"], "instructions": item["condition"]}
            for item in conditions
        ])

    def test_refusal_has_no_routing_confidence(self):
        with patch.object(self.gate, "_http_post_json", return_value={"answers": [{"type": "refusal", "name": "classification"}]}):
            result = self.gate.classify_choice("Work.", "Outcome?", ["complete", "pending"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["status"], "undecided")

    def test_pre_compact_scores_and_folds_irrelevant_large_outputs(self):
        config = {
            "gates": {
                "context_fold_relevance": {
                    "mode": "active",
                    "fold_if_done_score_above": 0.8,
                }
            }
        }
        payload = {
            "active_task": "Fix runtime.ts state machine",
            "tool_outputs": [
                {"id": "out_relevant", "content": "Inspecting runtime.ts line 120"},
                {"id": "out_stale", "content": "x" * 800},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            trace = Path(directory) / "trace.jsonl"
            with redirect_stdout(output):
                code = self.gate.handle_pre_compact(payload, config, trace)
            self.assertEqual(code, 0)
            parsed = json.loads(output.getvalue())
            self.assertEqual(parsed["hookSpecificOutput"]["foldIds"], ["out_stale"])

    def test_unknown_and_refused_predicates_cannot_route(self):
        answers = [{"type": "refusal", "name": "debug"}, {"type": "predicate", "name": "unknown", "probability": 1.0}]
        with patch.object(self.gate, "_http_post_json", return_value={"answers": answers}):
            self.assertEqual(self.gate.classify_predicates("Work.", [{"id": "debug", "condition": "Bug?"}]), {"debug": 0.0})

    def test_provider_failures_are_logged_without_blocking_prompt(self):
        config = {"gates": {"prompt_skill_router": {"mode": "active", "conditions": [{"id": "debug", "condition": "Bug?"}]}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(self.gate, "_http_post_json", side_effect=OSError("offline")):
            output = StringIO()
            trace = Path(directory) / "trace.jsonl"
            with redirect_stdout(output):
                code = self.gate.handle_user_prompt_submit({"prompt": "Debug."}, config, trace)
            self.assertEqual(code, 0)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("offline", trace.read_text())

    def test_invalid_probabilities_are_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gate._probability(value)

    def test_unknown_choice_is_rejected(self):
        with patch.object(self.gate, "_http_post_json", return_value={"answers": [{"type": "choice", "choice": "unexpected"}]}):
            with self.assertRaises(ValueError):
                self.gate.classify_choice("Work.", "Outcome?", ["complete", "pending"])

    def test_credentials_are_redacted_before_classification(self):
        secret = "sk-" + "x" * 24
        with patch.object(self.gate, "_http_post_json", return_value={"answers": []}) as post:
            self.gate.classify_predicates("Debug " + secret, [{"id": "debug", "condition": "Bug?"}])
        self.assertNotIn(secret, post.call_args.args[2]["input"])
        self.assertIn("<REDACTED>", post.call_args.args[2]["input"])

    def test_malformed_response_is_rejected(self):
        with patch.object(self.gate, "_http_post_json", return_value={"answers": {}}):
            with self.assertRaises(ValueError):
                self.gate.classify_predicates("Work.", [{"id": "debug", "condition": "Bug?"}])


class OfflineRoutingTests(unittest.TestCase):
    def setUp(self):
        self.gate = load_script("offline_routing_under_test", REPO_ROOT / ".harness/gates/run_gate.py")
        self.config = self.gate.load_configuration()
        self.conditions = self.config["gates"]["prompt_skill_router"]["conditions"]

    def matched(self, prompt):
        with patch.dict(os.environ, {}, clear=True):
            scores = self.gate.classify_predicates(prompt, self.conditions)
        threshold = self.config["gates"]["prompt_skill_router"]["threshold"]
        return {name for name, score in scores.items() if score >= threshold}

    def test_representative_specialist_prompts(self):
        cases = {
            "There is a flaky race condition crash in the worker channel teardown, debug this": "grounded-prm-debugger",
            "Debug a crash.": "grounded-prm-debugger",
            "Design the system architecture.": "system-architect-atam",
            "Implement a state machine.": "cmcts-solver",
            "Review this pull request.": "review-tribunal",
            "Write an AST codemod.": "ast-grep",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertIn(expected, self.matched(prompt))

    def test_unrelated_words_do_not_route(self):
        for prompt in ("", "The user requests lunch.", "A preview of the new sign-designator."):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.matched(prompt), set())

    def test_multiple_specialists_can_match(self):
        self.assertTrue({"grounded-prm-debugger", "review-tribunal"}.issubset(self.matched("Review this bug fix.")))


class ClientConfigurationTests(unittest.TestCase):
    def test_codex_launches_the_same_servers_in_the_client_workspace(self):
        config_path = REPO_ROOT / ".codex/config.toml"
        config = tomllib.loads(config_path.read_text())
        shared = json.loads((REPO_ROOT / ".mcp.json").read_text())["mcpServers"]
        self.assertTrue(set(config["mcp_servers"]).issubset(set(shared)))
        for name, server in config["mcp_servers"].items():
            with self.subTest(server=name):
                self.assertEqual(server["command"], shared[name]["command"])
                self.assertEqual(server["args"], shared[name]["args"])
                self.assertNotIn("cwd", server)


if __name__ == "__main__":
    unittest.main()
