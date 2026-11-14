from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / ".harness/gates"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AntigravityHookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="three-antigravity-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trace = self.root / "decision-gates.jsonl"
        self.core = load_script("google_shared_gate", GATES / "run_gate.py")
        with patch.dict(sys.modules, {"run_gate": self.core}):
            self.adapter = load_script("google_adapter", GATES / "antigravity_gate.py")
        self.cfg = json.loads((GATES / "decision-gates.json").read_text(encoding="utf-8"))
        self.original_env = os.environ.copy()
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        network = patch.object(self.core.urllib.request, "urlopen", side_effect=AssertionError("Unexpected provider call"))
        network.start()
        self.addCleanup(network.stop)

    def dispatch(self, hook, payload):
        return self.adapter.dispatch(hook, payload, self.cfg, self.trace)

    def user(self, index=0):
        return {"type": "USER_INPUT", "source": "USER_EXPLICIT", "status": "DONE", "step_index": index, "content": "Fix validation"}

    def response(self, text="I will run the checks.", index=1):
        return {"type": "GENERIC", "source": "MODEL", "status": "DONE", "step_index": index, "content": text}

    def transcript(self, records):
        path = self.root / "transcript.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    def stop_event(self, path):
        return {"conversationId": "test-conversation", "lastUserInput": "Fix validation", "transcriptPath": str(path), "fullyIdle": True, "terminationReason": "model_stop", "executionNum": 1}

    def test_prompt_routing_uses_native_ephemeral_steps(self):
        with patch.object(self.core, "classify_predicates", return_value={"grounded-prm-debugger": 0.99}):
            result = self.dispatch("PreInvocation", {"lastUserInput": "Debug a crash", "invocationNum": 3})
        self.assertEqual(set(result), {"injectSteps"})
        self.assertIn("grounded-prm-debugger.md", result["injectSteps"][0]["ephemeralMessage"])
        self.assertNotIn("modifiedPrompt", result)

    def test_offline_routing_works_for_short_requests(self):
        result = self.dispatch("PreInvocation", {"lastUserInput": "debug this crash"})
        self.assertIn("grounded-prm-debugger.md", result["injectSteps"][0]["ephemeralMessage"])

    def test_pre_invocation_deduplicates_within_same_user_turn_and_resets_on_new_turn(self):
        path = self.transcript([self.user(index=0)])
        event = {
            "conversationId": "conv-dedup-1",
            "lastUserInput": "debug this crash",
            "transcriptPath": str(path),
            "invocationNum": 1,
        }
        first = self.dispatch("PreInvocation", event)
        self.assertIn("injectSteps", first)
        # Second invocation in the same user turn (e.g. after a tool call) must not re-inject
        second = self.dispatch("PreInvocation", {**event, "invocationNum": 2})
        self.assertEqual(second, {})
        # New user step with the same text must route again
        self.transcript([self.user(index=0), self.response(index=1), self.user(index=2)])
        third = self.dispatch("PreInvocation", {**event, "invocationNum": 3})
        self.assertIn("injectSteps", third)

    def test_shadow_or_unmatched_routing_injects_nothing(self):
        self.assertEqual(self.dispatch("PreInvocation", {"lastUserInput": "hello"}), {})
        self.cfg["gates"]["prompt_skill_router"]["mode"] = "shadow"
        self.assertEqual(self.dispatch("PreInvocation", {"lastUserInput": "debug this crash"}), {})

    def test_provider_failure_does_not_rewrite_or_block_prompt(self):
        with patch.object(self.core, "classify_predicates", side_effect=RuntimeError("Fixture failure")):
            self.assertEqual(self.dispatch("PreInvocation", {"lastUserInput": "debug this crash"}), {})

    def test_tool_decision_normalizes_google_arguments(self):
        def deny(payload, config, trace):
            self.assertEqual(payload, {"tool_name": "view_file", "tool_input": {"AbsolutePath": "example.txt"}})
            print(json.dumps({"decision": "block", "reason": "Fixture policy denial"}))
            return 0

        with patch.object(self.core, "handle_pre_tool_use", side_effect=deny):
            result = self.dispatch("PreToolUse", {"toolCall": {"name": "view_file", "args": {"AbsolutePath": "example.txt"}}})
        self.assertEqual(result, {"decision": "deny", "reason": "Fixture policy denial"})

    def test_allowed_tool_keeps_normal_permission_policy(self):
        result = self.dispatch("PreToolUse", {"toolCall": {"name": "view_file", "args": {"AbsolutePath": "README.md"}}})
        self.assertEqual(result, {"decision": "allow"})
        self.assertNotIn("permissionOverrides", result)

    def test_malformed_tool_call_is_rejected(self):
        for call in (None, [], {}, {"name": "view_file"}, {"name": 1, "args": {}}):
            with self.subTest(call=call), self.assertRaises(ValueError):
                self.dispatch("PreToolUse", {"toolCall": call})

    def test_native_completed_response_requests_continuation(self):
        path = self.transcript([self.user(), self.response()])
        result = self.dispatch("Stop", self.stop_event(path))
        self.assertEqual(result["decision"], "continue")
        self.assertTrue(result["reason"])

    def test_turn_continues_only_once_even_if_reply_changes(self):
        path = self.transcript([self.user(), self.response()])
        with patch.object(self.core, "handle_stop_gate", wraps=self.core.handle_stop_gate) as policy:
            self.assertEqual(self.dispatch("Stop", self.stop_event(path))["decision"], "continue")
            self.transcript([self.user(), self.response("I will inspect the code.", index=2)])
            self.assertEqual(self.dispatch("Stop", self.stop_event(path)), {"decision": "stop"})
        policy.assert_called_once()

    def test_same_prompt_on_a_new_user_step_can_continue(self):
        path = self.transcript([self.user(), self.response()])
        self.assertEqual(self.dispatch("Stop", self.stop_event(path))["decision"], "continue")
        self.transcript([self.user(index=3), self.response(index=4)])
        self.assertEqual(self.dispatch("Stop", self.stop_event(path))["decision"], "continue")

    def test_losing_user_step_in_a_large_tail_does_not_reset_guard(self):
        path = self.transcript([self.user(), self.response()])
        self.assertEqual(self.dispatch("Stop", self.stop_event(path))["decision"], "continue")
        path.write_bytes(b"x" * (self.adapter.MAX_TRANSCRIPT_BYTES + 100) + b"\n" + json.dumps(self.response(index=2)).encode() + b"\n")
        self.assertEqual(self.dispatch("Stop", self.stop_event(path)), {"decision": "stop"})

    def test_completed_report_question_and_shadow_allow_stopping(self):
        for text in ("All checks passed.", "Which storage backend should we use?"):
            with self.subTest(text=text):
                path = self.transcript([self.user(), self.response(text)])
                self.assertEqual(self.dispatch("Stop", self.stop_event(path)), {"decision": "stop"})
        self.cfg["gates"]["premature_stop_guard"]["mode"] = "shadow"
        path = self.transcript([self.user(), self.response()])
        self.assertEqual(self.dispatch("Stop", self.stop_event(path)), {"decision": "stop"})
        self.assertFalse((self.root / "antigravity").exists())

    def test_background_and_error_stops_do_not_read_transcript(self):
        cases = ({"fullyIdle": False}, {"terminationReason": "error"}, {"terminationReason": "max_steps_exceeded"})
        with patch.object(self.adapter, "read_turn", side_effect=AssertionError("Unexpected transcript read")):
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    self.assertEqual(self.dispatch("Stop", {**self.stop_event("unused"), **overrides}), {"decision": "stop"})

    def test_tool_output_and_partial_or_previous_responses_are_excluded(self):
        cases = [
            [self.response(), self.user(index=3)],
            [self.user(), self.response(), {"type": "VIEW_FILE", "source": "MODEL", "status": "DONE", "content": "I will run the checks."}],
            [self.user(), {**self.response(), "status": "RUNNING"}],
            [self.user(), {**self.response(), "tool_calls": [{"name": "view_file"}]}],
            [{"type": "assistant.message", "data": {"content": "I will run the checks."}}],
        ]
        for records in cases:
            with self.subTest(records=records):
                path = self.transcript(records)
                self.assertEqual(self.dispatch("Stop", self.stop_event(path)), {"decision": "stop"})

    def test_missing_transcript_logs_only_a_status(self):
        result = self.dispatch("Stop", self.stop_event(self.root / "missing.jsonl"))
        self.assertEqual(result, {"decision": "stop"})
        record = json.loads(self.trace.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "transcript_unavailable")
        self.assertNotIn("transcriptPath", record)

    def test_unwritable_state_prevents_a_continuation_request(self):
        path = self.transcript([self.user(), self.response()])
        with patch.object(self.adapter, "_save_continuation", return_value=False):
            self.assertEqual(self.dispatch("Stop", self.stop_event(path)), {"decision": "stop"})

    def test_state_contains_hash_and_step_without_conversation_text(self):
        path = self.transcript([self.user(), self.response()])
        self.dispatch("Stop", self.stop_event(path))
        files = list((self.root / "antigravity").glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertNotIn("test-conversation", files[0].name)
        state = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(set(state), {"prompt_token", "user_step"})
        self.assertEqual(state["user_step"], 0)
        self.assertEqual(len(state["prompt_token"]), 64)
        self.assertNotIn("Fix validation", files[0].read_text(encoding="utf-8"))

    def test_mcp_definitions_match_shared_source_with_distinct_names(self):
        shared = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
        google = json.loads((ROOT / ".agents/mcp_config.json").read_text(encoding="utf-8"))["mcpServers"]
        self.assertTrue(all(name.startswith("harness-") for name in google))
        google_unprefixed = {name.removeprefix("harness-"): config for name, config in google.items()}
        self.assertTrue(set(google_unprefixed).issubset(set(shared)))
        self.assertEqual(google_unprefixed, {name: shared[name] for name in google_unprefixed})
        self.assertTrue(all("cwd" not in config for config in google.values()))

    def test_hook_layout_uses_google_named_groups_and_timeouts(self):
        hooks = json.loads((ROOT / ".agents/hooks.json").read_text(encoding="utf-8"))["harness-decision-gates"]
        self.assertTrue(hooks["enabled"])
        self.assertEqual(set(hooks) - {"enabled"}, {"PreInvocation", "PreToolUse", "Stop"})
        self.assertEqual(hooks["PreToolUse"][0]["matcher"], "*")
        for name in ("PreInvocation", "PreToolUse", "Stop"):
            item = hooks[name][0]
            handler = item["hooks"][0] if "hooks" in item else item
            command = shlex.split(handler["command"])
            self.assertEqual(command[:4], ["uv", "run", "--no-project", "python"])
            self.assertEqual(command[4], ".harness/gates/antigravity_gate.py")
            # Live Windows clients parse command strings with their own splitter;
            # any quote or shell-special character is a portability regression.
            self.assertTrue(all(ch not in handler["command"] for ch in "'\";&|<>$`"), handler["command"])
            self.assertEqual(command[-2:], ["--hook", name])
            self.assertEqual(handler["timeout"], 15)

    def test_configured_launchers_run_from_workspace_root_without_quoting(self):
        project = self.root / "workspace with spaces"
        scripts = project / ".harness/gates"
        scripts.mkdir(parents=True)
        for name in ("run_gate.py", "routing.py", "antigravity_gate.py", "decision-gates.json"):
            shutil.copyfile(GATES / name, scripts / name)
        config_dir = project / ".agents"
        config_dir.mkdir()
        shutil.copyfile(ROOT / ".agents/hooks.json", config_dir / "hooks.json")
        hooks = json.loads((config_dir / "hooks.json").read_text(encoding="utf-8"))["harness-decision-gates"]
        excluded = {"OPENAI_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY", "UV_ENV_FILE"}
        env = {name: value for name, value in self.original_env.items() if name not in excluded}
        fixtures = {
            "PreInvocation": ({"lastUserInput": "Debug a crash"}, "injectSteps"),
            "PreToolUse": ({"toolCall": {"name": "view_file", "args": {"AbsolutePath": "README.md"}}}, "allow"),
            "Stop": ({"fullyIdle": False, "terminationReason": "model_stop"}, "stop"),
        }
        for name, (payload, expected) in fixtures.items():
            item = hooks[name][0]
            handler = item["hooks"][0] if "hooks" in item else item
            args = shlex.split(handler["command"])
            self.assertEqual(args[:4], ["uv", "run", "--no-project", "python"])
            with self.subTest(hook=name):
                result = subprocess.run(
                    [sys.executable, *args[4:]], input=json.dumps(payload),
                    capture_output=True, text=True, cwd=project, env=env, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                response = json.loads(result.stdout)
                if name == "PreInvocation":
                    self.assertTrue(response[expected])
                else:
                    self.assertEqual(response, {"decision": expected})
                if os.name == "nt":
                    cmd_result = subprocess.run(
                        ["cmd", "/c", handler["command"]], input=json.dumps(payload),
                        capture_output=True, text=True, cwd=project, env=env, timeout=15,
                    )
                    self.assertEqual(cmd_result.returncode, 0, cmd_result.stderr)
                    self.assertEqual(json.loads(cmd_result.stdout), response)

    def test_cli_returns_native_json_and_rejects_malformed_input(self):
        project = self.root / "workspace"
        scripts = project / ".harness/gates"
        scripts.mkdir(parents=True)
        for name in ("run_gate.py", "routing.py", "antigravity_gate.py", "decision-gates.json"):
            shutil.copyfile(GATES / name, scripts / name)
        excluded = {"OPENAI_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY", "UV_ENV_FILE"}
        env = {name: value for name, value in self.original_env.items() if name not in excluded}
        command = [sys.executable, str(scripts / "antigravity_gate.py"), "--hook", "PreToolUse"]
        event = {"toolCall": {"name": "view_file", "args": {"AbsolutePath": "README.md"}}}
        result = subprocess.run(command, input=json.dumps(event), capture_output=True, text=True, cwd=project, env=env, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"decision": "allow"})
        malformed = subprocess.run(command, input="not-json", capture_output=True, text=True, cwd=project, env=env, timeout=15)
        self.assertEqual(malformed.returncode, 1)
        self.assertEqual(malformed.stdout, "")
        self.assertNotIn("not-json", malformed.stderr)


if __name__ == "__main__":
    unittest.main()
