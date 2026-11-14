from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
GATES_DIR = REPO_ROOT / ".harness/gates"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CopilotHookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="three-copilot-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trace = self.root / "gate-events.jsonl"
        self.core = load_script("shared_gate_under_test", GATES_DIR / "run_gate.py")
        with patch.dict(sys.modules, {"run_gate": self.core}):
            self.adapter = load_script("copilot_gate_under_test", GATES_DIR / "copilot_gate.py")
        self.cfg = json.loads((GATES_DIR / "decision-gates.json").read_text(encoding="utf-8"))
        self.original_env = os.environ.copy()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network = patch.object(self.core.urllib.request, "urlopen", side_effect=AssertionError("Unexpected network request"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def dispatch(self, hook, payload):
        return self.adapter.dispatch(hook, payload, self.cfg, self.trace)

    def transcript(self, events):
        path = self.root / "events.jsonl"
        path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
        return path

    def stop_payload(self, path):
        return {"transcriptPath": str(path), "stopReason": "end_turn", "stop_hook_active": False}

    def test_prompt_routing_preserves_runtime_context(self):
        original = "Please debug this.\n<current_datetime>2026-11-06</current_datetime>"
        with patch.object(self.core, "classify_predicates", return_value={"grounded-prm-debugger": 0.99}):
            result = self.dispatch("userPromptTransformed", {"prompt": "Debug this crash", "transformedPrompt": original})
        self.assertEqual(set(result), {"modifiedTransformedPrompt"})
        self.assertTrue(result["modifiedTransformedPrompt"].startswith(original + "\n\n"))
        self.assertIn(".harness/specialists/grounded-prm-debugger.md", result["modifiedTransformedPrompt"])

    def test_prompt_context_is_injected_once(self):
        payload = {"prompt": "Debug this crash", "transformedPrompt": "Debug this crash"}
        with patch.object(self.core, "classify_predicates", return_value={"grounded-prm-debugger": 0.99}) as classify:
            first = self.dispatch("userPromptTransformed", payload)
            second = self.dispatch("userPromptTransformed", {**payload, "transformedPrompt": first["modifiedTransformedPrompt"]})
        self.assertEqual(second, {})
        classify.assert_called_once()

    def test_shadow_router_leaves_prompt_unchanged(self):
        self.cfg["gates"]["prompt_skill_router"]["mode"] = "shadow"
        with patch.object(self.core, "classify_predicates", return_value={"grounded-prm-debugger": 0.99}):
            result = self.dispatch("userPromptTransformed", {"prompt": "Debug crash", "transformedPrompt": "Original prompt"})
        self.assertEqual(result, {})

    def test_unmatched_prompt_is_unchanged(self):
        result = self.dispatch("userPromptTransformed", {"prompt": "Hello", "transformedPrompt": "Hello with runtime context"})
        self.assertEqual(result, {})

    def test_secret_paths_return_native_denial(self):
        for path in (".env", ".env.production", r"C:\work\.env.local", "/home/user/id_rsa", "client.pem", "vault.kdbx", "credentials.json"):
            with self.subTest(path=path):
                result = self.dispatch("preToolUse", {"toolName": "view", "toolArgs": {"path": path}})
                self.assertEqual(result["permissionDecision"], "deny")
                self.assertTrue(result["permissionDecisionReason"])
                self.assertNotIn("decision", result)

    def test_string_encoded_tool_arguments_are_checked(self):
        result = self.dispatch("preToolUse", {"toolName": "view", "toolArgs": json.dumps({"path": ".env.local"})})
        self.assertEqual(result["permissionDecision"], "deny")

    def test_ordinary_paths_do_not_return_a_permission_override(self):
        for path in ("src/app.py", "README.md", ".environment", "client.pem.notes"):
            with self.subTest(path=path):
                self.assertEqual(self.dispatch("preToolUse", {"toolName": "view", "toolArgs": {"path": path}}), {})

    def test_announced_work_in_transcript_blocks_stop(self):
        path = self.transcript([
            {"type": "user.message", "data": {"content": "Fix the bug"}},
            {"type": "assistant.message", "data": {"content": "I will run the tests."}},
            {"type": "session.idle", "data": {}},
        ])
        self.assertEqual(self.dispatch("agentStop", self.stop_payload(path))["decision"], "block")

    def test_completed_report_allows_stop(self):
        path = self.transcript([{"type": "assistant.message", "data": {"content": "Fixed the bug. All tests pass."}}])
        self.assertEqual(self.dispatch("agentStop", self.stop_payload(path)), {})

    def test_question_allows_stop(self):
        path = self.transcript([{"type": "assistant.message", "data": {"content": "Which database should we use?"}}])
        self.assertEqual(self.dispatch("agentStop", self.stop_payload(path)), {})

    def test_already_continued_turn_does_not_read_transcript(self):
        with patch.object(self.adapter, "last_assistant_message", side_effect=AssertionError("Unexpected transcript read")):
            result = self.dispatch("agentStop", {"stop_hook_active": True, "transcriptPath": "unused.jsonl"})
        self.assertEqual(result, {})

    def test_shadow_stop_gate_records_without_blocking(self):
        self.cfg["gates"]["premature_stop_guard"]["mode"] = "shadow"
        path = self.transcript([{"type": "assistant.message", "data": {"content": "I will run the tests."}}])
        self.assertEqual(self.dispatch("agentStop", self.stop_payload(path)), {})
        events = [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(events[-1]["would_block"])

    def test_missing_transcript_allows_stop_with_metadata_only(self):
        self.assertEqual(self.dispatch("agentStop", self.stop_payload(self.root / "missing.jsonl")), {})
        record = json.loads(self.trace.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "no_assistant_message")
        self.assertNotIn("transcriptPath", record)

    def test_previous_turn_and_subagent_text_are_not_used(self):
        path = self.transcript([
            {"type": "assistant.message", "data": {"content": "I will run the tests."}},
            {"type": "user.message", "data": {"content": "A new task"}},
            {"type": "assistant.message", "agentId": "child", "data": {"content": "I will check the code."}},
        ])
        self.assertEqual(self.adapter.last_assistant_message(str(path)), "")

    def test_tool_request_announcement_is_not_a_final_response(self):
        path = self.transcript([{
            "type": "assistant.message",
            "data": {"content": "I will run the tests.", "toolRequests": [{"name": "bash"}]},
        }])
        self.assertEqual(self.adapter.last_assistant_message(str(path)), "")

    def test_transcript_tail_is_bounded_and_tolerates_partial_records(self):
        path = self.root / "large.jsonl"
        event = {"type": "assistant.message", "data": {"content": "I will run the tests."}}
        path.write_bytes(b"x" * (self.adapter.MAX_TRANSCRIPT_BYTES + 100) + b"\n" + json.dumps(event).encode() + b"\nnot-json\n")
        self.assertEqual(self.adapter.last_assistant_message(str(path)), "I will run the tests.")

    def test_compatibility_callbacks_skip_before_any_gate_work(self):
        for hook in ("UserPromptSubmit", "PreToolUse", "Stop"):
            with self.subTest(hook=hook):
                payload = {"hook_event_name": hook, "timestamp": "2026-11-06T00:00:00Z"}
                with patch.object(sys, "argv", ["run_gate.py", "--hook", hook, "--skip-copilot-compat"]), patch.object(sys, "stdin", StringIO(json.dumps(payload))), patch.object(self.core, "load_configuration", side_effect=AssertionError("Duplicate gate execution")), redirect_stdout(StringIO()) as output:
                    self.assertEqual(self.core.main(), 0)
                self.assertEqual(output.getvalue(), "")

    def test_ordinary_claude_callbacks_still_dispatch(self):
        payload = {"hook_event_name": "UserPromptSubmit", "prompt": "Debug crash"}
        with patch.object(sys, "argv", ["run_gate.py", "--hook", "UserPromptSubmit", "--skip-copilot-compat"]), patch.object(sys, "stdin", StringIO(json.dumps(payload))), patch.object(self.core, "load_configuration", return_value=self.cfg), patch.object(self.core, "_dispatch_gate", return_value=0) as dispatch:
            self.assertEqual(self.core.main(), 0)
        dispatch.assert_called_once()

    def test_native_configuration_and_duplicate_suppression_are_wired(self):
        config = json.loads((REPO_ROOT / ".github/hooks/decision-gates.json").read_text(encoding="utf-8"))
        self.assertEqual(config["version"], 1)
        self.assertEqual(set(config["hooks"]), {"userPromptTransformed", "preToolUse", "agentStop"})
        for event, entries in config["hooks"].items():
            self.assertEqual(len(entries), 1)
            self.assertIn(f"copilot_gate.py --hook {event}", entries[0]["command"])
            self.assertEqual(entries[0]["cwd"], ".")
        claude = json.loads((REPO_ROOT / ".claude/settings.json").read_text(encoding="utf-8"))
        for entries in claude["hooks"].values():
            for group in entries:
                for handler in group["hooks"]:
                    self.assertIn("--skip-copilot-compat", handler["command"])

    def test_cli_emits_one_native_response_and_handles_malformed_json(self):
        project = self.root / "workspace"
        destination = project / ".harness/gates"
        destination.mkdir(parents=True)
        for name in ("run_gate.py", "routing.py", "copilot_gate.py", "decision-gates.json"):
            shutil.copyfile(GATES_DIR / name, destination / name)
        env = {key: value for key, value in self.original_env.items() if key not in {"OPENAI_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY", "UV_ENV_FILE"}}
        command = [sys.executable, str(destination / "copilot_gate.py"), "--hook", "preToolUse"]
        result = subprocess.run(command, input=json.dumps({"toolName": "view", "toolArgs": {"path": ".env"}}), capture_output=True, text=True, cwd=project, env=env, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["permissionDecision"], "deny")
        malformed = subprocess.run(command, input="not-json", capture_output=True, text=True, cwd=project, env=env, timeout=15)
        self.assertEqual(malformed.returncode, 1)
        self.assertEqual(malformed.stdout, "")
        self.assertNotIn("not-json", malformed.stderr)

if __name__ == "__main__":
    unittest.main()
