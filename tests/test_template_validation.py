from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = Path(".autoresearch/prepare_and_eval.py")
TOOLING_FILES = (
    Path(".harness/tools/ast-grep.md"),
    Path(".harness/gates/decision-gates.json"),
    Path(".harness/gates/run_gate.py"),
    Path(".harness/gates/routing.py"),
    Path(".context/subsystems/tooling.md"),
)


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaticValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="three-validation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        source_evaluator = load_script("evaluator_source", REPO_ROOT / EVALUATOR_PATH)
        required = [path.relative_to(REPO_ROOT) for path in source_evaluator.get_required_files()]
        paths = set(required) | set(TOOLING_FILES) | {
            EVALUATOR_PATH, Path("AGENTS.md"), Path("CLAUDE.md")
        }
        for relative in paths:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / relative, destination)
        self.module = load_script("evaluator_under_test", self.root / EVALUATOR_PATH)
        self.gates_path = self.root / ".harness/gates/decision-gates.json"
        self.gates_config = json.loads(self.gates_path.read_text(encoding="utf-8"))

    def snapshot(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*") if path.is_file()
        }

    def run_evaluator(self, *args: str):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run(
            [sys.executable, str(self.root / EVALUATOR_PATH), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=15,
            check=False,
        )

    def test_valid_template_passes(self):
        self.assertEqual(self.module.validate_static_architecture(), [])

    def test_cold_memory_secret_scan_fails_closed(self):
        target = self.root / ".context" / "plan.md"
        original = target.read_bytes()
        try:
            target.write_text(original.decode("utf-8") + "\ntoken: gsk_" + "a" * 24 + "\n", encoding="utf-8")
            errors = self.module.validate_static_architecture()
            self.assertTrue(any("SECRET_IN_COLD_MEMORY" in error and "plan.md" in error for error in errors), errors)
        finally:
            target.write_bytes(original)

    def test_prose_mentions_of_bearer_auth_are_not_secrets(self):
        target = self.root / ".context" / "plan.md"
        original = target.read_bytes()
        try:
            target.write_text(original.decode("utf-8") + "\nThe gateway enforces bearer auth on every request.\n", encoding="utf-8")
            self.assertEqual([error for error in self.module.validate_static_architecture() if "SECRET" in error], [])
        finally:
            target.write_bytes(original)

    def test_evaluator_helpers_are_bound_to_fixture(self):
        self.assertEqual(self.module.ROOT, self.root)
        self.assertEqual(self.module.TRACES_DIR, self.root / ".autoresearch/traces")
        for helper, relative in ((self.module.routing, ".harness/gates/routing.py"), (self.module.task_process, ".autoresearch/task_process.py")):
            with self.subTest(helper=relative):
                self.assertEqual(Path(helper.__file__).resolve(), (self.root / relative).resolve())

    def test_missing_helpers_are_reported_in_process_and_on_fresh_import(self):
        for attribute, relative in (("routing", ".harness/gates/routing.py"), ("task_process", ".autoresearch/task_process.py"), ("run_gate", ".harness/gates/run_gate.py")):
            with self.subTest(helper=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.unlink()
                try:
                    errors = self.module.validate_static_architecture()
                    self.assertTrue(any("MISSING_REQUIRED_FILE" in error and path.name in error for error in errors), errors)
                    fresh = load_script("evaluator_missing_helper", self.root / EVALUATOR_PATH)
                    self.assertIsNone(getattr(fresh, attribute))
                    self.assertTrue(any("MISSING_REQUIRED_FILE" in error and path.name in error for error in fresh.validate_static_architecture()))
                finally:
                    path.write_bytes(original)

    def install_domain(self):
        condition = {
            "id": "example-domain", "condition": "Handle example-domain requests.",
            "inject_path": ".harness/specialists/example-domain.md",
            "cold_memory": ".context/references/example-domain/main.md",
            "references": [".context/references/example-domain/details.md"],
            "keywords": ["example-domain"], "priority": 200,
        }
        self.gates_config["gates"]["prompt_skill_router"]["conditions"].append(condition)
        self.gates_path.write_text(json.dumps(self.gates_config), encoding="utf-8")
        paths = [condition["inject_path"], condition["cold_memory"], *condition["references"]]
        for relative in paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Example domain contract\n" + "Applicable domain requirements and verification.\n" * 3, encoding="utf-8")
        return condition, paths

    def test_unregistering_a_domain_removes_its_required_files(self):
        condition, paths = self.install_domain()
        self.assertEqual(self.module.validate_static_architecture(), [])
        self.gates_config["gates"]["prompt_skill_router"]["conditions"].remove(condition)
        self.gates_path.write_text(json.dumps(self.gates_config), encoding="utf-8")
        for relative in paths:
            (self.root / relative).unlink()
        self.assertEqual(self.module.validate_static_architecture(), [])
        result = self.run_evaluator("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registration_validates_and_hashes_new_domain_files(self):
        condition, paths = self.install_domain()
        self.assertTrue({self.root / path for path in paths}.issubset(self.module.get_required_files()))
        before = self.module.compute_harness_hash()
        reference = self.root / condition["references"][0]
        reference.write_text(reference.read_text(encoding="utf-8") + "Changed domain guidance.\n", encoding="utf-8")
        self.assertNotEqual(self.module.compute_harness_hash(), before)
        reference.unlink()
        self.assertTrue(any("BROKEN_GATE_CONTEXT_PATH" in error for error in self.module.validate_static_architecture()))

    def test_invalid_registered_context_paths_are_rejected(self):
        condition, paths = self.install_domain()
        for value in (None, "", 42, "../outside.md", ".context", str(self.root / paths[1])):
            with self.subTest(value=value):
                condition["cold_memory"] = value
                self.gates_path.write_text(json.dumps(self.gates_config), encoding="utf-8")
                errors = self.module.validate_static_architecture()
                self.assertTrue(any("BROKEN_GATE_CONTEXT_PATH" in error for error in errors), errors)

    def test_missing_routing_module_is_reported_by_check_only(self):
        (self.root / ".harness/gates/routing.py").unlink()
        result = self.run_evaluator("--check-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn("MISSING_REQUIRED_FILE", result.stderr)
        self.assertIn("routing.py", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_empty_design_lore_is_valid(self):
        (self.root / ".context/design-lore-taxonomy.json").write_text("[]\n", encoding="utf-8")
        before = self.snapshot()
        self.assertEqual(self.module.validate_static_architecture(), [])
        result = self.run_evaluator("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_invalid_design_lore_is_rejected(self):
        path = self.root / ".context/design-lore-taxonomy.json"
        for value in ("", "{broken", "{}", "null"):
            with self.subTest(value=value):
                path.write_text(value, encoding="utf-8")
                errors = self.module.validate_static_architecture()
                self.assertTrue(any(error.startswith("INVALID_DESIGN_LORE") for error in errors), errors)

    def test_missing_design_lore_is_rejected(self):
        (self.root / ".context/design-lore-taxonomy.json").unlink()
        errors = self.module.validate_static_architecture()
        self.assertTrue(any("MISSING_REQUIRED_FILE" in error and "design-lore" in error for error in errors), errors)

    def test_short_required_markdown_is_rejected(self):
        (self.root / ".context/subsystems/tooling.md").write_text("# TODO\n", encoding="utf-8")
        errors = self.module.validate_static_architecture()
        self.assertTrue(any("EMPTY_OR_STUB_FILE" in error and "tooling.md" in error for error in errors), errors)

    def test_tooling_files_are_required(self):
        for relative in TOOLING_FILES:
            with self.subTest(path=str(relative)):
                path = self.root / relative
                original = path.read_bytes()
                path.unlink()
                try:
                    errors = self.module.validate_static_architecture()
                    self.assertTrue(any("MISSING_REQUIRED_FILE" in error for error in errors), errors)
                finally:
                    path.write_bytes(original)

    def test_missing_mirrors_are_rejected(self):
        for name in ("AGENTS.md", "CLAUDE.md"):
            with self.subTest(mirror=name):
                path = self.root / name
                original = path.read_bytes()
                path.unlink()
                try:
                    self.assertIn(
                        f"MISSING_CONSTITUTION_MIRROR: {name}",
                        self.module.validate_static_architecture(),
                    )
                finally:
                    path.write_bytes(original)

    def test_mirror_drift_is_rejected(self):
        for name in ("AGENTS.md", "CLAUDE.md"):
            with self.subTest(mirror=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"\nChanged independently.\n")
                try:
                    errors = self.module.validate_static_architecture()
                    self.assertIn(f"CONSTITUTION_DRIFT: {name} differs from CONSTITUTION.md", errors)
                finally:
                    path.write_bytes(original)

    def test_invalid_injection_paths_are_rejected(self):
        outside = Path(self.temp.name) / "outside.md"
        outside.write_text("Outside the repository.\n", encoding="utf-8")
        cases = (
            None, "", "   ", 42, ".harness", ".harness/missing.md",
            "../outside.md", str(self.root / ".harness/tools/ast-grep.md"),
        )
        for value in cases:
            with self.subTest(inject_path=value):
                config = json.loads(json.dumps(self.gates_config))
                config["gates"]["prompt_skill_router"]["conditions"][0]["inject_path"] = value
                self.gates_path.write_text(json.dumps(config), encoding="utf-8")
                errors = self.module.validate_static_architecture()
                self.assertTrue(any("BROKEN_GATE_INJECT_PATH" in error for error in errors), errors)

    def test_invalid_gate_json_is_rejected(self):
        self.gates_path.write_text("{not-json}", encoding="utf-8")
        errors = self.module.validate_static_architecture()
        self.assertTrue(any("INVALID_DECISION_GATES_JSON" in error for error in errors), errors)

    def test_invalid_gate_configuration_is_rejected(self):
        for value in ([], {"gates": {}}, {"gates": {"prompt_skill_router": {"conditions": {}}}}):
            with self.subTest(config=value):
                self.gates_path.write_text(json.dumps(value), encoding="utf-8")
                errors = self.module.validate_static_architecture()
                self.assertTrue(any("INVALID_DECISION_GATES" in error for error in errors), errors)

    def test_check_only_creates_no_files(self):
        before = self.snapshot()
        result = self.run_evaluator("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[PASS]", result.stdout)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / ".autoresearch/traces").exists())
        self.assertFalse((self.root / ".autoresearch/pareto_frontier.json").exists())

    def test_check_only_preserves_existing_evaluation_outputs(self):
        trace = self.root / ".autoresearch/traces/previous/eval_trace.json"
        trace.parent.mkdir(parents=True)
        trace.write_text('{"preserve": true}\n', encoding="utf-8")
        frontier = self.root / ".autoresearch/pareto_frontier.json"
        frontier.write_text('[{"preserve": true}]\n', encoding="utf-8")
        before = self.snapshot()
        result = self.run_evaluator("--check-only", "--candidate", "must-not-be-recorded")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_failed_check_only_also_creates_no_files(self):
        (self.root / "AGENTS.md").unlink()
        before = self.snapshot()
        result = self.run_evaluator("--check-only")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("MISSING_CONSTITUTION_MIRROR: AGENTS.md", result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_normal_evaluation_still_records_trace_and_frontier(self):
        result = self.run_evaluator("--candidate", "regression-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        traces = list((self.root / ".autoresearch/traces").glob("*/eval_trace.json"))
        self.assertEqual(len(traces), 1)
        record = json.loads(traces[0].read_text(encoding="utf-8"))
        self.assertEqual(record["candidate_id"], "regression-test")
        self.assertEqual(record["task_vector"]["__static_architecture__"], 1.0)
        frontier = json.loads((self.root / ".autoresearch/pareto_frontier.json").read_text(encoding="utf-8"))
        self.assertEqual(frontier[0]["candidate_id"], "regression-test")

    def test_task_failures_record_vectors_and_return_failure(self):
        tasks = [
            {"task_id": "pass", "test_argv": ["{python}", "-c", "print('ok')"]},
            {"task_id": "fail", "test_argv": ["{python}", "-c", "raise SystemExit(3)"]},
        ]
        (self.root / ".autoresearch/tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        result = self.run_evaluator("--candidate", "mixed")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("[FAIL]", result.stdout)
        frontier = json.loads((self.root / ".autoresearch/pareto_frontier.json").read_text(encoding="utf-8"))
        self.assertEqual(frontier[0]["task_vector"], {"__static_architecture__": 1.0, "pass": 1.0, "fail": 0.0})
        self.assertTrue(list((self.root / ".autoresearch/traces").glob("*/pass/stdout.log")))

    def test_check_only_never_launches_configured_tasks(self):
        marker = self.root / "must-not-exist"
        tasks = [{"task_id": "sentinel", "test_argv": ["{python}", "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]}]
        (self.root / ".autoresearch/tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        before = self.snapshot()
        result = self.run_evaluator("--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(marker.exists())

    def test_malformed_tasks_fail_without_updating_frontier(self):
        (self.root / ".autoresearch/tasks.json").write_text("{broken", encoding="utf-8")
        result = self.run_evaluator("--candidate", "invalid")
        self.assertEqual(result.returncode, 1)
        self.assertIn("BEHAVIORAL_TASK_CONFIG", result.stderr)
        self.assertFalse((self.root / ".autoresearch/pareto_frontier.json").exists())

    def test_missing_process_helper_is_reported_by_check_only(self):
        (self.root / ".autoresearch/task_process.py").unlink()
        result = self.run_evaluator("--check-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn("MISSING_REQUIRED_FILE", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node required for end-to-end fixture grading")
    def test_example_headless_tasks_record_real_pareto_dimensions(self):
        conditions = self.gates_config["gates"]["prompt_skill_router"]["conditions"]
        if not any(item["id"] == "typescript-browser-apps" for item in conditions):
            self.skipTest("Optional browser domain pack is not registered")
        bundle = Path(".context/evals/typescript-browser-apps")
        shutil.copytree(REPO_ROOT / bundle, self.root / bundle)
        shutil.copyfile(REPO_ROOT / ".autoresearch/tasks.example.json", self.root / ".autoresearch/tasks.json")
        runner = {"argv": [sys.executable, "-c", "print('Fake solver: no changes')"], "timeout_sec": 5}
        (self.root / ".autoresearch/runner.local.json").write_text(json.dumps(runner), encoding="utf-8")
        with patch.dict(os.environ, {"TMP": self.temp.name, "TEMP": self.temp.name, "TMPDIR": self.temp.name}):
            result = self.run_evaluator("--candidate", "headless-integration")
        self.assertEqual(result.returncode, 1, result.stderr)
        frontier = json.loads((self.root / ".autoresearch/pareto_frontier.json").read_text(encoding="utf-8"))
        self.assertEqual(frontier[0]["task_vector"], {
            "__static_architecture__": 1.0, "browser-manifest": 0.0, "browser-storage": 0.0, "browser-camera": 0.0,
        })
        traces = self.root / frontier[0]["trace_dir"]
        for case in ("manifest", "storage", "camera"):
            record = json.loads((traces / f"browser-{case}-result/result.json").read_text(encoding="utf-8"))
            self.assertEqual(record["solver"]["returncode"], 0)
            self.assertNotEqual(record["grader"]["returncode"], 0)
            self.assertFalse(Path(record["workspace"]).is_relative_to(self.root))


class ProviderUrlTests(unittest.TestCase):
    def setUp(self):
        self.module = load_script("gate_runner_under_test", REPO_ROOT / ".harness/gates/run_gate.py")

    def exercise_provider(self, provider: str, operation: str, override: str | None = None):
        if provider == "jev":
            env = {"JEV_API_KEY": "test-key"}
            expected = "https://api.typesafe.ai/v1/jev/decide"
            url_variable = "JEV_API_URL"
        else:
            env = {"OPENAI_API_KEY": "test-key"}
            expected = "https://api.openai.com/v1/decisions"
            url_variable = "OPENAI_DECISIONS_URL"
        if override:
            env[url_variable] = expected = override
        probabilities = {"complete": 0.95, "pending": 0.05}
        if operation == "predicates":
            probabilities = {"debugger": 0.91} if provider == "jev" else {"true": 0.91, "false": 0.09}
        response_body = {"probabilities": probabilities}
        if provider == "openai":
            answer = {
                "type": "choice", "name": "classification", "choice": "complete", "confidence": 0.95,
                "probabilities": [{"value": name, "probability": probability} for name, probability in probabilities.items()],
            }
            if operation == "predicates":
                answer = {"type": "predicate", "name": "debugger", "probability": 0.91}
            response_body = {"answers": [answer]}
        with patch.dict(os.environ, env, clear=True), patch.object(self.module.urllib.request, "urlopen") as request:
            response = request.return_value.__enter__.return_value
            response.read.return_value = json.dumps(response_body).encode("utf-8")
            if operation == "choice":
                result = self.module.classify_choice("Work complete.", "Outcome?", ["complete", "pending"])
                self.assertEqual(result["top_choice"], "complete")
            else:
                result = self.module.classify_predicates("Debug a crash.", [{"id": "debugger", "condition": "A bug report."}])
                self.assertEqual(result["debugger"], 0.91)
            self.assertEqual(request.call_count, 1)
            http_request = request.call_args.args[0]
            self.assertEqual(http_request.full_url, expected)
            self.assertEqual(http_request.type, "https")
            self.assertEqual(http_request.get_method(), "POST")

    def test_all_provider_defaults_are_plain_https_urls(self):
        for provider in ("jev", "openai"):
            for operation in ("choice", "predicates"):
                with self.subTest(provider=provider, operation=operation):
                    self.exercise_provider(provider, operation)

    def test_provider_url_overrides_are_preserved(self):
        for provider in ("jev", "openai"):
            for operation in ("choice", "predicates"):
                with self.subTest(provider=provider, operation=operation):
                    self.exercise_provider(provider, operation, "https://classifier.example.test/custom")

    def test_schema_is_a_plain_https_uri(self):
        config = json.loads((REPO_ROOT / ".harness/gates/decision-gates.json").read_text(encoding="utf-8"))
        parsed = urlsplit(config["$schema"])
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "typesafe.ai")


if __name__ == "__main__":
    unittest.main()
