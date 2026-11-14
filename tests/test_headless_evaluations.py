from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / ".context/evals/typescript-browser-apps"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TaskExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="evaluation process spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.process = load_script("test_task_process", ROOT / ".autoresearch/task_process.py")
        self.evaluator = load_script("test_headless_evaluator", ROOT / ".autoresearch/prepare_and_eval.py")
        self.evaluator.ROOT = self.root
        self.evaluator.TASKS_FILE = self.root / "tasks.json"

    def write_tasks(self, tasks):
        self.evaluator.TASKS_FILE.write_text(json.dumps(tasks), encoding="utf-8")

    def test_argv_preserves_unicode_and_shell_metacharacters(self):
        payload = "camera café ; echo should-not-run & ${literal}"
        result = self.process.run_process(
            [sys.executable, "-c", "import sys; print(sys.argv[1]); print(sys.stdin.read()); print('diagnostic', file=sys.stderr)", payload],
            self.root, 5, self.root / "logs", "input ✓",
        )
        self.assertTrue(self.process.passed(result))
        self.assertEqual((self.root / "logs/stdout.log").read_text(encoding="utf-8").splitlines(), [payload, "input ✓"])
        self.assertIn("diagnostic", (self.root / "logs/stderr.log").read_text(encoding="utf-8"))

    def test_timeout_retains_output_and_scores_zero_then_continues(self):
        self.write_tasks([
            {"task_id": "slow", "test_argv": ["{python}", "-c", "import time; print('started', flush=True); time.sleep(30)"], "timeout_sec": 1},
            {"task_id": "next", "test_argv": ["{python}", "-c", "print('finished')"]},
        ])
        run_dir = self.root / "traces"
        self.assertEqual(self.evaluator.run_behavioral_tasks(run_dir), {"slow": 0.0, "next": 1.0})
        trace = json.loads((run_dir / "trace_slow.log").read_text(encoding="utf-8"))
        self.assertEqual(trace["status"], "timed_out")
        self.assertIn("started", (run_dir / "slow/stdout.log").read_text(encoding="utf-8"))

    def test_launch_failure_retains_diagnostic_and_next_task_runs(self):
        self.write_tasks([
            {"task_id": "missing", "test_argv": [str(self.root / "missing-agent.exe")]},
            {"task_id": "failed", "test_argv": ["{python}", "-c", "raise SystemExit(7)"]},
        ])
        run_dir = self.root / "traces"
        self.assertEqual(self.evaluator.run_behavioral_tasks(run_dir), {"missing": 0.0, "failed": 0.0})
        self.assertTrue((run_dir / "missing/stderr.log").read_text(encoding="utf-8"))

    def test_timeout_stops_spawned_child_process(self):
        marker = self.root / "child-survived"
        child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text('leak')"
        parent = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
        result = self.process.run_process([sys.executable, "-c", parent], self.root, 1, self.root / "logs")
        self.assertEqual(result["status"], "timed_out")
        time.sleep(1.5)
        self.assertFalse(marker.exists(), "Timed-out task left a running child process")

    def test_legacy_shell_command_remains_supported(self):
        self.write_tasks([{"task_id": "legacy", "test_command": f'"{sys.executable}" -c "print(123)"'}])
        self.assertEqual(self.evaluator.run_behavioral_tasks(self.root / "traces"), {"legacy": 1.0})

    def test_argv_placeholders_are_literal_substitutions(self):
        argv = self.evaluator._task_command({"test_argv": ["{python}", "{repo_root}", "{trace_dir}/case"]}, self.root / "traces")
        self.assertEqual(argv, [sys.executable, str(self.root), str(self.root / "traces") + "/case"])

    def test_absent_tasks_is_opt_in(self):
        self.assertEqual(self.evaluator.run_behavioral_tasks(self.root / "traces"), {})
        self.assertFalse((self.root / "traces").exists())

    def test_invalid_config_never_starts_partial_task_set(self):
        valid = {"task_id": "ok", "test_argv": ["{python}", "-c", "print(1)"]}
        invalid = (
            {}, "bad", [None], [{"task_id": "../escape", "test_command": "echo hi"}],
            [valid, valid], [{**valid, "test_command": "echo hi"}],
            [{**valid, "test_argv": "python"}], [{**valid, "test_argv": [""]}],
            [{**valid, "test_argv": []}], [{**valid, "test_argv": ["nul\0arg"]}], [{"task_id": "no-command"}],
            [{"task_id": "empty-command", "test_command": ""}],
            [valid, {"task_id": "invalid", "test_argv": [1]}],
        )
        for tasks in invalid:
            with self.subTest(tasks=tasks), patch.object(self.evaluator.task_process, "run_process") as launch:
                self.write_tasks(tasks)
                with self.assertRaises(ValueError):
                    self.evaluator.run_behavioral_tasks(self.root / "traces")
                launch.assert_not_called()

    def test_invalid_timeouts_rejected(self):
        for timeout in (0, -1, True, 1.5, "10", 86401, None):
            with self.subTest(timeout=timeout):
                self.write_tasks([{"task_id": "bad", "test_argv": ["unused"], "timeout_sec": timeout}])
                with self.assertRaises(ValueError):
                    self.evaluator.run_behavioral_tasks(self.root / "traces")


class HeadlessSkillTests(unittest.TestCase):
    def setUp(self):
        registry = json.loads((ROOT / ".harness/gates/decision-gates.json").read_text(encoding="utf-8"))
        if not any(item["id"] == "typescript-browser-apps" for item in registry["gates"]["prompt_skill_router"]["conditions"]):
            self.skipTest("Optional browser domain pack is not registered")
        self.temp = tempfile.TemporaryDirectory(prefix="headless evaluation spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = load_script("test_skill_runner", BUNDLE / "run.py")
        self.script = self.root / "fake solver.py"
        self.config = self.root / "runner.json"

    def configure(self, source, timeout=5):
        self.script.write_text(source, encoding="utf-8")
        self.config.write_text(json.dumps({"argv": [sys.executable, str(self.script), "{workspace}", "{request_file}"], "timeout_sec": timeout}), encoding="utf-8")

    def run_case(self, case="manifest"):
        output = self.root / "results"
        with patch.object(self.runner.tempfile, "mkdtemp", return_value=str(self.root / "prepared")), redirect_stdout(StringIO()):
            status = self.runner.headless(case, self.config, output)
        return status, json.loads((output / "result.json").read_text(encoding="utf-8"))

    def test_failed_solver_is_not_graded_and_guidance_is_snapshotted(self):
        self.configure("import sys; print(sys.stdin.read()); raise SystemExit(9)")
        status, result = self.run_case()
        self.assertEqual(status, 1)
        self.assertEqual(result["solver"]["returncode"], 9)
        self.assertIsNone(result["grader"])
        workspace = Path(result["workspace"])
        specialist = ".harness/specialists/typescript-browser-apps.md"
        self.assertEqual((workspace / specialist).read_bytes(), (ROOT / specialist).read_bytes())
        self.assertTrue((workspace / ".context/references/typescript-browser-apps/browser-runtime.md").is_file())
        self.assertFalse((workspace / ".context/evals").exists())
        self.assertFalse((workspace / "checks").exists())
        self.assertNotIn(str(ROOT), (workspace / "REQUEST.md").read_text(encoding="utf-8"))
        self.assertEqual(result["solver"]["command"][-2:], [str(workspace), str(workspace / "REQUEST.md")])

    def test_timed_out_solver_is_not_graded(self):
        self.configure("import time; time.sleep(30)", timeout=1)
        status, result = self.run_case()
        self.assertEqual(status, 1)
        self.assertEqual(result["solver"]["status"], "timed_out")
        self.assertIsNone(result["grader"])

    def test_missing_solver_is_logged(self):
        self.config.write_text(json.dumps({"argv": [str(self.root / "not-installed.exe")]}), encoding="utf-8")
        status, result = self.run_case()
        self.assertEqual(status, 1)
        self.assertEqual(result["solver"]["status"], "launch_error")

    def test_invalid_runner_config_creates_no_workspace(self):
        for value in ([], {}, {"argv": "codex"}, {"argv": []}, {"argv": [1]}, {"argv": [""]}, {"argv": ["nul\0arg"]}, {"argv": ["unused"], "timeout_sec": True}):
            with self.subTest(config=value):
                self.config.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.runner.headless("manifest", self.config, self.root / "output")
                self.assertFalse((self.root / "output").exists())

    def test_existing_output_is_never_overwritten(self):
        self.configure("pass")
        output = self.root / "results"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.runner.headless("manifest", self.config, output)

    @unittest.skipUnless(shutil.which("node"), "Node required for real fixture grading")
    def test_noop_solver_fails_real_grading(self):
        self.configure("print('No changes')")
        status, result = self.run_case()
        self.assertEqual(status, 1)
        self.assertNotEqual(result["grader"]["returncode"], 0)
        self.assertIn("ENOENT", (self.root / "results/grader/stdout.log").read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("node"), "Node required for real fixture grading")
    def test_fake_solver_solution_passes_real_grading(self):
        self.configure('''import json, pathlib, subprocess, sys
assert str(pathlib.Path.cwd()) == sys.argv[1]
assert pathlib.Path(sys.argv[2]).read_text(encoding="utf-8") == sys.stdin.read()
path = pathlib.Path("public/manifest.webmanifest")
manifest = json.loads(path.read_text())
manifest.update(name="Field Notes", short_name="Notes", display="standalone", start_url="/field/?mode=camera", id="/field/", scope="/field/")
path.write_text(json.dumps(manifest))
subprocess.run(["node", "build.mjs"], check=True)
''')
        before = {str(p.relative_to(BUNDLE)): p.read_bytes() for p in BUNDLE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        status, result = self.run_case()
        self.assertEqual(status, 0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["grader"]["returncode"], 0)
        after = {str(p.relative_to(BUNDLE)): p.read_bytes() for p in BUNDLE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
