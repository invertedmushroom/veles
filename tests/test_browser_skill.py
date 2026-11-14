from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPECIALIST = ".harness/specialists/typescript-browser-apps.md"
REFERENCES = (".context/references/typescript-browser-apps/browser-runtime.md", ".context/references/typescript-browser-apps/pwa-delivery.md")
EVAL_ROOT = ROOT / ".context/evals/typescript-browser-apps"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BrowserSkillTests(unittest.TestCase):
    def setUp(self):
        self.gate = load_script("browser_skill_gate", ROOT / ".harness/gates/run_gate.py")
        self.config = self.gate.load_configuration()
        self.conditions = self.config["gates"]["prompt_skill_router"]["conditions"]
        if not any(item["id"] == "typescript-browser-apps" for item in self.conditions):
            self.skipTest("Optional browser domain pack is not registered")

    def matched(self, prompt):
        with patch.dict(os.environ, {}, clear=True):
            scores = self.gate.classify_predicates(prompt, self.conditions)
        threshold = self.config["gates"]["prompt_skill_router"]["threshold"]
        return {key for key, value in scores.items() if value >= threshold}

    def test_browser_boundaries_route_without_credentials(self):
        prompts = (
            "Fix IndexedDB migration.", "Handle OPFS quota failure.",
            "Dispose Web Workers when the panel closes.", "Update the PWA installation flow.",
            "Release the camera after disposal.", "Restrict iframe access to browser storage.",
            "Change MessagePort input validation.", "Review public/manifest.webmanifest.",
            "Implement service worker update handling.", "Change the web app manifest identity.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIn("typescript-browser-apps", self.matched(prompt))

    def test_unrelated_work_does_not_load_browser_specialist(self):
        for prompt in ("Refactor a TypeScript sorting helper.", "Edit the Cargo manifest.", "Resume the project plan."):
            with self.subTest(prompt=prompt):
                self.assertNotIn("typescript-browser-apps", self.matched(prompt))

    def test_root_and_nested_worker_paths_route_without_keywords(self):
        for prompt in ("sw.js", "service-worker.mjs", "public/sw.js", r"public\service-worker.js", "Review `SW.JS`."):
            with self.subTest(prompt=prompt):
                self.assertIn("typescript-browser-apps", self.matched(prompt))
                self.assertEqual(self.gate.routing.recommend(ROOT, prompt)["specialist"], SPECIALIST)

    def test_browser_and_debugger_can_load_together(self):
        self.assertTrue({"typescript-browser-apps", "grounded-prm-debugger"}.issubset(self.matched("Debug an IndexedDB crash.")))

    def test_prompt_hook_injects_existing_specialist(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            output = StringIO()
            with redirect_stdout(output):
                self.gate.handle_user_prompt_submit({"prompt": "Fix OPFS quota handling."}, self.config, Path(temporary) / "trace.jsonl")
            decision = json.loads(output.getvalue())
        self.assertIn(SPECIALIST, decision["hookSpecificOutput"]["additionalContext"])
        self.assertTrue((ROOT / SPECIALIST).is_file())

    def test_validator_requires_specialist_and_domain_references(self):
        evaluator = load_script("browser_skill_evaluator", ROOT / ".autoresearch/prepare_and_eval.py")
        required = {path.relative_to(ROOT).as_posix() for path in evaluator.get_required_files()}
        self.assertTrue({SPECIALIST, *REFERENCES}.issubset(required))
        self.assertEqual(evaluator.validate_static_architecture(), [])

    def test_eval_preparation_points_to_the_ported_specialist(self):
        evaluator = load_script("browser_skill_fixtures", EVAL_ROOT / "run.py")
        with tempfile.TemporaryDirectory(prefix="browser-skill-test-") as temporary:
            target = Path(temporary) / "tasks"
            output = StringIO()
            with patch.object(evaluator.tempfile, "mkdtemp", return_value=str(target)), redirect_stdout(output):
                evaluator.prepare()
            tasks = json.loads(output.getvalue())["tasks"]
            self.assertEqual({task["id"] for task in tasks}, {"manifest", "storage", "camera"})
            for task in tasks:
                with self.subTest(task=task["id"]):
                    self.assertIn(str(ROOT / SPECIALIST), task["request"])
                    self.assertNotIn(".agents/skills", task["request"])
                    self.assertTrue((Path(task["workspace"]) / "REQUEST.md").is_file())


if __name__ == "__main__":
    unittest.main()
