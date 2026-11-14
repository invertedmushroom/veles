from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_template_validation import REPO_ROOT, load_script


class DomainRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="domain-registry-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = load_script("domain_registry", REPO_ROOT / ".harness/gates/routing.py")
        self.fallback = {
            "id": "solver", "condition": "Implement a feature.", "keywords": ["implement"],
            "inject_path": ".harness/specialists/cmcts-solver.md",
            "cold_memory": ".context/architecture.l1-l3.md", "fallback": True,
            "paradigm": "CHUNKED_SYMBOLISM",
        }
        self.domain = {
            "id": "ownership-domain", "condition": "Resolve borrow checker boundaries.",
            "keywords": ["borrow checker"], "globs": ["crates/**/*.rs"], "priority": 100,
            "inject_path": ".harness/specialists/ownership.md",
            "cold_memory": ".context/references/ownership/main.md",
            "references": [".context/references/ownership/async.md"],
            "paradigm": "EXPERT_LEXICON",
        }
        self.conditions = [self.fallback, self.domain]
        self.write_registry()

    def write_registry(self):
        path = self.root / ".harness/gates/decision-gates.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"gates": {"prompt_skill_router": {"conditions": self.conditions}}}), encoding="utf-8")

    def context_server(self):
        server = SimpleNamespace(tool=lambda: lambda function: function)
        stub = SimpleNamespace(FastMCP=lambda name: server)
        with patch.dict("sys.modules", {"mcp.server.fastmcp": stub}):
            module = load_script("context_registry_test", REPO_ROOT / ".context/mcp_context_server.py")
        module.REPO_ROOT = self.root
        module.CONTEXT_DIR = self.root / ".context"
        module.SUBSYSTEMS_DIR = module.CONTEXT_DIR / "subsystems"
        module.SYMPTOM_FILE = module.CONTEXT_DIR / "symptom-cause-fix.md"
        return module

    def test_new_domain_routes_by_keywords_without_python_changes(self):
        result = self.registry.recommend(self.root, "Debug a borrow checker error")
        self.assertEqual(result["specialist"], self.domain["inject_path"])
        self.assertEqual(result["cold_memory"], self.domain["cold_memory"])
        self.assertEqual(result["paradigm"], "EXPERT_LEXICON")
        self.assertEqual(self.context_server().suggest_agent("borrow checker"), result)

    def test_paths_match_registered_globs_on_windows_and_posix(self):
        for signal in ("crates/core/src/lib.rs", "Review `crates/core/src/lib.rs`.", r"crates\core\src\lib.rs"):
            with self.subTest(signal=signal):
                self.assertTrue(self.registry.matches(signal, self.domain))
                self.assertEqual(self.registry.offline_score(signal, self.domain), 1.0)
                self.assertEqual(self.registry.recommend(self.root, signal)["specialist"], self.domain["inject_path"])

    def test_leading_globstar_matches_root_and_nested_paths(self):
        self.domain.update(keywords=[], globs=["**/artifact.rule"])
        self.write_registry()
        for signal in ("artifact.rule", "nested/artifact.rule", "deep/nested/artifact.rule", r"nested\artifact.rule", "Review `ARTIFACT.RULE`."):
            with self.subTest(signal=signal):
                self.assertTrue(self.registry.matches(signal, self.domain))
                self.assertEqual(self.registry.offline_score(signal, self.domain), 1.0)
                self.assertEqual(self.context_server().suggest_agent(signal)["specialist"], self.domain["inject_path"])
        for signal in ("artifacts.rule", "nested/artifact.rules", "unrelated.txt"):
            with self.subTest(signal=signal):
                self.assertFalse(self.registry.matches(signal, self.domain))

    def test_unrelated_substrings_use_registered_fallback(self):
        self.assertFalse(self.registry.matches("borrow checkers", self.domain))
        self.assertEqual(self.registry.recommend(self.root, "plain prose")["specialist"], self.fallback["inject_path"])

    def test_priority_and_configuration_order_resolve_single_recommendation(self):
        self.fallback.update(keywords=["borrow checker"], priority=101)
        self.write_registry()
        self.assertEqual(self.registry.recommend(self.root, "borrow checker")["specialist"], self.fallback["inject_path"])
        self.domain["priority"] = 101
        self.write_registry()
        self.assertEqual(self.registry.recommend(self.root, "borrow checker")["specialist"], self.fallback["inject_path"])

    def test_removing_registration_removes_recommendation(self):
        self.conditions.remove(self.domain)
        self.write_registry()
        self.assertEqual(self.registry.recommend(self.root, "borrow checker")["specialist"], self.fallback["inject_path"])

    def test_invalid_metadata_is_rejected(self):
        for field, value in (("keywords", "text"), ("globs", [42]), ("references", "path"), ("priority", True), ("fallback", 1), ("paradigm", "unknown")):
            with self.subTest(field=field):
                original = dict(self.domain)
                self.domain[field] = value
                self.write_registry()
                with self.assertRaises(ValueError):
                    self.registry.read_conditions(self.root)
                self.domain.clear()
                self.domain.update(original)

    def test_duplicate_ids_and_fallbacks_are_rejected(self):
        self.conditions.append(dict(self.domain))
        self.write_registry()
        with self.assertRaises(ValueError):
            self.registry.read_conditions(self.root)
        self.conditions.pop()
        self.domain["fallback"] = True
        self.write_registry()
        with self.assertRaises(ValueError):
            self.registry.read_conditions(self.root)

    def test_reference_search_is_separate_from_subsystem_listing(self):
        for relative in (self.domain["cold_memory"], *self.domain["references"], ".context/references/unregistered.md"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Ownership guidance\nBorrow checker ownership rules.\n", encoding="utf-8")
        subsystem = self.root / ".context/subsystems/tooling.md"
        subsystem.parent.mkdir(parents=True)
        subsystem.write_text("# Tooling\n@META subsystem=tooling\n", encoding="utf-8")
        module = self.context_server()
        self.assertEqual([item["subsystem"] for item in module.list_subsystems()], ["tooling"])
        matches = {item["path"].replace("\\", "/") for item in module.find_relevant_context("ownership")}
        self.assertEqual(matches, {self.domain["cold_memory"], *self.domain["references"]})
        self.conditions.remove(self.domain)
        self.write_registry()
        self.assertEqual(module.find_relevant_context("ownership"), [])

    def test_escaping_paths_and_invalid_registry_do_not_break_contract_lookup(self):
        self.domain["cold_memory"] = "../outside.md"
        self.write_registry()
        with self.assertRaises(ValueError):
            self.registry.registered_files(self.root)
        module = self.context_server()
        self.assertEqual(module.suggest_agent("borrow checker")["specialist"], ".harness/specialists/cmcts-solver.md")
        contract = self.root / ".context/subsystems/tooling.md"
        contract.parent.mkdir(parents=True)
        contract.write_text("# Tooling\nLocal contract lookup remains available.\n", encoding="utf-8")
        (self.root / ".harness/gates/decision-gates.json").write_text("{broken", encoding="utf-8")
        self.assertEqual(len(module.find_relevant_context("local contract")), 1)

    def test_directory_reference_does_not_break_contract_lookup(self):
        (self.root / self.domain["cold_memory"]).mkdir(parents=True)
        contract = self.root / ".context/subsystems/tooling.md"
        contract.parent.mkdir(parents=True)
        contract.write_text("# Tooling\nOwnership contract lookup.\n", encoding="utf-8")
        result = self.context_server().find_relevant_context("ownership")
        self.assertEqual([item["path"].replace("\\", "/") for item in result], [".context/subsystems/tooling.md"])


if __name__ == "__main__":
    unittest.main()
