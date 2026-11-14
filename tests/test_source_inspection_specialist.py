"""Routing and retrieval for the optional TypeScript source-inspection pack."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_template_validation import REPO_ROOT, load_script

REGISTRY = REPO_ROOT / '.harness/gates/decision-gates.json'
CONFIG = json.loads(REGISTRY.read_text(encoding='utf-8'))
CONDITIONS = CONFIG['gates']['prompt_skill_router']['conditions']
PACK_ID = 'typescript-source-inspection'
SPECIALIST = f'.harness/specialists/{PACK_ID}.md'
MANUAL = f'.context/references/{PACK_ID}/package-inspection.md'


@unittest.skipUnless(any(item['id'] == PACK_ID for item in CONDITIONS),
                     'Optional source-inspection pack is not registered')
class SourceInspectionSpecialistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routing = load_script('inspection_routing', REPO_ROOT / '.harness/gates/routing.py')
        cls.gate = load_script('inspection_gate', REPO_ROOT / '.harness/gates/run_gate.py')

    def matched(self, prompt):
        with patch.dict(os.environ, {}, clear=True):
            scores = self.gate.classify_predicates(prompt, CONDITIONS)
        threshold = CONFIG['gates']['prompt_skill_router']['threshold']
        return {name for name, score in scores.items() if score >= threshold}

    def context_server(self):
        server = SimpleNamespace(tool=lambda: lambda function: function)
        stub = SimpleNamespace(FastMCP=lambda name: server)
        with patch.dict('sys.modules', {'mcp.server.fastmcp': stub}):
            return load_script('inspection_context', REPO_ROOT / '.context/mcp_context_server.py')

    def test_installed_surface_and_declaration_prompts_route_to_the_pack(self):
        prompts = (
            'Check how the installed package implements retries in node_modules.',
            'Why does lib.webworker lack this fetch overload?',
            'fetch throws Illegal invocation inside the worker bundle.',
            'Verify the package exports map before changing the import.',
            'Review src/types/global.d.ts against the runtime implementation.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIn(PACK_ID, self.matched(prompt))
                self.assertEqual(self.routing.recommend(REPO_ROOT, prompt)['specialist'], SPECIALIST)

    def test_ordinary_application_edits_do_not_load_the_pack(self):
        prompts = (
            'Rename the helper in src/app.ts.',
            'Fix the button color and padding.',
            'Add a unit test for the parser.',
            'Update the readme wording about installation.',
            'Refactor the node of the linked list.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertNotIn(PACK_ID, self.matched(prompt))

    def test_browser_runtime_ownership_outranks_inspection_on_mixed_prompts(self):
        prompt = 'Inspect node_modules types for the IndexedDB wrapper.'
        matches = self.matched(prompt)
        self.assertIn(PACK_ID, matches)
        self.assertIn('typescript-browser-apps', matches)
        self.assertEqual(self.routing.recommend(REPO_ROOT, prompt)['specialist'],
                         '.harness/specialists/typescript-browser-apps.md')

    def test_manual_is_registered_cold_memory_and_searchable(self):
        condition = next(item for item in CONDITIONS if item['id'] == PACK_ID)
        self.assertEqual(condition['cold_memory'], MANUAL)
        paths = {item['path'].replace('\\', '/')
                 for item in self.context_server().find_relevant_context('exports map declaration drift')}
        self.assertIn(MANUAL, paths)

    def test_specialist_links_resolve_to_registered_manual_sections(self):
        text = (REPO_ROOT / SPECIALIST).read_text(encoding='utf-8')
        manual = (REPO_ROOT / MANUAL).read_text(encoding='utf-8')
        for anchor in ('resolve-the-installed-surface', 'declaration-drift', 'compiler-lib-boundaries',
                       'web-idl-and-node-boundary-semantics', 'minimal-runtime-probes'):
            with self.subTest(anchor=anchor):
                self.assertIn(f'package-inspection.md#{anchor}', text)
                heading = '# ' + anchor.replace('-', ' ')
                self.assertIn(heading.casefold(), manual.casefold())


if __name__ == '__main__':
    unittest.main()
