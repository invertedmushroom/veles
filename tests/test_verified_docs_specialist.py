"""Routing and retrieval for the optional verified-docs pack."""

import json
import os
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_template_validation import REPO_ROOT, load_script

REGISTRY = REPO_ROOT / '.harness/gates/decision-gates.json'
CONFIG = json.loads(REGISTRY.read_text(encoding='utf-8'))
CONDITIONS = CONFIG['gates']['prompt_skill_router']['conditions']
PACK_ID = 'verified-docs'
SPECIALIST = f'.harness/specialists/{PACK_ID}.md'
MANUAL = f'.context/references/{PACK_ID}/authoring.md'

FENCED_RE = re.compile(r'```.*?```', re.DOTALL)
LINK_RE = re.compile(r'\[[^\]]+\]\(([^)\s]+)\)')
HEADING_RE = re.compile(r'^#{1,6}\s+(.+?)\s*$', re.MULTILINE)


def github_slug(heading):
    clean = re.sub(r'`([^`]*)`', r'\1', heading.strip().lower())
    clean = re.sub(r'[^\w\s-]', '', clean)
    return re.sub(r'\s+', '-', clean)


@unittest.skipUnless(any(item['id'] == PACK_ID for item in CONDITIONS),
                     'Optional verified-docs pack is not registered')
class VerifiedDocsSpecialistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routing = load_script('docs_routing', REPO_ROOT / '.harness/gates/routing.py')
        cls.gate = load_script('docs_gate', REPO_ROOT / '.harness/gates/run_gate.py')

    def matched(self, prompt):
        with patch.dict(os.environ, {}, clear=True):
            scores = self.gate.classify_predicates(prompt, CONDITIONS)
        threshold = CONFIG['gates']['prompt_skill_router']['threshold']
        return {name for name, score in scores.items() if score >= threshold}

    def context_server(self):
        server = SimpleNamespace(tool=lambda: lambda function: function)
        stub = SimpleNamespace(FastMCP=lambda name: server)
        with patch.dict('sys.modules', {'mcp.server.fastmcp': stub}):
            return load_script('docs_context', REPO_ROOT / '.context/mcp_context_server.py')

    def test_explicit_documentation_requests_route_to_the_pack(self):
        prompts = (
            'Update the README quickstart commands.',
            'Write an onboarding guide for new contributors.',
            'Document the CLI for humans.',
            'Draft the release notes for the next version.',
            'Create getting started docs for the runtime.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIn(PACK_ID, self.matched(prompt))
                self.assertEqual(self.routing.recommend(REPO_ROOT, prompt)['specialist'], SPECIALIST)

    def test_implementation_and_contract_tasks_do_not_load_the_pack(self):
        prompts = (
            'Implement the retry logic for the worker.',
            'Fix the failing storage test.',
            'Refactor the session module.',
            'Update the subsystem contract for routing.',
            'Design the system architecture.',
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertNotIn(PACK_ID, self.matched(prompt))

    def test_manual_is_registered_cold_memory_and_searchable(self):
        condition = next(item for item in CONDITIONS if item['id'] == PACK_ID)
        self.assertEqual(condition['cold_memory'], MANUAL)
        paths = {item['path'].replace('\\', '/')
                 for item in self.context_server().find_relevant_context('claim verification sink rules')}
        self.assertIn(MANUAL, paths)

    def test_article_request_outranks_docs_pack_on_mixed_prompts(self):
        prompt = 'Write a technical article walkthrough of the onboarding flow.'
        if not any(item['id'] == 'concept-article' for item in CONDITIONS):
            self.skipTest('concept-article pack not registered')
        matches = self.matched(prompt)
        self.assertIn(PACK_ID, matches)
        self.assertIn('concept-article', matches)
        self.assertEqual(self.routing.recommend(REPO_ROOT, prompt)['specialist'],
                         '.harness/specialists/concept-article.md')

    def test_readme_links_resolve_and_docs_stays_markdown_free(self):
        readme = REPO_ROOT / 'README.md'
        stripped = FENCED_RE.sub('', readme.read_text(encoding='utf-8'))
        for target in LINK_RE.findall(stripped):
            if target.startswith(('http://', 'https://', 'mailto:')):
                continue
            path_part, _, anchor = target.partition('#')
            resolved = (REPO_ROOT / path_part).resolve() if path_part else readme.resolve()
            with self.subTest(link=target):
                self.assertTrue(resolved.is_relative_to(REPO_ROOT.resolve()))
                self.assertTrue(resolved.is_file(), f'Broken README link: {target}')
                if anchor and resolved.suffix.lower() == '.md':
                    headings = {github_slug(match.group(1))
                                for match in HEADING_RE.finditer(resolved.read_text(encoding='utf-8'))}
                    self.assertIn(anchor.lower(), headings, f'Missing anchor #{anchor} in {path_part}')
        docs_dir = REPO_ROOT / 'docs'
        if docs_dir.exists():
            self.assertEqual(list(docs_dir.rglob('*.md')), [],
                             'Narrative Markdown is forbidden in docs/; human deep-dives are standalone HTML')


if __name__ == '__main__':
    unittest.main()
