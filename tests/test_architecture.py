"""Enforce module boundaries so future features preserve the refactored design."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest


PACKAGE = Path(__file__).resolve().parents[1] / "src/sts2_advisor"


def module_sources():
    return {"sts2_advisor." + ".".join(p.relative_to(PACKAGE).with_suffix("").parts):
            ast.parse(p.read_text(encoding="utf-8")) for p in PACKAGE.rglob("*.py")
            if p.name != "__init__.py"}


def imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


class ArchitectureTests(unittest.TestCase):
    def test_game_rules_depend_only_on_game_contracts_and_standard_library(self):
        for module, tree in module_sources().items():
            if not module.startswith("sts2_advisor.game."):
                continue
            for dependency in imports(tree):
                allowed = (dependency == "sts2_advisor.contracts"
                           or dependency.startswith("sts2_advisor.game.")
                           or dependency.split(".")[0] in sys.stdlib_module_names)
                self.assertTrue(allowed, f"{module} must not depend on {dependency}")

    def test_workflow_and_decisions_do_not_import_concrete_integrations(self):
        for module, tree in module_sources().items():
            if module.startswith(("sts2_advisor.application.", "sts2_advisor.decision.")):
                for dependency in imports(tree):
                    self.assertFalse(dependency.startswith("sts2_advisor.integrations."),
                                     f"Inject interfaces into {module}, not {dependency}")

    def test_recording_does_not_depend_on_decision_or_game_implementation(self):
        for module, tree in module_sources().items():
            if module.startswith("sts2_advisor.recording."):
                for dependency in imports(tree):
                    if dependency.startswith("sts2_advisor."):
                        self.assertTrue(dependency == "sts2_advisor.contracts"
                                        or dependency.startswith("sts2_advisor.recording."),
                                        f"{module} must not depend on {dependency}")

    def test_internal_import_graph_has_no_cycles_or_missing_modules(self):
        sources = module_sources()
        packages = {"sts2_advisor." + ".".join(p.parent.relative_to(PACKAGE).parts)
                    for p in PACKAGE.rglob("__init__.py")}
        graph = {}
        for module, tree in sources.items():
            graph[module] = set()
            for dependency in imports(tree):
                if dependency.startswith("sts2_advisor."):
                    self.assertTrue(dependency in sources or dependency in packages,
                                    f"{module} imports missing {dependency}")
                    if dependency in sources:
                        graph[module].add(dependency)
        completed = set()

        def visit(module, path):
            self.assertNotIn(module, path, " -> ".join([*path, module]))
            if module in completed:
                return
            for dependency in graph[module]:
                visit(dependency, [*path, module])
            completed.add(module)

        for module in graph:
            visit(module, [])

    def test_cli_import_does_not_load_optional_model_or_database_sdks(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys,json; import sts2_advisor.cli; "
             "print(json.dumps(sorted(set(sys.modules) & {'torch','transformers','openai','chromadb'})))"],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_legacy_module_commands_still_offer_help(self):
        for module in ("sts2_advisor.record", "sts2_advisor.ingest"):
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-m", module, "--help"],
                                        capture_output=True, text=True, encoding="utf-8", timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
