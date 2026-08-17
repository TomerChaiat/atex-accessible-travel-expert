"""Module names must be identical across the diagram, the trace, and the docs.

The assignment calls this out explicitly, and it is the kind of consistency that
rots silently during refactoring, so it gets its own test file.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

from atex import MODULE_NAMES, api  # noqa: E402
from atex.graph import MODULES, run_agent  # noqa: E402


class TestModuleNames(unittest.TestCase):
    def test_dispatch_table_matches_the_canonical_names(self):
        # Supervisor routes; it is not a routable destination.
        routable = set(MODULE_NAMES) - {"Supervisor"}
        self.assertEqual(set(MODULES), routable)

    def test_trace_only_emits_canonical_names(self):
        result = run_agent("Three days in Amsterdam, manual wheelchair, relaxed pace, hotel needed.")
        emitted = {step["module"] for step in result.steps}
        self.assertTrue(emitted)
        self.assertTrue(
            emitted <= set(MODULE_NAMES),
            f"trace emitted unknown modules: {emitted - set(MODULE_NAMES)}",
        )

    def test_all_five_modules_are_exercised_by_a_full_run(self):
        result = run_agent(
            "We are a family of four visiting Amsterdam for three days. Our daughter uses a "
            "manual wheelchair. Relaxed pace, no more than two activities per day. We need a hotel."
        )
        emitted = {step["module"] for step in result.steps}
        self.assertEqual(
            emitted,
            set(MODULE_NAMES),
            f"a full run should touch every module; missing {set(MODULE_NAMES) - emitted}",
        )

    def test_agent_info_advertises_the_same_names(self):
        _, body = api.agent_info()
        self.assertEqual(set(body["modules"]), set(MODULE_NAMES))

    def test_diagram_is_generated_from_the_canonical_constants(self):
        """The PNG script must import the names, not retype them."""
        source = (ROOT / "scripts" / "build_architecture_png.py").read_text(encoding="utf-8")
        self.assertIn("from atex import", source)
        for name in MODULE_NAMES:
            self.assertNotIn(
                f'"{name}"',
                source,
                f"{name} is hard-coded in the diagram script; import the constant instead",
            )

    def test_prompts_reference_the_same_names(self):
        from atex.prompts import SUPERVISOR_SYSTEM

        for name in set(MODULE_NAMES) - {"Supervisor"}:
            self.assertIn(name, SUPERVISOR_SYSTEM)

    def test_documentation_uses_only_canonical_spellings(self):
        """The assignment requires consistent names in *any* description we give.

        Prose drifts more easily than code, and a design document that renames
        the modules is a gradeable inconsistency, so the docs are checked too.
        """
        forbidden = (
            "Supervisor Agent",
            "User Profile Agent",
            "Activity & Logistics Finder",
            "Activity and Logistics Finder",
            "Accessibility Validator",
            "Schedule Planner",
            "Activity Finder",
        )

        offences: list[str] = []
        for path in sorted(ROOT.rglob("*.md")):
            if ".git" in path.parts:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                for phrase in forbidden:
                    if phrase in line:
                        rel = path.relative_to(ROOT).as_posix()
                        offences.append(f"{rel}:{lineno} uses {phrase!r}")

        self.assertEqual(
            offences,
            [],
            "documentation must use the canonical module names:\n  "
            + "\n  ".join(offences),
        )


if __name__ == "__main__":
    unittest.main()
