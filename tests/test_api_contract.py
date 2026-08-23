"""The API contract from the assignment, asserted field by field.

These run on the stdlib alone (`python -m unittest discover tests`) because the
endpoint logic lives in atex/api.py rather than in the FastAPI layer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex import MODULE_NAMES, api  # noqa: E402

EXECUTE_FIELDS = {"status", "error", "response", "steps"}


class TestExecuteContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.status, cls.body = api.execute(
            {"prompt": "Two days in Berlin, I use a powered wheelchair. I like history."}
        )

    def test_http_ok(self):
        self.assertEqual(self.status, 200)

    def test_top_level_fields_match_exactly(self):
        # "must match exactly these top-level fields" -- no more, no less.
        self.assertEqual(set(self.body), EXECUTE_FIELDS)

    def test_success_shape(self):
        self.assertEqual(self.body["status"], "ok")
        self.assertIsNone(self.body["error"])
        self.assertIsInstance(self.body["response"], str)
        self.assertTrue(self.body["response"].strip())
        self.assertIsInstance(self.body["steps"], list)
        self.assertGreater(len(self.body["steps"]), 0)

    def test_every_step_matches_the_required_schema(self):
        for index, step in enumerate(self.body["steps"]):
            with self.subTest(step=index):
                self.assertEqual(set(step), {"module", "prompt", "response"})
                self.assertIn(step["module"], MODULE_NAMES)

                prompt = step["prompt"]
                self.assertIsInstance(prompt, dict)
                # The assignment's required step schema spells these keys
                # "System_prompt"/"User_prompt". See atex/tracing.py.
                self.assertEqual(set(prompt), {"System_prompt", "User_prompt"})
                for key in ("System_prompt", "User_prompt"):
                    self.assertIsInstance(prompt[key], str)
                    self.assertTrue(prompt[key].strip())
                self.assertIsInstance(step["response"], dict)

    def test_json_serialisable(self):
        import json

        json.dumps(self.body)

    def test_debug_flag_is_opt_in_only(self):
        _, plain = api.execute({"prompt": "One day in Berlin."})
        self.assertNotIn("_diagnostics", plain)

        _, debug = api.execute({"prompt": "One day in Berlin."}, debug=True)
        self.assertIn("_diagnostics", debug)
        self.assertIn("usage", debug["_diagnostics"])


class TestExecuteErrors(unittest.TestCase):
    def _assert_error_shape(self, status, body, expected_status=400):
        self.assertEqual(status, expected_status)
        self.assertEqual(set(body), EXECUTE_FIELDS)
        self.assertEqual(body["status"], "error")
        self.assertIsInstance(body["error"], str)
        self.assertTrue(body["error"])
        self.assertIsNone(body["response"])
        self.assertEqual(body["steps"], [])

    def test_missing_prompt(self):
        self._assert_error_shape(*api.execute({}))

    def test_empty_prompt(self):
        self._assert_error_shape(*api.execute({"prompt": "   "}))

    def test_wrong_type(self):
        self._assert_error_shape(*api.execute({"prompt": 42}))

    def test_body_not_an_object(self):
        self._assert_error_shape(*api.execute("not json"))

    def test_overlong_prompt(self):
        self._assert_error_shape(*api.execute({"prompt": "a" * 8001}))


class TestTeamInfo(unittest.TestCase):
    def test_shape(self):
        status, body = api.team_info()
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"group_batch_order_number", "team_name", "students"})
        self.assertIsInstance(body["students"], list)
        self.assertEqual(len(body["students"]), 3)
        for student in body["students"]:
            self.assertEqual(set(student), {"name", "email"})

    def test_placeholders_are_flagged_not_silently_shipped(self):
        _, body = api.team_info()
        placeholders = [s for s in body["students"] if "todo" in s["email"].lower()]
        if placeholders or "TODO" in body["group_batch_order_number"]:
            self.skipTest(
                "data/team.json still holds placeholders - fill it in before submission"
            )


class TestAgentInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.status, cls.body = api.agent_info()

    def test_required_keys(self):
        self.assertEqual(self.status, 200)
        for key in ("description", "purpose", "prompt_template", "prompt_examples"):
            self.assertIn(key, self.body)

    def test_template_and_examples(self):
        self.assertIn("template", self.body["prompt_template"])
        self.assertTrue(self.body["prompt_examples"])

        for example in self.body["prompt_examples"]:
            self.assertIn("prompt", example)
            self.assertIn("full_response", example)
            self.assertIn("steps", example)
            self.assertTrue(example["steps"])
            for step in example["steps"]:
                self.assertIn(step["module"], MODULE_NAMES)

    def test_description_states_the_constraints(self):
        text = self.body["description"].lower()
        self.assertIn("cannot", text)
        self.assertIn("unverified", text)


class TestArchitecture(unittest.TestCase):
    def test_returns_a_png(self):
        status, body = api.model_architecture()
        self.assertEqual(status, 200, "run scripts/build_architecture_png.py")
        self.assertIsInstance(body, bytes)
        self.assertTrue(body.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(body), 5000)


class TestIndex(unittest.TestCase):
    def test_serves_the_gui_without_auth(self):
        status, html = api.index_html()
        self.assertEqual(status, 200)
        self.assertIn("<textarea", html)
        self.assertIn("/api/execute", html)
        for word in ("login", "signin", "password"):
            self.assertNotIn(word, html.lower())


if __name__ == "__main__":
    unittest.main()
