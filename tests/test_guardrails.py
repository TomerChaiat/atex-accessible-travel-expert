"""Budget guardrails, termination, and the forced-finalize path.

These are the tests that protect the $13 budget and the 300s Vercel limit, so
they assert behaviour under deliberately starved budgets rather than happy paths.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.config import Budget, settings  # noqa: E402
from atex.graph import run_agent  # noqa: E402
from atex.tracing import RunTrace  # noqa: E402

PROMPT = (
    "We are a family of four visiting Amsterdam for three days. Our daughter uses a manual "
    "wheelchair. Relaxed pace, no more than two activities per day. We need an accessible hotel."
)


def with_budget(**overrides):
    base = settings()
    return replace(base, budget=replace(base.budget, **overrides))


class TestBudgetsAreRespected(unittest.TestCase):
    def test_default_run_stays_well_inside_every_limit(self):
        cfg = settings()
        result = run_agent(PROMPT, settings=cfg)
        usage = result.usage

        self.assertLessEqual(usage["llm_calls"], cfg.budget.max_total_llm_calls)
        self.assertLessEqual(usage["supervisor_turns"], cfg.budget.max_supervisor_turns)
        self.assertLessEqual(usage["total_tokens"], cfg.budget.max_tokens_per_run)
        self.assertLess(usage["elapsed_seconds"], cfg.budget.wall_clock_budget_s)

    def test_supervisor_turn_cap_is_enforced(self):
        cfg = with_budget(max_supervisor_turns=2)
        result = run_agent(PROMPT, settings=cfg)
        self.assertLessEqual(result.usage["supervisor_turns"], 2)

    def test_llm_call_cap_is_enforced(self):
        cfg = with_budget(max_total_llm_calls=6)
        result = run_agent(PROMPT, settings=cfg)
        self.assertLessEqual(result.usage["llm_calls"], 6)


class TestAlwaysReturnsSomething(unittest.TestCase):
    def test_forced_finalize_still_produces_an_itinerary(self):
        """Starve the loop: it must finalize from the reserve, not fail."""
        cfg = with_budget(max_supervisor_turns=3, max_total_llm_calls=9)
        result = run_agent(PROMPT, settings=cfg)

        self.assertIsNone(result.error)
        self.assertTrue(result.response.strip())
        self.assertIsNotNone(result.state.itinerary)

    def test_severely_starved_run_still_returns_valid_shape(self):
        cfg = with_budget(max_supervisor_turns=1, max_total_llm_calls=4, max_tokens_per_run=3000)
        result = run_agent(PROMPT, settings=cfg)

        self.assertIsNone(result.error)
        self.assertIsInstance(result.response, str)
        self.assertTrue(result.response.strip())
        self.assertIsInstance(result.steps, list)

    def test_unchecked_places_are_reported_as_unknown_not_dropped(self):
        cfg = with_budget(max_supervisor_turns=3, max_total_llm_calls=9)
        result = run_agent(PROMPT, settings=cfg)
        for candidate in result.state.candidates.values():
            self.assertIsNotNone(
                candidate.verdict, f"{candidate.place_id} left without a verdict"
            )


class TestTermination(unittest.TestCase):
    def test_no_livelock_when_validation_cap_is_below_candidate_count(self):
        """A cap lower than the candidate count must not spin the supervisor."""
        cfg = with_budget(max_validations_per_run=2)
        result = run_agent(PROMPT, settings=cfg)

        self.assertLess(
            result.usage["supervisor_turns"],
            cfg.budget.max_supervisor_turns,
            "supervisor burned every turn - the validator likely livelocked",
        )
        self.assertFalse(result.state.unvalidated())

    def test_missing_destination_asks_rather_than_inventing_one(self):
        result = run_agent("I would like to go somewhere nice for a few days.")
        self.assertIsNone(result.error)
        self.assertIsNone(result.state.destination or None)
        self.assertIn("?", result.response)


class TestTraceAccounting(unittest.TestCase):
    def test_every_llm_call_is_recorded_exactly_once(self):
        result = run_agent(PROMPT)
        self.assertEqual(len(result.steps), result.usage["llm_calls"])

    def test_tokens_are_counted_even_without_provider_usage(self):
        result = run_agent(PROMPT)
        self.assertGreater(result.usage["prompt_tokens"], 0)
        self.assertGreater(result.usage["completion_tokens"], 0)

    def test_soft_limit_leaves_reserve_for_finalize(self):
        budget = Budget()
        trace = RunTrace(budget=budget)
        trace.llm_calls = budget.max_total_llm_calls - budget.reserve_llm_calls

        self.assertIsNotNone(trace.soft_exhausted())
        trace.check_hard_limit()  # reserve is still available


class TestValidationBatching(unittest.TestCase):
    def test_batching_reduces_validator_calls(self):
        one = run_agent(PROMPT, settings=with_budget(validation_batch_size=1))
        three = run_agent(PROMPT, settings=with_budget(validation_batch_size=3))

        def validator_calls(result):
            return sum(1 for s in result.steps if s["module"] == "AccessibilityValidator")

        self.assertLess(validator_calls(three), validator_calls(one))


if __name__ == "__main__":
    unittest.main()
