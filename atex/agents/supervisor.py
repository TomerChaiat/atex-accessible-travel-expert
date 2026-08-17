"""Supervisor: the autonomous router.

The supervisor is a real LLM decision every turn -- it is not a hard-coded
pipeline, and it may revisit modules, ask for more candidates, or stop early.
What the code enforces around it are *legality* constraints, not a route: a
decision that would break an invariant (planning before validating, looping the
finder forever) is corrected and the correction is fed back into the trace.
"""

from __future__ import annotations

from typing import Any

from .. import (
    ACCESSIBILITY_VALIDATOR,
    ACTIVITY_LOGISTICS_FINDER,
    SCHEDULE_PLANNER,
    SUPERVISOR,
    USER_PROFILE_AGENT,
)
from ..context import AgentContext
from ..prompts import SUPERVISOR_SYSTEM, supervisor_user_prompt
from ..state import RunState

ASK_USER = "ASK_USER"
FINISH = "FINISH"

CHOICES = {
    USER_PROFILE_AGENT,
    ACTIVITY_LOGISTICS_FINDER,
    ACCESSIBILITY_VALIDATOR,
    SCHEDULE_PLANNER,
    ASK_USER,
    FINISH,
}

MAX_FINDER_ROUNDS = 2


class Decision:
    def __init__(
        self,
        next_module: str,
        instruction: str = "",
        reasoning: str = "",
        clarification_question: str | None = None,
        corrected_from: str | None = None,
    ):
        self.next_module = next_module
        self.instruction = instruction
        self.reasoning = reasoning
        self.clarification_question = clarification_question
        self.corrected_from = corrected_from

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Decision({self.next_module!r}, corrected_from={self.corrected_from!r})"


def _legalize(state: RunState, choice: str) -> tuple[str, str | None]:
    """Return (allowed_choice, corrected_from).

    Guards invariants the prompt already states, because a prompt is a request
    and this is a guarantee.
    """
    if (
        state.profile is not None
        and not state.candidates
        and state.finder_rounds >= MAX_FINDER_ROUNDS
    ):
        # Live discovery genuinely found nothing (or its provider failed).
        # Let the planner produce the explicit empty-result response instead
        # of cycling until the supervisor turn limit.
        return SCHEDULE_PLANNER, choice if choice != SCHEDULE_PLANNER else None

    if choice == SCHEDULE_PLANNER:
        if state.profile is None:
            return USER_PROFILE_AGENT, choice
        if not state.candidates:
            return ACTIVITY_LOGISTICS_FINDER, choice
        if state.unvalidated():
            return ACCESSIBILITY_VALIDATOR, choice

    if choice == ACCESSIBILITY_VALIDATOR:
        if state.profile is None:
            return USER_PROFILE_AGENT, choice
        if not state.candidates:
            return ACTIVITY_LOGISTICS_FINDER, choice
        if not state.unvalidated():
            return SCHEDULE_PLANNER, choice

    if choice == ACTIVITY_LOGISTICS_FINDER:
        if state.profile is None:
            return USER_PROFILE_AGENT, choice
        if state.finder_rounds >= MAX_FINDER_ROUNDS:
            # Rule 3 in the prompt; enforced here so a stubborn model cannot
            # burn the whole budget re-searching.
            if state.unvalidated():
                return ACCESSIBILITY_VALIDATOR, choice
            return SCHEDULE_PLANNER, choice

    if choice == FINISH and state.itinerary is None:
        if state.profile is None:
            return USER_PROFILE_AGENT, choice
        if not state.candidates:
            return ACTIVITY_LOGISTICS_FINDER, choice
        if state.unvalidated():
            return ACCESSIBILITY_VALIDATOR, choice
        return SCHEDULE_PLANNER, choice

    if choice == USER_PROFILE_AGENT and state.profile is not None:
        # Re-profiling the same request wastes a call; move forward instead.
        if not state.candidates:
            return ACTIVITY_LOGISTICS_FINDER, choice

    return choice, None


def decide(ctx: AgentContext, state: RunState) -> Decision:
    # Once every selected candidate has a verdict, the only legal productive
    # next step is planning. Skip a redundant model call that can only be
    # corrected back to SchedulePlanner.
    if (
        state.itinerary is None
        and state.profile is not None
        and state.candidates
        and not state.unvalidated()
    ):
        state.log("Supervisor: -> SchedulePlanner (validation complete)")
        return Decision(
            next_module=SCHEDULE_PLANNER,
            instruction="Build the itinerary from the validated candidates.",
            reasoning="Validation is complete.",
        )

    raw: dict[str, Any] = ctx.llm.complete_json(
        SUPERVISOR,
        SUPERVISOR_SYSTEM,
        supervisor_user_prompt(state.summary_for_supervisor(ctx.trace)),
        max_tokens=300,
    )
    ctx.trace.supervisor_turns += 1

    choice = str(raw.get("next_module") or "").strip()
    if choice not in CHOICES:
        choice = FINISH if state.itinerary else USER_PROFILE_AGENT

    allowed, corrected_from = _legalize(state, choice)
    if corrected_from:
        ctx.trace.note(
            f"Supervisor chose {corrected_from} but {allowed} was required first; corrected."
        )
        state.log(f"Supervisor: {corrected_from} -> {allowed} (invariant)")
    else:
        state.log(f"Supervisor: -> {allowed}")

    question = raw.get("clarification_question")
    return Decision(
        next_module=allowed,
        instruction=str(raw.get("instruction") or "")[:400],
        reasoning=str(raw.get("reasoning") or "")[:300],
        clarification_question=str(question) if question else None,
        corrected_from=corrected_from,
    )
