"""The orchestrator: an LLM-driven supervisor loop with hard guardrails.

Shape of a run:

    Supervisor -> module -> Supervisor -> module -> ... -> FINISH
                                                        |
                          soft budget trips ------------+--> forced finalize

The supervisor genuinely chooses the path. What the loop guarantees is that the
run terminates, stays inside the Vercel time limit, and always ends with a real
response -- `forced finalize` runs SchedulePlanner from the reserve headroom
when the budget runs out mid-plan.

Nodes are `(ctx, state, instruction) -> None`, which is LangGraph's node shape.
The loop below is hand-rolled rather than a StateGraph because the budget
checks, the invariant corrections, and the finalize path all need to inspect
state between every hop; swapping in LangGraph later means wiring these same
functions as nodes and is not a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agents import (
    accessibility_validator,
    activity_finder,
    schedule_planner,
    supervisor,
    user_profile,
)
from .config import Settings, settings as load_settings
from .context import AgentContext
from .llm import ModuleOutputError
from .render import render_response
from .state import Candidate, RunState
from .tracing import BudgetExceeded, RunTrace

MODULES: dict[str, Callable[[AgentContext, RunState, str], None]] = {
    "UserProfileAgent": user_profile.run,
    "ActivityLogisticsFinder": activity_finder.run,
    "AccessibilityValidator": accessibility_validator.run,
    "SchedulePlanner": schedule_planner.run,
}

MAX_MODULE_FAILURES = 2


@dataclass
class RunResult:
    response: str
    steps: list[dict[str, Any]]
    state: RunState
    usage: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def session_state(self) -> dict[str, Any]:
        """The slice worth persisting for a follow-up turn."""
        return {
            "profile": self.state.profile,
            "plan_shape": self.state.plan_shape,
            "selected_hotel_id": self.state.selected_hotel_id,
            "selected_hotel_stays": self.state.selected_hotel_stays,
            "candidates": [
                {
                    "place_id": c.place_id,
                    "name": c.name,
                    "kind": c.kind,
                    "brief": c.brief,
                    "verdict": c.verdict,
                    "verdict_detail": c.verdict_detail,
                }
                for c in self.state.candidates.values()
            ],
            "itinerary": self.state.itinerary,
            "history": self.state.history[-8:],
        }


def restore_state(request: str, session_id: str | None, saved: dict[str, Any] | None) -> RunState:
    state = RunState(request=request, session_id=session_id)
    if not saved:
        return state

    state.profile = saved.get("profile")
    state.plan_shape = saved.get("plan_shape")
    state.selected_hotel_id = saved.get("selected_hotel_id")
    state.selected_hotel_stays = list(saved.get("selected_hotel_stays") or [])
    if not state.selected_hotel_stays and state.selected_hotel_id:
        # Backward compatibility with sessions created before multi-city stays.
        state.selected_hotel_stays = [
            {
                "place_id": state.selected_hotel_id,
                "location": (state.profile or {}).get("destination") or "",
                "start_day": 1,
                "end_day": max(1, int((state.profile or {}).get("trip_days") or 1)),
            }
        ]
    state.history = list(saved.get("history") or [])
    state.turn_index = int(saved.get("turn_index") or 0) + 1

    for raw in saved.get("candidates") or []:
        candidate = Candidate(
            place_id=raw["place_id"],
            name=raw.get("name", raw["place_id"]),
            kind=raw.get("kind", "activity"),
            brief=raw.get("brief") or {},
            verdict=raw.get("verdict"),
            verdict_detail=raw.get("verdict_detail") or {},
        )
        state.candidates[candidate.place_id] = candidate

    # The previous itinerary is deliberately not restored: a follow-up turn asks
    # for a change, so it must be replanned. Verdicts *are* kept, because they
    # cost LLM calls and do not change between turns.
    return state


def _forced_finalize(ctx: AgentContext, state: RunState) -> None:
    """Produce an itinerary from the reserve when the loop ran out of budget."""
    if state.itinerary is not None or state.clarification_question:
        return
    ctx.trace.finalizing = True
    ctx.trace.note("Forced finalize: producing an itinerary from reserve budget")

    # Anything still unchecked is honestly unknown rather than silently omitted.
    for candidate in state.unvalidated():
        candidate.verdict = "unknown"
        candidate.verdict_detail = {
            "verdict": "unknown",
            "summary": "Not checked: the planning budget ran out before validation.",
            "evidence_ids": [],
            "concerns": [],
        }

    try:
        schedule_planner.run(ctx, state, "Budget is nearly exhausted. Produce the plan now.")
    except (BudgetExceeded, ModuleOutputError) as exc:
        ctx.trace.note(f"Forced finalize could not run the planner: {exc}")


def run_agent(
    request: str,
    session_id: str | None = None,
    saved_state: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> RunResult:
    cfg = settings or load_settings()
    trace = RunTrace(budget=cfg.budget)
    ctx = AgentContext.build(trace, cfg)
    state = restore_state(request, session_id, saved_state)

    error: str | None = None
    failures = 0

    try:
        while True:
            reason = trace.soft_exhausted()
            if reason:
                trace.stop_reason = reason
                break

            decision = supervisor.decide(ctx, state)

            if decision.next_module == supervisor.ASK_USER:
                state.clarification_question = (
                    decision.clarification_question
                    or "Which city would you like to visit, and for how many days?"
                )
                trace.stop_reason = "clarification requested"
                break

            if decision.next_module == supervisor.FINISH:
                trace.stop_reason = "supervisor finished"
                break

            module = MODULES.get(decision.next_module)
            if module is None:
                trace.note(f"Supervisor named an unknown module: {decision.next_module}")
                trace.stop_reason = "unknown module"
                break

            try:
                module(ctx, state, decision.instruction)
            except ModuleOutputError as exc:
                failures += 1
                trace.note(f"{decision.next_module} returned unusable output: {exc}")
                state.log(f"{decision.next_module}: failed ({failures})")
                if failures >= MAX_MODULE_FAILURES:
                    trace.stop_reason = "repeated module failures"
                    break

    except BudgetExceeded as exc:
        trace.stop_reason = f"hard budget limit: {exc}"
    except Exception as exc:  # noqa: BLE001 - the API must still return a trace
        error = f"{type(exc).__name__}: {exc}"
        trace.stop_reason = "unhandled error"

    if error is None:
        try:
            _forced_finalize(ctx, state)
        except Exception as exc:  # noqa: BLE001
            trace.note(f"Forced finalize failed: {type(exc).__name__}: {exc}")

    response = render_response(state, trace, cfg)

    return RunResult(
        response=response,
        steps=trace.steps_payload(),
        state=state,
        usage={**trace.usage(), "backends": cfg.backend_summary()},
        notes=trace.notes,
        error=error,
    )
