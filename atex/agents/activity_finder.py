"""ActivityLogisticsFinder: a bounded ReAct agent over the curated catalogue.

Thought -> Action -> Observation, capped at `react_max_iters`. Two details
matter for robustness:

* Observations are windowed. Only the most recent few are replayed, so a long
  loop cannot inflate the prompt.
* The loop always yields a selection. If the model never calls `finish`, we
  salvage every place it looked at and pick the best-ranked ones ourselves,
  rather than returning nothing after paying for the calls.
"""

from __future__ import annotations

from typing import Any

from .. import ACTIVITY_LOGISTICS_FINDER
from ..context import AgentContext
from ..prompts import FINDER_SYSTEM, finder_user_prompt
from ..repository import Place
from ..state import Candidate, RunState
from ..tools import ToolError, build_toolset

OBSERVATION_WINDOW = 3
DINING_TERMS = {
    "cafe",
    "café",
    "cuisine",
    "dining",
    "food",
    "meal",
    "restaurant",
}


def _wants_restaurants(state: RunState) -> bool:
    profile = state.profile or {}
    interests = " ".join(str(value) for value in (profile.get("interests") or []))
    haystack = f"{state.request} {interests}".casefold()
    return any(term in haystack for term in DINING_TERMS)


def _register(state: RunState, place: Place) -> None:
    state.add_candidate(
        Candidate(
            place_id=place.id,
            name=place.name,
            kind=place.kind,
            brief=place.to_brief(),
        )
    )


def _ids_seen(observations: list[dict[str, Any]]) -> list[str]:
    """Every place id the agent has actually looked at, in first-seen order."""
    seen: list[str] = []
    for obs in observations:
        result = obs.get("result") or {}
        rows = result.get("results") if isinstance(result, dict) else None
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("id") and row["id"] not in seen:
                    seen.append(row["id"])
        elif isinstance(result, dict) and result.get("id") and result["id"] not in seen:
            seen.append(result["id"])
    return seen


def _select(ctx: AgentContext, state: RunState, ids: list[str], hotel_id: str | None) -> int:
    added = 0
    for place_id in ids:
        place = ctx.repo.get_place(place_id)
        if place is None:
            continue
        if place.kind == "restaurant" and not _wants_restaurants(state):
            continue
        _register(state, place)
        added += 1
    if hotel_id:
        hotel = ctx.repo.get_place(hotel_id)
        if hotel is not None and hotel.kind == "hotel":
            _register(state, hotel)
            state.selected_hotel_id = hotel.id
    return added


def _fallback_select(ctx: AgentContext, state: RunState, observations) -> int:
    """Salvage a selection when the model never called finish."""
    profile = state.profile or {}
    wanted = max(2, (profile.get("trip_days") or 3) * (profile.get("max_activities_per_day") or 2))
    ids = _ids_seen(observations)

    activities, hotels = [], []
    for place_id in ids:
        place = ctx.repo.get_place(place_id)
        if place is None:
            continue
        if place.kind == "hotel":
            hotels.append(place.id)
        elif place.kind != "restaurant" or _wants_restaurants(state):
            activities.append(place.id)

    hotel_id = hotels[0] if (hotels and profile.get("needs_hotel")) else None
    return _select(ctx, state, activities[: wanted + 2], hotel_id)


def run(ctx: AgentContext, state: RunState, instruction: str = "") -> None:
    profile = state.profile or {}
    budget = ctx.settings.budget
    tools = build_toolset(ctx.repo, state.required_needs, budget.max_candidates_per_search)

    observations: list[dict[str, Any]] = []
    finished = False
    state.finder_rounds += 1

    for iteration in range(budget.react_max_iters):
        turns_left = budget.react_max_iters - iteration
        if ctx.trace.soft_exhausted():
            break

        data = ctx.llm.complete_json(
            ACTIVITY_LOGISTICS_FINDER,
            FINDER_SYSTEM,
            finder_user_prompt(
                profile, instruction, observations[-OBSERVATION_WINDOW:], turns_left
            ),
            max_tokens=600,
        )

        action = data.get("action") or {}
        tool_name = str(action.get("tool") or "").strip()
        args = action.get("args") if isinstance(action.get("args"), dict) else {}

        if tool_name == "finish":
            activity_ids = [str(i) for i in (args.get("selected_activity_ids") or [])]
            restaurant_ids = [str(i) for i in (args.get("selected_restaurant_ids") or [])]
            hotel_id = args.get("selected_hotel_id")
            added = _select(
                ctx, state, activity_ids + restaurant_ids, str(hotel_id) if hotel_id else None
            )
            state.log(f"ActivityLogisticsFinder: selected {added} places in {iteration + 1} turns")
            finished = True
            break

        tool = tools.get(tool_name)
        if tool is None:
            observations.append(
                {"tool": tool_name or "(missing)", "error": "unknown tool", "args": args}
            )
            continue

        try:
            result: dict[str, Any] = tool(args)
        except ToolError as exc:
            result = {"error": str(exc)}
        observations.append({"tool": tool_name, "args": args, "result": result})

    if not finished:
        added = _fallback_select(ctx, state, observations)
        reason = "budget stopped the loop" if ctx.trace.soft_exhausted() else "no finish call"
        state.log(
            f"ActivityLogisticsFinder: {reason}; auto-selected {added} places from observations"
        )
        ctx.trace.note(f"ActivityLogisticsFinder fell back to auto-selection ({reason})")
