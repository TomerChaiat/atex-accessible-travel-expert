"""ActivityLogisticsFinder: a bounded ReAct agent over a place provider.

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
from ..repository import Place, RepositoryError
from ..state import Candidate, RunState
from ..tools import ToolError, build_toolset

OBSERVATION_WINDOW = 3

# A longer ReAct budget is only worth having if a dead end can end early.
# Two searches in a row that return nothing mean the provider has nothing for
# this destination, and a third will not change that.
MAX_EMPTY_SEARCHES = 2
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


def _finish_selection(
    args: dict[str, Any], observations: list[dict[str, Any]]
) -> tuple[list[str], list[str], str | None]:
    """Accept only exact IDs that came from a provider observation."""
    observed_ids = set(_ids_seen(observations))
    activity_ids = [
        str(value)
        for value in (args.get("selected_activity_ids") or [])
        if str(value) in observed_ids
    ]
    restaurant_ids = [
        str(value)
        for value in (args.get("selected_restaurant_ids") or [])
        if str(value) in observed_ids
    ]
    raw_hotel_id = args.get("selected_hotel_id")
    hotel_id = (
        str(raw_hotel_id)
        if raw_hotel_id and str(raw_hotel_id) in observed_ids
        else None
    )
    return activity_ids, restaurant_ids, hotel_id


def _select(ctx: AgentContext, state: RunState, ids: list[str], hotel_id: str | None) -> int:
    added = 0
    for place_id in ids:
        try:
            place = ctx.repo.get_place(place_id)
        except RepositoryError as exc:
            state.log(f"ActivityLogisticsFinder: skipped unavailable place {place_id}: {exc}")
            continue
        if place is None:
            continue
        if place.kind == "restaurant" and not _wants_restaurants(state):
            continue
        _register(state, place)
        added += 1
    if hotel_id:
        try:
            hotel = ctx.repo.get_place(hotel_id)
        except RepositoryError as exc:
            state.log(f"ActivityLogisticsFinder: skipped unavailable hotel {hotel_id}: {exc}")
            hotel = None
        if hotel is not None and hotel.kind == "hotel":
            _register(state, hotel)
            state.selected_hotel_id = hotel.id
    return added


def activities_needed(state: RunState) -> int:
    """How many activities this trip actually requires.

    A two-week trip at three stops a day needs forty-odd places. Leaving the
    model to infer that from `trip_days` produced itineraries with one short
    activity on most days, so the number is computed and stated outright.
    """
    profile = state.profile or {}
    days = max(1, int(profile.get("trip_days") or 3))
    per_day = max(1, int(profile.get("max_activities_per_day") or 2))
    return days * per_day


def _fallback_select(ctx: AgentContext, state: RunState, observations) -> int:
    """Salvage a selection when the model never called finish."""
    wanted = max(2, activities_needed(state))
    ids = _ids_seen(observations)
    profile = state.profile or {}

    activities, hotels = [], []
    for place_id in ids:
        try:
            place = ctx.repo.get_place(place_id)
        except RepositoryError as exc:
            state.log(f"ActivityLogisticsFinder: skipped unavailable place {place_id}: {exc}")
            continue
        if place is None:
            continue
        if place.kind == "hotel":
            hotels.append(place.id)
        elif place.kind != "restaurant" or _wants_restaurants(state):
            activities.append(place.id)

    hotel_id = hotels[0] if (hotels and profile.get("needs_hotel")) else None
    return _select(ctx, state, activities[: wanted + 4], hotel_id)


def run(ctx: AgentContext, state: RunState, instruction: str = "") -> None:
    profile = state.profile or {}
    budget = ctx.settings.budget
    previously_checked = set(state.candidates)
    tools = build_toolset(
        ctx.repo,
        state.required_needs,
        budget.max_candidates_per_search,
        exclude_place_ids=previously_checked,
    )

    observations: list[dict[str, Any]] = []
    finished = False
    empty_searches = 0
    state.finder_rounds += 1
    target = activities_needed(state)

    for iteration in range(budget.react_max_iters):
        turns_left = budget.react_max_iters - iteration
        if ctx.trace.soft_exhausted():
            break

        data = ctx.llm.complete_json(
            ACTIVITY_LOGISTICS_FINDER,
            FINDER_SYSTEM,
            finder_user_prompt(
                profile,
                instruction,
                observations[-OBSERVATION_WINDOW:],
                turns_left,
                activities_needed=target,
                already_selected=len(previously_checked),
            ),
            max_tokens=600,
        )

        action = data.get("action") or {}
        tool_name = str(action.get("tool") or "").strip()
        args = action.get("args") if isinstance(action.get("args"), dict) else {}

        if tool_name == "finish":
            activity_ids, restaurant_ids, hotel_id = _finish_selection(args, observations)
            added = _select(
                ctx, state, activity_ids + restaurant_ids, hotel_id
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

        if tool_name.startswith("search_"):
            if result.get("results"):
                empty_searches = 0
            else:
                empty_searches += 1
                if empty_searches >= MAX_EMPTY_SEARCHES:
                    state.log(
                        f"ActivityLogisticsFinder: {empty_searches} empty searches; "
                        "stopping instead of spending the rest of the loop"
                    )
                    break

    if not finished:
        added = _fallback_select(ctx, state, observations)
        reason = "budget stopped the loop" if ctx.trace.soft_exhausted() else "no finish call"
        state.log(
            f"ActivityLogisticsFinder: {reason}; auto-selected {added} places from observations"
        )
        ctx.trace.note(f"ActivityLogisticsFinder fell back to auto-selection ({reason})")
