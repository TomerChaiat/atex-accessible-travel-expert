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


def _candidate_memory(observations: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Compact every observed place so exact IDs survive the 3-turn window."""
    remembered: dict[str, dict[str, str]] = {}
    for observation in observations:
        result = observation.get("result") or {}
        rows = result.get("results") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            rows = [result] if isinstance(result, dict) and result.get("id") else []
        searched_city = str((observation.get("args") or {}).get("city") or "")
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            place_id = str(row["id"])
            remembered.setdefault(
                place_id,
                {
                    "id": place_id,
                    "name": str(row.get("name") or ""),
                    "kind": str(row.get("kind") or "activity"),
                    "city": str(row.get("city") or searched_city),
                },
            )
    return list(remembered.values())[:80]


def _finish_selection(
    args: dict[str, Any], observations: list[dict[str, Any]]
) -> tuple[list[str], list[str], list[dict[str, str]]]:
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
    hotels: list[dict[str, str]] = []
    raw_hotels = args.get("selected_hotels")
    if isinstance(raw_hotels, list):
        for value in raw_hotels:
            if not isinstance(value, dict):
                continue
            place_id = str(value.get("place_id") or "")
            if place_id in observed_ids and place_id not in {
                hotel["place_id"] for hotel in hotels
            }:
                hotels.append(
                    {
                        "place_id": place_id,
                        "location": str(value.get("location") or "").strip(),
                    }
                )
    # Accept the old single-hotel shape from saved prompts or older models.
    raw_hotel_id = args.get("selected_hotel_id")
    if raw_hotel_id and str(raw_hotel_id) in observed_ids and not hotels:
        hotels.append({"place_id": str(raw_hotel_id), "location": ""})
    return activity_ids, restaurant_ids, hotels


def _record_hotel_stay(
    state: RunState, place: Place, requested_location: str = ""
) -> None:
    if place.id in state.selected_hotel_ids:
        return
    location = requested_location.strip() or str(place.city or "").strip()
    planned_stays = state.shape.get("hotel_stays") or []

    def usable_selected(selected: dict[str, Any]) -> bool:
        candidate = state.candidates.get(str(selected.get("place_id") or ""))
        return candidate is not None and candidate.verdict != "flagged"

    matching = next(
        (
            stay
            for stay in planned_stays
            if str(stay.get("location") or "").casefold() == location.casefold()
            and not any(
                selected.get("place_id")
                and usable_selected(selected)
                and str(selected.get("location") or "").casefold() == location.casefold()
                for selected in state.selected_hotel_stays
            )
        ),
        None,
    )
    if matching is None:
        matching = next(
            (
                stay
                for stay in planned_stays
                if not any(
                    selected.get("start_day") == stay.get("start_day")
                    and selected.get("end_day") == stay.get("end_day")
                    and usable_selected(selected)
                    for selected in state.selected_hotel_stays
                )
            ),
            {},
        )
    state.selected_hotel_stays.append(
        {
            "place_id": place.id,
            "location": str(matching.get("location") or location or place.city or ""),
            "start_day": matching.get("start_day", 1),
            "end_day": matching.get(
                "end_day", max(1, int((state.profile or {}).get("trip_days") or 1))
            ),
        }
    )
    if state.selected_hotel_id is None:
        state.selected_hotel_id = place.id


def _select(
    ctx: AgentContext,
    state: RunState,
    ids: list[str],
    hotels: list[dict[str, str]],
) -> int:
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
    for hotel_selection in hotels:
        hotel_id = hotel_selection.get("place_id")
        if not hotel_id:
            continue
        try:
            hotel = ctx.repo.get_place(hotel_id)
        except RepositoryError as exc:
            state.log(f"ActivityLogisticsFinder: skipped unavailable hotel {hotel_id}: {exc}")
            hotel = None
        if hotel is not None and hotel.kind == "hotel":
            _register(state, hotel)
            _record_hotel_stay(state, hotel, hotel_selection.get("location") or "")
    return added


def activities_needed(state: RunState) -> int:
    """How many activities this trip actually requires.

    Trip length times the Supervisor's chosen stops-per-day. Leaving the model
    to infer this produced itineraries with one short activity on most days, so
    the number is computed and stated outright.
    """
    return state.activity_target


def _fallback_select(ctx: AgentContext, state: RunState, observations) -> int:
    """Salvage a selection when the model never called finish."""
    wanted = max(2, activities_needed(state))
    ids = _ids_seen(observations)
    profile = state.profile or {}

    activities_by_location: dict[str, list[str]] = {}
    hotels: list[dict[str, str]] = []
    for place_id in ids:
        try:
            place = ctx.repo.get_place(place_id)
        except RepositoryError as exc:
            state.log(f"ActivityLogisticsFinder: skipped unavailable place {place_id}: {exc}")
            continue
        if place is None:
            continue
        if place.kind == "hotel":
            hotels.append({"place_id": place.id, "location": str(place.city or "")})
        elif place.kind != "restaurant" or _wants_restaurants(state):
            location = str(place.city or state.destination).strip().casefold()
            activities_by_location.setdefault(location, []).append(place.id)

    needed_hotels = len(state.shape.get("hotel_stays") or [])
    selected_hotels = hotels[:needed_hotels] if profile.get("needs_hotel") else []
    selected_activities: list[str] = []
    for location, target in state.activity_targets_by_location.items():
        selected_activities.extend(
            activities_by_location.get(location.casefold(), [])[: target + 2]
        )
    if not selected_activities:
        selected_activities = [
            place_id
            for values in activities_by_location.values()
            for place_id in values
        ][: wanted + 4]
    return _select(ctx, state, selected_activities, selected_hotels)


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
                state.shape,
                state.selected_hotel_stays,
                instruction,
                observations[-OBSERVATION_WINDOW:],
                _candidate_memory(observations),
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
            activity_ids, restaurant_ids, hotels = _finish_selection(args, observations)
            added = _select(
                ctx, state, activity_ids + restaurant_ids, hotels
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
