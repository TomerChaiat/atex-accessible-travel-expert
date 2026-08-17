"""Place-search tools available to ActivityLogisticsFinder.

The functions share one interface across the live Google Places provider and
the bundled offline fallback. Provider responses are reduced to compact briefs
before they enter the ReAct prompt.

Tool results are returned in `brief` form so observations stay small when they
are fed back into the ReAct loop.
"""

from __future__ import annotations

from typing import Any, Callable

from .repository import Place, Repository, RepositoryError, travel_estimate

MAX_LIMIT = 8

TRAVEL_MODES = ("wheelchair_walk", "accessible_transit", "accessible_taxi")


class ToolError(RuntimeError):
    """A tool was called with arguments it cannot satisfy."""


def _clean_limit(args: dict[str, Any], default: int, cap: int) -> int:
    try:
        value = int(args.get("limit", default))
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, cap))


def _search(repo: Repository, kind: str, args: dict[str, Any], needs: list[str], cap: int):
    city = (args.get("city") or "").strip()
    if not city:
        raise ToolError("city is required")
    categories = args.get("categories") or []
    if not isinstance(categories, list):
        categories = [str(categories)]
    try:
        places = repo.search_places(
            city=city,
            kind=kind,
            categories=[str(c) for c in categories],
            needs=needs,
            limit=_clean_limit(args, 6, cap),
        )
    except RepositoryError as exc:
        raise ToolError(str(exc)) from exc
    if not places:
        if getattr(repo, "name", "") == "google_places":
            note = (
                f"The live place provider returned no {kind} results for '{city}'. "
                "Try a broader category once, then finish without inventing places."
            )
        else:
            note = (
                f"No {kind} entries for '{city}' in the offline sample. "
                f"Sample cities: {', '.join(repo.list_cities()) or 'none'}."
            )
        return {
            "results": [],
            "note": note,
        }
    return {"results": [p.to_brief() for p in places]}


def build_toolset(
    repo: Repository, needs: list[str], max_candidates: int
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Bind the place provider and traveller's needs into ready-to-call tools."""

    cap = max(1, min(max_candidates, MAX_LIMIT))

    def search_activities(args: dict[str, Any]) -> dict[str, Any]:
        return _search(repo, "activity", args, needs, cap)

    def search_hotels(args: dict[str, Any]) -> dict[str, Any]:
        return _search(repo, "hotel", args, needs, min(cap, 4))

    def search_restaurants(args: dict[str, Any]) -> dict[str, Any]:
        return _search(repo, "restaurant", args, needs, min(cap, 4))

    def get_place_details(args: dict[str, Any]) -> dict[str, Any]:
        place_id = (args.get("place_id") or "").strip()
        try:
            place = repo.get_place(place_id)
        except RepositoryError as exc:
            raise ToolError(str(exc)) from exc
        if place is None:
            raise ToolError(f"unknown place_id '{place_id}'")
        detail = place.to_brief()
        detail["notes"] = place.notes
        detail["source"] = place.accessibility_source
        detail["crowd_level"] = place.crowd_level
        detail["noise_level"] = place.noise_level
        return detail

    def estimate_travel(args: dict[str, Any]) -> dict[str, Any]:
        try:
            origin = repo.get_place((args.get("from_place_id") or "").strip())
            destination = repo.get_place((args.get("to_place_id") or "").strip())
        except RepositoryError as exc:
            raise ToolError(str(exc)) from exc
        if origin is None or destination is None:
            raise ToolError("both from_place_id and to_place_id must be known places")
        mode = args.get("mode") or "accessible_transit"
        if mode not in TRAVEL_MODES:
            mode = "accessible_transit"
        return travel_estimate(origin, destination, mode)

    return {
        "search_activities": search_activities,
        "search_hotels": search_hotels,
        "search_restaurants": search_restaurants,
        "get_place_details": get_place_details,
        "estimate_travel": estimate_travel,
    }


def travel_matrix(places: list[Place], mode: str = "accessible_transit") -> list[dict[str, Any]]:
    """Pairwise estimates for the planner.

    Capped at a modest number of places because this is quadratic and lands in
    a prompt; beyond that the planner gets geography by area label instead.
    """
    subset = places[:8]
    matrix = []
    for i, a in enumerate(subset):
        for b in subset[i + 1 :]:
            estimate = travel_estimate(a, b, mode)
            matrix.append(
                {
                    "from": estimate["from"],
                    "to": estimate["to"],
                    "min": estimate["duration_min"],
                    "km": estimate["distance_km"],
                }
            )
    return matrix
