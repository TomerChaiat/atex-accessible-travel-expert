"""SchedulePlanner: validated candidates in, day-by-day itinerary out.

The model does the arranging. The code does the enforcing: after the itinerary
comes back, every accessibility label is overwritten from the recorded verdicts.
A plan can never describe an unknown place as accessible, regardless of what the
model wrote.
"""

from __future__ import annotations

from typing import Any

from .. import SCHEDULE_PLANNER
from ..context import AgentContext
from ..prompts import PLANNER_SYSTEM, planner_user_prompt
from ..repository import RepositoryError
from ..state import RunState
from ..tools import travel_matrix
from ..util import truncate

RANK = {"supported": 0, "unknown": 1, "flagged": 2, None: 3}
GENERIC_PLACEHOLDER_NAMES = {
    "break",
    "free time",
    "hotel rest",
    "lunch",
    "lunch break",
    "meal break",
    "rest",
    "rest and recharge",
    "rest break",
}


def _normalise_label(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _is_generic_placeholder(item: dict[str, Any], candidate: Any) -> bool:
    """Return true when an itinerary row does not identify a real venue."""
    kind = _normalise_label(item.get("kind"))
    name = _normalise_label(item.get("name"))
    if kind in {"rest", "transfer"}:
        return True
    if name in GENERIC_PLACEHOLDER_NAMES:
        return True
    if name.startswith(("hotel rest ", "lunch break ", "meal break ", "rest break ")):
        return True
    # A meal receives a verdict only if the row actually names a selected
    # restaurant. The model must not borrow an attraction or hotel ID.
    return kind == "meal" and (candidate is None or candidate.kind != "restaurant")


def _canonical_placeholder_id(item: dict[str, Any]) -> str:
    kind = _normalise_label(item.get("kind"))
    name = _normalise_label(item.get("name"))
    if kind == "meal" or "lunch" in name or "meal" in name:
        return "meal-break"
    if kind == "transfer":
        return "transfer"
    return "rest-break"


def _ordered_candidates(state: RunState) -> list:
    """Order candidates for planning while retaining rejected ones in state.

    Dropping unknowns would quietly hide exactly the information the traveller
    most needs to see.
    """
    return sorted(
        state.candidates.values(),
        key=lambda c: (RANK.get(c.verdict, 3), c.kind != "activity", c.name),
    )


def _flagged_reason(candidate: Any) -> str:
    """Turn the validator's structured result into a concrete user-facing reason."""
    detail = candidate.verdict_detail or {}
    parts: list[str] = []
    summary = str(detail.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    concerns = [str(value).strip() for value in (detail.get("concerns") or []) if value]
    if concerns:
        parts.append("Concerns: " + "; ".join(concerns[:2]))
    unmet = [str(value).replace("_", " ") for value in (detail.get("unmet_needs") or [])]
    if unmet:
        parts.append("Unmet needs: " + ", ".join(unmet[:3]))
    return truncate(
        " ".join(parts) or "The accessibility evidence conflicts with the traveller's needs.",
        320,
    )


def _enforce_verdicts(state: RunState, itinerary: dict[str, Any]) -> dict[str, Any]:
    scheduled_non_ok: list[tuple[str, str, str]] = []

    days = itinerary.get("days")
    if not isinstance(days, list):
        days = []

    clean_days = []
    for index, day in enumerate(days, start=1):
        if not isinstance(day, dict):
            continue
        items = day.get("items") if isinstance(day.get("items"), list) else []
        clean_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            place_id = item.get("place_id")
            kind = str(item.get("kind") or "activity")
            candidate = state.candidates.get(str(place_id or ""))
            if _is_generic_placeholder(item, candidate):
                # Generic downtime has no venue and therefore no accessibility
                # claim. Also detach any candidate ID the model borrowed.
                item["place_id"] = _canonical_placeholder_id(item)
                item["accessibility"] = "n/a"
            elif candidate is not None:
                verdict = candidate.verdict or "unknown"
                if verdict == "flagged":
                    # Defense in depth: even if the planner ignored its prompt,
                    # a venue with known concerns cannot reach the itinerary.
                    continue
                item["accessibility"] = verdict
                if verdict == "unknown":
                    scheduled_non_ok.append(
                        (candidate.place_id, str(item.get("name") or place_id), verdict)
                    )
            elif kind in ("rest", "transfer", "meal"):
                # A generic break the planner invented rather than a real venue;
                # there is nothing to verify, so labelling it is misleading.
                item["accessibility"] = "n/a"
            else:
                # An item the planner invented rather than selected: it has no
                # verdict, so it cannot be presented as checked.
                item["accessibility"] = "unknown"
            clean_items.append(item)
        day["items"] = clean_items
        day["day"] = day.get("day") or index
        clean_days.append(day)

    itinerary["days"] = clean_days

    generic_confirmation_terms = (
        "hotel rest",
        "lunch break",
        "meal break",
        "rest and recharge",
        "rest-break",
        "meal-break",
    )
    confirm = [
        str(c)
        for c in (itinerary.get("things_to_confirm") or [])
        if not any(term in str(c).casefold() for term in generic_confirmation_terms)
    ]
    existing = " ".join(confirm).lower()
    noted: set[str] = set()
    for place_id, name, verdict in scheduled_non_ok:
        # A place scheduled on two days still only needs confirming once.
        if place_id in noted:
            continue
        noted.add(place_id)
        if place_id.lower() in existing or name.lower() in existing:
            continue
        detail = state.candidates[place_id].verdict_detail if place_id in state.candidates else {}
        why = truncate(detail.get("summary", ""), 120) if detail else ""
        if verdict == "unknown":
            confirm.append(
                f"{name}: accessibility is UNVERIFIED. Contact them directly before relying on it."
                + (f" ({why})" if why else "")
            )
        else:
            confirm.append(f"{name}: accessibility concerns found. {why}".strip())
    itinerary["things_to_confirm"] = confirm[:12]

    itinerary.setdefault("warnings", [])
    itinerary.setdefault("summary", "")

    # Preserve model-provided spare reasons, then deterministically add every
    # rejected candidate with the validator's concrete explanation. The model
    # never gets to hide a concern by omitting it from not_scheduled.
    not_scheduled = [
        entry
        for entry in (itinerary.get("not_scheduled") or [])
        if not (
            isinstance(entry, dict)
            and entry.get("place_id") == state.selected_hotel_id
            and state.candidates.get(str(state.selected_hotel_id or "")) is not None
            and state.candidates[str(state.selected_hotel_id)].verdict != "flagged"
        )
    ]
    listed_ids = {
        str(entry.get("place_id") or "")
        for entry in not_scheduled
        if isinstance(entry, dict)
    }
    for candidate in state.by_verdict("flagged"):
        if candidate.place_id in listed_ids:
            # Replace a generic model reason with the validator's explanation.
            for entry in not_scheduled:
                if isinstance(entry, dict) and entry.get("place_id") == candidate.place_id:
                    entry["reason"] = _flagged_reason(candidate)
            continue
        not_scheduled.append(
            {
                "place_id": candidate.place_id,
                "name": candidate.name,
                "reason": _flagged_reason(candidate),
            }
        )
    itinerary["not_scheduled"] = not_scheduled
    return itinerary


def run(ctx: AgentContext, state: RunState, instruction: str = "") -> None:
    all_candidates = _ordered_candidates(state)
    candidates = [candidate for candidate in all_candidates if candidate.verdict != "flagged"]
    if not candidates:
        message = (
            "No candidate could be scheduled without known accessibility concerns."
            if all_candidates
            else "The live place search returned no candidate places for this request."
        )
        warning = (
            "Flagged venues were excluded rather than presented as recommendations."
            if all_candidates
            else "No matching places were returned; no venues were invented."
        )
        state.itinerary = _enforce_verdicts(state, {
            "summary": message,
            "days": [],
            "not_scheduled": [],
            "warnings": [warning],
            "things_to_confirm": [],
        })
        state.log("SchedulePlanner: nothing to schedule")
        return

    places = []
    try:
        places = [p for p in (ctx.repo.get_place(c.place_id) for c in candidates) if p]
    except RepositoryError as exc:
        # Candidate briefs remain enough to build an itinerary. A temporary
        # details outage should only remove the optional travel matrix.
        state.log(f"SchedulePlanner: place details unavailable: {exc}")
    matrix = travel_matrix(places)

    result = ctx.llm.complete_json(
        SCHEDULE_PLANNER,
        PLANNER_SYSTEM,
        planner_user_prompt(
            profile=state.profile or {},
            candidates=[c.to_planner_dict() for c in candidates],
            travel_matrix=matrix,
            instruction=instruction or "Build the itinerary now.",
        ),
        max_tokens=2000,
    )

    state.itinerary = _enforce_verdicts(state, result)
    day_count = len(state.itinerary.get("days", []))
    state.log(f"SchedulePlanner: produced a {day_count}-day itinerary")
