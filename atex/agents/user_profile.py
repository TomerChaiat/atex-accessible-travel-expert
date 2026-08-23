"""UserProfileAgent: free text in, structured travel profile out.

The LLM extracts; the normalisation below is deterministic. Anything that can
be derived by a rule (a wheelchair implies step-free entry and an accessible
toilet) is derived by a rule, so the model is never asked to be consistent
about something we can simply guarantee.
"""

from __future__ import annotations

import re
from typing import Any

from .. import USER_PROFILE_AGENT
from ..context import AgentContext
from ..prompts import USER_PROFILE_SYSTEM, user_profile_user_prompt
from ..state import MAX_ACTIVITIES_PER_DAY, RunState

VALID_NEEDS = {
    "step_free_entrance",
    "accessible_toilet",
    "lift_access",
    "wheelchair_rental",
    "accessible_parking",
    "quiet_space",
    "audio_guide_captioned",
    "tactile_or_braille",
    "assistance_animals",
}

VALID_PACE = {"relaxed", "moderate", "packed"}

VALID_TRANSPORT = {"wheelchair_walk", "accessible_transit", "accessible_taxi"}

# Someone asking for two weeks must get two weeks. This exists only to stop a
# misparsed "365 days" turning into an unplannable run; the real cost control
# is the Budget, not a truncated trip.
MAX_TRIP_DAYS = 21

# Nights away from home need somewhere to sleep. Below this, a day trip is
# plausible and accommodation should not be assumed.
HOTEL_ASSUMED_FROM_DAYS = 2

STRICT_LOCATION_PATTERNS = (
    r"\b(?:only|exclusively|entirely)\s+in\b",
    r"\bonly\s+(?:visit|explore|stay|remain)\b",
    r"\bstay\s+(?:only\s+)?within\b",
    r"\bdo\s+not\s+leave\b",
    r"\bno\s+other\s+(?:cities|locations|places)\b",
    r"\bno\s+(?:nearby\s+)?day\s*trips?\b",
)


def _strict_location_requested(request: str) -> bool:
    return any(re.search(pattern, request, re.I) for pattern in STRICT_LOCATION_PATTERNS)


# "hotels" is at least as common as "hotel" in these requests, and every
# pattern used to end at `hotel\b` -- so "I want 2 different hotels, one for
# the first week and another for the second" matched nothing at all.
_STAY_NOUN = r"(?:hotels?|accommodations?|places? to stay)"

REPLACE_HOTEL_PATTERNS = (
    rf"\b(?:\d+\s+|a\s+|two\s+)?(?:different|another|other|new|separate|alternative)\s+{_STAY_NOUN}\b",
    rf"\b(?:change|replace|swap|split)\s+(?:the\s+|my\s+|our\s+)?{_STAY_NOUN}\b",
    rf"\b{_STAY_NOUN}\b[^.?!]{{0,40}}\b(?:instead|elsewhere)\b",
    rf"\b(?:don'?t|do not|not)\s+(?:like|want)\s+(?:the\s+|this\s+)?{_STAY_NOUN}\b",
    # "one hotel for the first week and another for the second"
    rf"\b{_STAY_NOUN}\b[^.?!]{{0,60}}\b(?:first|second|each)\s+week\b",
)


def _replacement_hotel_requested(request: str) -> bool:
    """True when a follow-up asks for somewhere else to stay.

    Replanning alone cannot honour this: the itinerary is rebuilt from the same
    candidates, so the same hotel wins again. The selection has to be released
    before discovery runs, and that has to be certain rather than inferred by
    a model mid-conversation.
    """
    return any(re.search(pattern, request, re.I) for pattern in REPLACE_HOTEL_PATTERNS)


def _release_hotel_selection(state: RunState) -> None:
    """Drop the chosen hotel so the finder has to go and find another.

    The rejected hotel stays in `candidates`, which is what keeps the next
    search from simply offering it again -- discovery excludes every place
    already checked.
    """
    released = state.selected_hotel_ids
    if not released:
        return
    state.selected_hotel_id = None
    state.selected_hotel_stays.clear()
    # A hotel gap now exists, so the Supervisor routes back to discovery.
    state.finder_rounds = 0
    # Let the Supervisor re-decide the accommodation split too. "Change the
    # hotels and put me in a different one each week" is a change of shape,
    # not just of which building, and the saved shape cannot express it.
    state.plan_shape = None
    names = ", ".join(
        state.candidates[place_id].name
        for place_id in released
        if place_id in state.candidates
    )
    state.log(f"UserProfileAgent: traveller asked for a different hotel; released {names}")


def _locations(profile: dict[str, Any] | None) -> tuple[str, ...]:
    profile = profile or {}
    values = profile.get("destinations") or [profile.get("destination")]
    return tuple(
        str(value).strip().casefold()
        for value in values
        if str(value or "").strip()
    )


def _apply_profile_change(
    state: RunState,
    previous: dict[str, Any] | None,
    updated: dict[str, Any],
) -> None:
    """Invalidate only saved work whose assumptions the new profile changed."""
    if previous is None:
        return
    previous = previous or {}
    # An update that names no location at all has not moved the trip; it has
    # simply not mentioned it. Only a genuinely different set of places is a
    # change worth throwing away paid-for candidates and verdicts over.
    new_locations = _locations(updated)
    geography_changed = bool(new_locations) and _locations(previous) != new_locations
    trip_length_changed = previous.get("trip_days") != updated.get("trip_days")

    if geography_changed:
        state.candidates.clear()
        state.selected_hotel_id = None
        state.selected_hotel_stays.clear()
        state.plan_shape = None
        state.finder_rounds = 0
        state.validation_count = 0
        state.log("UserProfileAgent: destination changed; cleared prior trip candidates")
        return

    shape_fields = ("pace", "max_activities_per_day", "requested_locations_only")
    if trip_length_changed or any(
        previous.get(field) != updated.get(field) for field in shape_fields
    ):
        state.plan_shape = None
        state.finder_rounds = 0
    if trip_length_changed:
        state.selected_hotel_id = None
        state.selected_hotel_stays.clear()

    # Accessibility verdicts are relative to needs and companion context. A
    # verdict paid for on the previous turn is unsafe to reuse after either one
    # changes, even when the destination stays the same.
    validation_fields = ("accessibility_needs", "party_size", "mobility", "sensory")
    if any(previous.get(field) != updated.get(field) for field in validation_fields):
        for candidate in state.candidates.values():
            candidate.verdict = None
            candidate.verdict_detail = {}
        state.validation_count = 0


# Fields a follow-up may leave out because it is only changing one thing.
# "I want a different hotel" says nothing about the city, the trip length, or
# the wheelchair, and none of those stopped being true.
CARRIED_FIELDS = (
    "destination",
    "destinations",
    "requested_locations_only",
    "country",
    "trip_days",
    "party_size",
    "mobility",
    "sensory",
    "pace",
    "max_activities_per_day",
    "budget_level",
    "interests",
    "needs_hotel",
    "preferred_transport",
    # Dropping these would silently relax what the traveller has to have, and
    # would also look like a needs change and discard every paid-for verdict.
    "accessibility_needs",
)

# False and 0 are answers; these are absences.
_EMPTY = (None, "", [], {})


def _carry_forward(raw: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """Fill a follow-up extraction's gaps from the profile it is updating.

    A follow-up names only what it changes. Asked to re-extract "I want a
    different hotel", the model returns a profile with no destination at all --
    and the run then had nothing to plan, cleared the trip, and asked which
    city the traveller meant, one message after they said Los Angeles.

    The prompt asks the model to preserve prior fields, but a prompt is a
    request. Silence is not a retraction, so absent fields are refilled here.
    An explicitly stated value, including false or zero, always wins.
    """
    if not previous:
        return raw

    merged = dict(raw)
    for field in CARRIED_FIELDS:
        value = merged.get(field)
        earlier = previous.get(field)
        if isinstance(value, dict) and isinstance(earlier, dict):
            # Sub-fields go missing one at a time: a reply naming no wheelchair
            # must not turn a powered chair into "unknown".
            merged[field] = {
                **earlier,
                **{k: v for k, v in value.items() if v not in _EMPTY},
            }
        elif any(value is empty or value == empty for empty in _EMPTY):
            if earlier not in _EMPTY:
                merged[field] = earlier
    return merged


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_profile(raw: dict[str, Any]) -> dict[str, Any]:
    mobility = raw.get("mobility") or {}
    sensory = raw.get("sensory") or {}

    wheelchair = str(mobility.get("wheelchair") or "unknown").lower()
    uses_wheelchair = wheelchair in {"manual", "powered", "scooter"}

    needs = {n for n in (raw.get("accessibility_needs") or []) if n in VALID_NEEDS}
    if uses_wheelchair or bool(mobility.get("step_free_required")):
        needs |= {"step_free_entrance", "accessible_toilet"}
    if bool(sensory.get("low_noise")) or bool(sensory.get("autism_friendly")):
        needs.add("quiet_space")

    pace = raw.get("pace") if raw.get("pace") in VALID_PACE else None
    # Only what the traveller actually said. Inventing a number here made it
    # indistinguishable from a stated preference, and the Supervisor then had
    # nothing to decide. When this is None, the Supervisor chooses the shape of
    # the day from the request itself.
    per_day = _as_int(raw.get("max_activities_per_day"))
    if per_day is not None:
        per_day = max(1, min(per_day, MAX_ACTIVITIES_PER_DAY))

    trip_days = _as_int(raw.get("trip_days")) or 3
    trip_days = max(1, min(trip_days, MAX_TRIP_DAYS))

    # How far the traveller can cover under their own power decides which
    # travel modes are worth offering at all. A wheelchair already implies a
    # limit; a walker or cane has to be stated.
    walking_limited = bool(mobility.get("walking_limited")) or uses_wheelchair

    transport = str(raw.get("preferred_transport") or "").strip().lower()
    preferred_transport = transport if transport in VALID_TRANSPORT else None

    # "Two weeks in New York" does not say "I need a hotel", but it plainly
    # means it. Silence is not a no: only an explicit false -- staying with
    # family, living locally -- suppresses accommodation.
    needs_hotel = raw.get("needs_hotel")
    if not isinstance(needs_hotel, bool):
        needs_hotel = trip_days >= HOTEL_ASSUMED_FROM_DAYS

    destination = raw.get("destination")
    destination = destination.strip() if isinstance(destination, str) else None
    destinations: list[str] = []
    supplied_destinations = raw.get("destinations")
    if isinstance(supplied_destinations, list):
        for value in supplied_destinations:
            location = str(value or "").strip()[:80]
            if location and location.casefold() not in {v.casefold() for v in destinations}:
                destinations.append(location)
    if destination and destination.casefold() not in {v.casefold() for v in destinations}:
        destinations.insert(0, destination)
    destinations = destinations[:4]
    if destination is None and destinations:
        destination = destinations[0]

    return {
        "destination": destination,
        "destinations": destinations,
        # False is the normal case: the Supervisor may propose a realistic
        # nearby city or day trip. It becomes true only when the traveller says
        # to remain exclusively in the named locations.
        "requested_locations_only": bool(raw.get("requested_locations_only")),
        "country": raw.get("country"),
        "trip_days": trip_days,
        "party_size": _as_int(raw.get("party_size")) or 1,
        "mobility": {
            "wheelchair": wheelchair,
            "step_free_required": bool(mobility.get("step_free_required") or uses_wheelchair),
            "assistant_present": mobility.get("assistant_present"),
            "walking_limited": walking_limited,
        },
        "preferred_transport": preferred_transport,
        "sensory": {
            "low_noise": bool(sensory.get("low_noise")),
            "low_crowd": bool(sensory.get("low_crowd")),
            "autism_friendly": bool(sensory.get("autism_friendly")),
        },
        "pace": pace,
        "max_activities_per_day": per_day,
        "budget_level": raw.get("budget_level"),
        "interests": [str(i).lower() for i in (raw.get("interests") or [])][:8],
        "needs_hotel": needs_hotel,
        "accessibility_needs": sorted(needs),
        "missing_info": [str(m) for m in (raw.get("missing_info") or [])][:5],
        "notes": str(raw.get("notes") or "")[:300],
    }


def run(ctx: AgentContext, state: RunState, instruction: str = "") -> None:
    previous = state.profile
    raw = ctx.llm.complete_json(
        USER_PROFILE_AGENT,
        USER_PROFILE_SYSTEM,
        user_profile_user_prompt(state.request, state.profile),
        max_tokens=700,
    )
    updated = normalize_profile(_carry_forward(raw, previous))
    previous_destination = str((previous or {}).get("destination") or "").casefold()
    updated_destination = str(updated.get("destination") or "").casefold()
    if previous and updated_destination and updated_destination != previous_destination:
        # Defense in depth against a merge-style model reply such as
        # destinations=["Rome", "Haifa"] when Haifa appeared only in the saved
        # profile. Keep the new primary plus additional places actually named
        # in the current message.
        request_text = state.request.casefold()
        primary = str(updated.get("destination") or "")
        updated["destinations"] = [primary] + [
            location
            for location in (updated.get("destinations") or [])
            if location.casefold() != updated_destination
            and location.casefold() in request_text
        ]
    if _strict_location_requested(state.request):
        # This is a hard user boundary, so enforce it even if profile extraction
        # omitted the boolean.
        updated["requested_locations_only"] = True
    _apply_profile_change(state, previous, updated)
    if previous and _replacement_hotel_requested(state.request):
        _release_hotel_selection(state)
    state.profile = updated
    state.profile_needs_refresh = False
    destination = state.profile["destination"] or "no destination given"
    state.log(
        f"UserProfileAgent: profile built for {destination}, "
        f"{state.profile['trip_days']} days, needs={state.profile['accessibility_needs']}"
    )
