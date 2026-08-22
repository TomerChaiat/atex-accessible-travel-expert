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
    raw = ctx.llm.complete_json(
        USER_PROFILE_AGENT,
        USER_PROFILE_SYSTEM,
        user_profile_user_prompt(state.request, state.profile),
        max_tokens=700,
    )
    state.profile = normalize_profile(raw)
    if _strict_location_requested(state.request):
        # This is a hard user boundary, so enforce it even if profile extraction
        # omitted the boolean.
        state.profile["requested_locations_only"] = True
    destination = state.profile["destination"] or "no destination given"
    state.log(
        f"UserProfileAgent: profile built for {destination}, "
        f"{state.profile['trip_days']} days, needs={state.profile['accessibility_needs']}"
    )
