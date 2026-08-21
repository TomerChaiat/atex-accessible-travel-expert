"""UserProfileAgent: free text in, structured travel profile out.

The LLM extracts; the normalisation below is deterministic. Anything that can
be derived by a rule (a wheelchair implies step-free entry and an accessible
toilet) is derived by a rule, so the model is never asked to be consistent
about something we can simply guarantee.
"""

from __future__ import annotations

from typing import Any

from .. import USER_PROFILE_AGENT
from ..context import AgentContext
from ..prompts import USER_PROFILE_SYSTEM, user_profile_user_prompt
from ..state import RunState

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

PACE_DEFAULTS = {"relaxed": 2, "moderate": 3, "packed": 4}

VALID_TRANSPORT = {"wheelchair_walk", "accessible_transit", "accessible_taxi"}

# Someone asking for two weeks must get two weeks. This exists only to stop a
# misparsed "365 days" turning into an unplannable run; the real cost control
# is the Budget, not a truncated trip.
MAX_TRIP_DAYS = 21

# Nights away from home need somewhere to sleep. Below this, a day trip is
# plausible and accommodation should not be assumed.
HOTEL_ASSUMED_FROM_DAYS = 2


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

    pace = raw.get("pace") if raw.get("pace") in PACE_DEFAULTS else None
    per_day = _as_int(raw.get("max_activities_per_day"))
    if per_day is None:
        per_day = PACE_DEFAULTS.get(pace or "moderate", 3)
    per_day = max(1, min(per_day, 5))

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
    return {
        "destination": destination.strip() if isinstance(destination, str) else None,
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
        "pace": pace or "moderate",
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
    destination = state.profile["destination"] or "no destination given"
    state.log(
        f"UserProfileAgent: profile built for {destination}, "
        f"{state.profile['trip_days']} days, needs={state.profile['accessibility_needs']}"
    )
