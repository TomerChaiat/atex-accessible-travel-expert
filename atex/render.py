"""Deterministic rendering of the final user-facing response.

Rendering is code, not a model call. The itinerary is already structured, so
asking an LLM to restate it would add a call, add latency, and add a chance of
contradicting the verdicts we just enforced.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .state import RunState
from .tracing import RunTrace

MARK = {
    "supported": "[verified accessible]",
    "flagged": "[accessibility concerns]",
    "unknown": "[NOT VERIFIED]",
    "n/a": "",
}

LEGEND = (
    "**How to read the labels**\n"
    "- `[verified accessible]` - the knowledge base directly confirms your stated needs.\n"
    "- `[accessibility concerns]` - a need is unmet, conditional, seasonal, or needs arranging ahead.\n"
    "- `[NOT VERIFIED]` - ATEX found no reliable information. This is not a claim that the place is "
    "inaccessible, and it is not a claim that it is accessible. Please confirm it directly."
)


def _demo_data_warning(settings: Settings) -> str | None:
    if settings.vector_backend == "memory" or settings.repository_backend == "local":
        return (
            "> **Demo data notice.** This run used the bundled offline catalogue and the synthetic "
            "demo knowledge base, not live sources. Treat every accessibility statement here as "
            "illustrative only."
        )
    return None


def _render_hotel(state: RunState) -> str | None:
    """The hotel is a separate section, not an itinerary row.

    A traveller who asked for a verified accessible hotel needs to see its
    verdict prominently, not buried inside day one.
    """
    profile = state.profile or {}
    hotel = state.candidates.get(state.selected_hotel_id or "")
    if hotel is not None and hotel.verdict == "flagged":
        # Flagged hotels are explained under Considered but not scheduled;
        # presenting one as the selected stay would contradict that decision.
        hotel = None

    if hotel is None:
        if profile.get("needs_hotel"):
            return (
                "## Where you'll stay\n"
                "- No hotel without known accessibility concerns could be selected. "
                "Ask me again with your dates and budget, or tell me an area you prefer."
            )
        return None

    detail = hotel.verdict_detail or {}
    lines = [
        "## Where you'll stay",
        f"- **{hotel.name}** {MARK.get(hotel.verdict or 'unknown', '')}".rstrip(),
    ]
    if detail.get("summary"):
        lines.append(f"  - {detail['summary']}")
    for condition in (detail.get("conditions") or [])[:2]:
        lines.append(f"  - Condition: {condition}")
    for concern in (detail.get("concerns") or [])[:2]:
        lines.append(f"  - Note: {concern}")
    if hotel.verdict != "supported":
        lines.append(
            "  - Confirm the specific adapted room, doorway widths and bathroom layout "
            "directly with the property before booking."
        )
    return "\n".join(lines)


def render_clarification(state: RunState) -> str:
    question = state.clarification_question or "Could you tell me a little more about your trip?"
    lines = ["I need one more detail before I can plan this trip.", "", f"**{question}**"]
    profile = state.profile or {}
    known = [
        f"- Destination: {profile.get('destination')}" if profile.get("destination") else None,
        f"- Trip length: {profile.get('trip_days')} days" if profile.get("trip_days") else None,
        f"- Accessibility needs: {', '.join(profile.get('accessibility_needs') or [])}"
        if profile.get("accessibility_needs")
        else None,
    ]
    known = [k for k in known if k]
    if known:
        lines += ["", "What I have so far:", *known]
    return "\n".join(lines)


def render_itinerary(state: RunState, trace: RunTrace, settings: Settings) -> str:
    itinerary: dict[str, Any] = state.itinerary or {}
    profile = state.profile or {}
    destination = profile.get("destination") or "your destination"
    days = itinerary.get("days") or []

    title = f"# Accessible itinerary: {destination}"
    if profile.get("trip_days"):
        title = f"# Accessible {profile['trip_days']}-day itinerary: {destination}"

    parts: list[str] = [title]

    warning = _demo_data_warning(settings)
    if warning:
        parts += ["", warning]

    if itinerary.get("summary"):
        parts += ["", str(itinerary["summary"])]

    hotel_block = _render_hotel(state)
    if hotel_block:
        parts += ["", hotel_block]

    for day in days:
        header = f"\n## Day {day.get('day', '?')}"
        if day.get("theme"):
            header += f" - {day['theme']}"
        parts.append(header)

        for item in day.get("items") or []:
            time = item.get("time") or ""
            name = item.get("name") or item.get("place_id") or "Activity"
            duration = item.get("duration_min")
            mark = MARK.get(str(item.get("accessibility")), "")
            line = f"- **{time}** {name}".rstrip()
            if duration:
                line += f" ({duration} min)"
            if mark:
                line += f" {mark}"
            parts.append(line)
            travel = item.get("travel_from_previous")
            if isinstance(travel, dict) and travel.get("km") is not None:
                distance = travel.get("km")
                try:
                    distance_text = f"{float(distance):g} km"
                except (TypeError, ValueError):
                    distance_text = f"{distance} km"
                parts.append(f"  - {distance_text}")
            if item.get("note"):
                parts.append(f"  - {item['note']}")

        if day.get("day_note"):
            parts.append(f"\n_{day['day_note']}_")

    confirm = itinerary.get("things_to_confirm") or []
    if confirm:
        parts += ["\n## Confirm before you travel"]
        parts += [f"- {c}" for c in confirm]

    not_scheduled = itinerary.get("not_scheduled") or []
    if not_scheduled:
        parts += ["\n## Considered but not scheduled"]
        for entry in not_scheduled:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("place_id") or "A place"
                parts.append(f"- **{name}** - {entry.get('reason', 'no reason given')}")
            else:
                parts.append(f"- {entry}")

    warnings = itinerary.get("warnings") or []
    if warnings:
        parts += ["\n## Notes"]
        parts += [f"- {w}" for w in warnings]

    # Report unique real venues that actually appear in the plan, not unused
    # alternatives the finder considered and the validator checked.
    scheduled_ids = {
        str(item.get("place_id") or "")
        for day in days
        for item in (day.get("items") or [])
        if isinstance(item, dict)
        and str(item.get("place_id") or "") in state.candidates
    }
    selected_hotel = state.candidates.get(str(state.selected_hotel_id or ""))
    if selected_hotel is not None and selected_hotel.verdict != "flagged":
        scheduled_ids.add(selected_hotel.place_id)
    scheduled = [state.candidates[place_id] for place_id in scheduled_ids]
    counts = {
        verdict: sum(1 for candidate in scheduled if candidate.verdict == verdict)
        for verdict in ("supported", "flagged", "unknown")
    }
    parts += [
        "\n---",
        LEGEND,
        "",
        f"_Checked {sum(counts.values())} places: {counts['supported']} verified, "
        f"{counts['flagged']} with concerns, {counts['unknown']} unverified. "
        f"Planned in {trace.llm_calls} model calls over {trace.elapsed_s():.1f}s._",
    ]

    if trace.stop_reason and "finished" not in trace.stop_reason:
        parts.append(
            f"_Planning stopped early: {trace.stop_reason}. The itinerary above is the best "
            "complete plan within budget._"
        )

    return "\n".join(parts)


def render_response(state: RunState, trace: RunTrace, settings: Settings) -> str:
    if state.clarification_question and state.itinerary is None:
        return render_clarification(state)
    if state.itinerary is None:
        return (
            "I could not produce an itinerary for this request. "
            "Try naming a destination city and how many days you have."
        )
    return render_itinerary(state, trace, settings)
