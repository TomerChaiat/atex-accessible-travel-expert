"""SchedulePlanner: reviewed candidates in, day-by-day itinerary out.

The model does the arranging. The code does the enforcing: after the itinerary
comes back, every accessibility label is overwritten from the recorded verdicts.
A plan can never describe an unknown place as accessible, regardless of what the
model wrote.
"""

from __future__ import annotations

import re
from typing import Any

from .. import SCHEDULE_PLANNER
from ..context import AgentContext
from ..prompts import PLANNER_SYSTEM, planner_user_prompt
from ..repository import RepositoryError
from ..routing import MODE_LABELS, build_router, drop_unreasonable, planning_option
from ..state import RunState, same_location
from ..tools import travel_matrix
from ..util import haversine_km, truncate

RANK = {"supported": 0, "unknown": 1, "flagged": 2, None: 3}

# "gmp:ChIJWT0gUBz2wokRNcAxVUphAAs" and the bare provider IDs behind it. Long
# unbroken alphanumeric runs do not occur in the prose these fields hold.
PLACE_ID_PATTERN = re.compile(r"\b(?:gmp:)?[A-Za-z0-9_-]{22,}\b")
LEADING_SEPARATOR_PATTERN = re.compile(r"^[\s\W_]+")

# "Considered but not scheduled" is for places the traveller was denied and
# deserves a reason for. A live city search returns dozens of surplus venues;
# listing them all buries the handful that matter.
MAX_NOT_SCHEDULED = 8

# Checking out, crossing a city with luggage, and checking in again. Travel
# time between the two hotels is added on top of this.
HOTEL_MOVE_MINUTES = 45

# A hop between two stops on the same day. Los Angeles produced 165 minutes by
# taxi, which is not a journey between attractions -- it is the afternoon.
MAX_HOP_MINUTES = 75

# How far afield to look when topping a day up. Wide enough for a sprawling
# city, narrow enough that "anywhere in the metro area" is not an answer.
MAX_FILL_KM = 30
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


def _is_accommodation_change(
    item: dict[str, Any], state: RunState, candidate: Any
) -> bool:
    """True when a hotel row is a genuine move to different accommodation.

    A trip that changes hotel part-way has to show the move somewhere, so one
    hotel row per move is legitimate. It has to be a deliberate `stay` row for
    a hotel other than the one already presented under "Where you'll stay" --
    anything else is the planner duplicating that section.

    A move still goes through the normal verdict enforcement below, so a hotel
    with known accessibility concerns cannot reach the itinerary this way.
    """
    return (
        candidate.kind == "hotel"
        and _normalise_label(item.get("kind")) == "stay"
        and candidate.place_id != state.selected_hotel_id
        and (
            not state.selected_hotel_stays
            or candidate.place_id in state.selected_hotel_ids
        )
    )


def _readable(text: str, state: RunState) -> str:
    """Strip provider place IDs out of traveller-facing text.

    The planner habitually prefixes a confirmation line with the raw Google
    ID it was working from. That means nothing to a traveller, so any known ID
    becomes its venue name and any stray one is removed, along with the
    separator it was sitting behind.
    """
    cleaned = text or ""
    for place_id, candidate in state.candidates.items():
        if place_id and place_id in cleaned:
            cleaned = cleaned.replace(place_id, candidate.name)
    cleaned = PLACE_ID_PATTERN.sub("", cleaned)
    # "  - El Museo del Barrio: ..." once the ID in front of it is gone.
    cleaned = LEADING_SEPARATOR_PATTERN.sub("", cleaned)
    # A name replaced into a line that already named it reads as a stutter.
    cleaned = re.sub(r"^(.{2,60}?)\s*[-–—:]\s*\1\b", r"\1", cleaned)
    return " ".join(cleaned.split()).strip()


def _is_meal(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    return (
        _normalise_label(item.get("kind")) == "meal"
        or str(item.get("place_id") or "") == "meal-break"
    )


def _is_rest(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    return (
        _normalise_label(item.get("kind")) == "rest"
        or str(item.get("place_id") or "") == "rest-break"
    )


def _compact_breaks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove rest rows that add noise instead of useful recovery time.

    A midday meal already provides downtime. A rest immediately beside it is
    redundant, while an end-of-day rest is simply the end of the itinerary.
    At most one other explicit rest is retained in a day.
    """
    compact: list[dict[str, Any]] = []
    kept_rest = False
    activities_since_downtime = 0
    for index, item in enumerate(items):
        if not _is_rest(item):
            compact.append(item)
            if _is_meal(item):
                activities_since_downtime = 0
            elif _normalise_label(item.get("kind")) != "transfer":
                activities_since_downtime += 1
            continue

        previous = items[index - 1] if index else None
        following = items[index + 1] if index + 1 < len(items) else None
        redundant = (
            index == 0
            or index == len(items) - 1
            or _is_meal(previous)
            or _is_meal(following)
            or kept_rest
            or activities_since_downtime < 2
        )
        if redundant:
            continue
        compact.append(item)
        kept_rest = True
        activities_since_downtime = 0
    return compact


def _lead_with_something_to_do(
    state: RunState, items: list[dict[str, Any]]
) -> None:
    """Never open a day with a meal or a rest.

    A day that begins with lunch reads as though the traveller is expected to
    turn up at noon and eat. It happens when the stop that used to come first
    is removed -- for being unreachable, in the wrong city, or hours away --
    leaving the break behind at the top of the day.

    The break is moved after the first real stop rather than deleted, because
    it is still a meal the traveller wants.
    """
    first_activity = next(
        (index for index, item in enumerate(items) if _is_real_activity(item, state)),
        None,
    )
    if first_activity in (None, 0):
        return

    leading = items[:first_activity]
    if not all(_is_meal(item) or _is_rest(item) for item in leading):
        return

    items[:] = items[first_activity : first_activity + 1] + leading + items[first_activity + 1 :]
    state.log("SchedulePlanner: moved a leading break after the day's first stop")


def _parse_time(value: Any) -> int | None:
    try:
        hours, minutes = str(value).strip().split(":", 1)
        hours_int, minutes_int = int(hours), int(minutes)
    except (TypeError, ValueError):
        return None
    if not 0 <= hours_int <= 23 or not 0 <= minutes_int <= 59:
        return None
    return hours_int * 60 + minutes_int


def _duration(item: dict[str, Any]) -> int:
    # Checking into a new hotel is a real event in the day, but a short one.
    defaults = {"meal": 60, "rest": 30, "transfer": 0, "stay": 30}
    kind = _normalise_label(item.get("kind"))
    try:
        value = int(item.get("duration_min"))
    except (TypeError, ValueError):
        value = defaults.get(kind, 90)
    # The planner habitually gives accommodation a zero duration, which would
    # place the next item at the same minute. Only a transfer is genuinely
    # instantaneous.
    if value <= 0 and kind != "transfer":
        value = defaults.get(kind, 90)
    return max(0, value)


def _travel_minutes(item: dict[str, Any]) -> int:
    """Minutes the schedule allows for reaching this item, from its own row."""
    travel = item.get("travel_from_previous")
    if not isinstance(travel, dict):
        return 0
    try:
        return max(0, int(travel.get("min") or 0))
    except (TypeError, ValueError):
        return 0


def _align_item_times(items: list[dict[str, Any]], day_start: str | None = None) -> None:
    """Lay out contiguous start times that account for getting between venues.

    Each start equals the previous item's end plus the travel time shown on
    this item's own row. The traveller can therefore add the numbers up: no
    gap appears that the itinerary has not already explained.

    Travel must already be attached, so `_attach_travel_options` runs first.
    """
    if not items:
        return
    cursor = _parse_time(items[0].get("time"))
    if cursor is None:
        cursor = _parse_time(day_start)
    if cursor is None:
        cursor = 9 * 60
    for item in items:
        # Index 0 counts as well: its travel is the journey from the hotel.
        cursor += _travel_minutes(item)
        item["time"] = f"{(cursor // 60) % 24:02d}:{cursor % 60:02d}"
        cursor += _duration(item)


def _travel_lookup(matrix: list[dict[str, Any]] | None) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in matrix or []:
        if not isinstance(row, dict):
            continue
        origin = str(row.get("from") or "")
        destination = str(row.get("to") or "")
        if not origin or not destination:
            continue
        lookup[tuple(sorted((origin, destination)))] = row
    return lookup


def _matrix_options(estimate: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild a single travel option from a precomputed matrix row.

    Used when the routing provider has no coordinates to work from, so the
    itinerary still says how long the hop takes instead of going silent.
    """
    try:
        minutes = int(estimate.get("min") or 0)
    except (TypeError, ValueError):
        minutes = 0
    if minutes <= 0:
        return []
    return [
        {
            "mode": "accessible_transit",
            "label": MODE_LABELS["accessible_transit"],
            "km": estimate.get("km"),
            "minutes": minutes,
            "source": "estimate",
        }
    ]


def _attach_travel_options(
    state: RunState,
    items: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    router: Any = None,
    places: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> bool:
    """Attach per-mode travel options between consecutive real scheduled venues.

    The traveller sees every way of making the hop that suits their mobility,
    with a time for each. The schedule itself is laid out on the slowest of
    them, so the plan holds however they choose to travel.
    """
    places = places or {}
    previous_place_id: str | None = None
    previous_name: str = ""
    attached = False

    for item in items:
        place_id = str(item.get("place_id") or "")
        if place_id not in state.candidates:
            continue
        name = str(item.get("name") or state.candidates[place_id].name)

        if previous_place_id and previous_place_id != place_id:
            options: list[dict[str, Any]] = []
            origin, destination = places.get(previous_place_id), places.get(place_id)
            if router is not None and origin is not None and destination is not None:
                options = drop_unreasonable(
                    router.options(origin, destination, profile)
                )
            if not options:
                estimate = lookup.get(tuple(sorted((previous_place_id, place_id))))
                if estimate is not None:
                    options = _matrix_options(estimate)

            chosen = planning_option(options)
            if chosen is not None:
                item["travel_from_previous"] = {
                    "from_place_id": previous_place_id,
                    "from_name": previous_name,
                    "km": chosen.get("km"),
                    "min": chosen.get("minutes"),
                    "options": options,
                }
                attached = True

        previous_place_id = place_id
        previous_name = name
    return attached


def _has_location(place: Any) -> bool:
    try:
        lat, lon = float(place.lat), float(place.lon)
    except (AttributeError, TypeError, ValueError):
        return False
    # Google returns 0,0 for a record whose location field was dropped.
    return not (lat == 0.0 and lon == 0.0)


def _drop_unreachable(
    state: RunState, items: list[dict[str, Any]], places: dict[str, Any] | None
) -> list[tuple[str, str]]:
    """Remove scheduled venues whose location could not be resolved.

    A place whose details lookup failed -- an obsolete or altered Google ID --
    has no coordinates, so there is no distance to quote and no way to say how
    the traveller reaches it. A stop you cannot work out how to get to is not a
    plan, so it is dropped and explained rather than left in the day bare.

    Skipped entirely when no locations resolved at all, because a provider
    outage must empty the travel details, not the itinerary.
    """
    if not places:
        return []

    dropped: list[tuple[str, str]] = []
    kept: list[dict[str, Any]] = []
    for item in items:
        place_id = str(item.get("place_id") or "")
        candidate = state.candidates.get(place_id)
        if candidate is not None and not _has_location(places.get(place_id)):
            name = str(item.get("name") or candidate.name)
            state.log(f"SchedulePlanner: dropped unreachable {name}")
            dropped.append((place_id, name))
            continue
        kept.append(item)

    items[:] = kept
    return dropped


def _day_anchor(
    state: RunState, items: list[dict[str, Any]], places: dict[str, Any] | None
) -> Any:
    """Where the day currently is: its last real stop, or the hotel it starts from."""
    if not places:
        return None
    for item in reversed(items):
        place = places.get(str(item.get("place_id") or ""))
        if place is not None and _has_location(place):
            return place
    return None


def _clear_travel(items: list[dict[str, Any]]) -> None:
    for item in items:
        item.pop("travel_from_previous", None)


def _drop_distant_stops(
    state: RunState, items: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Remove stops the traveller would spend half a day reaching.

    A Los Angeles day held a taxi hop of 165 minutes. Nothing caught it: the
    transit cap only governs public transport, and once transit is dropped for
    being slow the schedule is laid out on the taxi instead -- so an absurd
    journey became the plan rather than a warning.
    """
    dropped: list[tuple[str, str]] = []
    kept: list[dict[str, Any]] = []
    for item in items:
        if _travel_minutes(item) > MAX_HOP_MINUTES and _is_real_activity(item, state):
            name = str(item.get("name") or item.get("place_id"))
            state.log(
                f"SchedulePlanner: dropped {name}; "
                f"{_travel_minutes(item)} min from the previous stop"
            )
            dropped.append((str(item.get("place_id") or ""), name))
            continue
        kept.append(item)
    items[:] = kept
    return dropped


def _recentre_hotels(state: RunState, places: dict[str, Any] | None) -> None:
    """Swap each stay's hotel for the one closest to what that stay will see.

    The finder picks hotels before the itinerary exists, so it cannot know
    where the traveller will actually spend their days. A fortnight in Los
    Angeles was based at the airport, which is why getting anywhere took
    hours. Choosing again once the candidates are known costs nothing and is
    the difference between a base and a commute.
    """
    if not places or not state.selected_hotel_stays:
        return

    # A traveller who asked for a different hotel each week must not be given
    # the same one twice because it happens to be the most central. Every
    # hotel another stay already holds is off limits, so recentring one stay
    # cannot quietly take the hotel out from under the next.
    taken: set[str] = set()
    assigned = {
        str(stay.get("place_id") or "")
        for stay in state.selected_hotel_stays
        if stay.get("place_id")
    }

    for stay in state.selected_hotel_stays:
        location = str(stay.get("location") or "")
        current = state.candidates.get(str(stay.get("place_id") or ""))
        reserved = taken | (assigned - {str(stay.get("place_id") or "")})
        targets = [
            places[c.place_id]
            for c in state.candidates.values()
            if c.kind == "activity"
            and c.verdict != "flagged"
            and c.place_id in places
            and _has_location(places[c.place_id])
            and (
                not location
                or not c.brief.get("city")
                or same_location(str(c.brief.get("city")), location)
            )
        ]
        options = [
            c
            for c in state.candidates.values()
            if c.kind == "hotel"
            and c.verdict != "flagged"
            and c.place_id not in reserved
            and c.place_id in places
            and _has_location(places[c.place_id])
            and (
                not location
                or not c.brief.get("city")
                or same_location(str(c.brief.get("city")), location)
            )
        ]
        if not targets or len(options) < 2:
            if current is not None:
                taken.add(current.place_id)
            continue

        def mean_km(candidate: Any) -> float:
            home = places[candidate.place_id]
            return sum(
                haversine_km(home.lat, home.lon, t.lat, t.lon) for t in targets
            ) / len(targets)

        # A verified hotel is worth a longer commute than an unverified one.
        best = min(options, key=lambda c: (RANK.get(c.verdict, 3), mean_km(c)))
        if current is not None and best.place_id == current.place_id:
            taken.add(current.place_id)
            continue
        if current is not None and RANK.get(best.verdict, 3) > RANK.get(current.verdict, 3):
            taken.add(current.place_id)
            continue
        state.log(
            f"SchedulePlanner: moved the {location or 'trip'} stay to {best.name}, "
            "closer to the planned attractions"
        )
        stay["place_id"] = best.place_id
        taken.add(best.place_id)

    if state.selected_hotel_stays:
        state.selected_hotel_id = state.selected_hotel_stays[0].get("place_id")


def _hotel_for_day(state: RunState, day_number: int) -> Any:
    """The candidate the traveller sleeps at on this day, if one was chosen."""
    for stay in state.selected_hotel_stays:
        try:
            start = int(stay.get("start_day") or 1)
            end = int(stay.get("end_day") or start)
        except (TypeError, ValueError):
            continue
        if start <= day_number <= end:
            candidate = state.candidates.get(str(stay.get("place_id") or ""))
            if candidate is not None and candidate.verdict != "flagged":
                return candidate
    return None


def _ensure_hotel_move(
    state: RunState, items: list[dict[str, Any]], day_number: int
) -> None:
    """Put the change of hotel in the day it happens.

    Moving between hotels costs the traveller real time -- checking out,
    carrying luggage across a city, checking in -- and a schedule that skips
    it quietly hands that time to an attraction instead.

    The row is logistics, not a visit, so it carries no accessibility label.
    The hotel's own verdict belongs in "Where you'll stay", which is where a
    traveller looks for it.
    """
    moving_to = next(
        (
            stay
            for stay in state.selected_hotel_stays
            if int(stay.get("start_day") or 1) == day_number
            and int(stay.get("start_day") or 1) > 1
        ),
        None,
    )
    if moving_to is None:
        return
    hotel = state.candidates.get(str(moving_to.get("place_id") or ""))
    if hotel is None or hotel.verdict == "flagged":
        return
    if any(str(item.get("place_id") or "") == hotel.place_id for item in items):
        return

    items.insert(
        0,
        {
            "time": "",
            "place_id": hotel.place_id,
            "name": f"Move to {hotel.name}",
            "kind": "stay",
            "duration_min": HOTEL_MOVE_MINUTES,
            "accessibility": "n/a",
            "note": "Check out, travel with luggage, and check in.",
        },
    )
    state.log(f"SchedulePlanner: added hotel move to {hotel.name} on day {day_number}")


def _attach_hotel_departure(
    state: RunState,
    items: list[dict[str, Any]],
    day_number: int,
    router: Any,
    places: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> None:
    """Account for getting from the hotel to the day's first stop.

    Days began at `day_start` sharp at the first attraction, as though the
    traveller woke up inside it. The journey out of the hotel is part of the
    morning and belongs in the arithmetic.
    """
    hotel = _hotel_for_day(state, day_number)
    if hotel is None or not places:
        return
    first = next((item for item in items if _is_real_activity(item, state)), None)
    if first is None or first.get("travel_from_previous"):
        return

    origin = places.get(hotel.place_id)
    destination = places.get(str(first.get("place_id") or ""))
    if origin is None or destination is None or origin.id == destination.id:
        return

    options = drop_unreasonable(router.options(origin, destination, profile))
    chosen = planning_option(options)
    if chosen is None:
        return
    first["travel_from_previous"] = {
        "from_place_id": hotel.place_id,
        "from_name": hotel.name,
        "km": chosen.get("km"),
        "min": chosen.get("minutes"),
        "options": options,
    }


def _is_real_activity(item: dict[str, Any], state: RunState) -> bool:
    candidate = state.candidates.get(str(item.get("place_id") or ""))
    return candidate is not None and candidate.kind == "activity"


def _fill_short_day(
    state: RunState,
    items: list[dict[str, Any]],
    target: int,
    planned_location: str,
    used_ids: set[str],
    places: dict[str, Any] | None,
) -> list[tuple[str, str, str]]:
    """Top a day up to its target using candidates nobody scheduled.

    The Supervisor decides how full a day should be and the prompt states the
    number, but the planner still returned two stops a day for a fourteen-day
    Los Angeles trip while forty-eight checked candidates sat unused. Asking
    the model again is not a guarantee; adding the places here is.

    Verified venues go in first, then unverified ones -- unverified is a real
    answer, not a reason to leave the day half empty. Returns the newly added
    unknown venues so they still reach "Confirm before you travel".
    """
    added_unknown: list[tuple[str, str, str]] = []
    scheduled = sum(1 for item in items if _is_real_activity(item, state))
    if scheduled >= target:
        return added_unknown

    def usable(candidate: Any) -> bool:
        if candidate.kind != "activity" or candidate.place_id in used_ids:
            return False
        if candidate.verdict == "flagged":
            return False
        if places and not _has_location(places.get(candidate.place_id)):
            # No coordinates means no way to say how the traveller gets there.
            return False
        city = str(candidate.brief.get("city") or "")
        if city and planned_location and city.casefold() != planned_location.casefold():
            return False
        return True

    # Nearest first, within reach of where the day already is. Ordering by
    # name alone put a venue nearly three hours away into a Los Angeles day
    # because it happened to sort early.
    anchor = _day_anchor(state, items, places)

    def distance_km(candidate: Any) -> float:
        if anchor is None or not places:
            return 0.0
        place = places.get(candidate.place_id)
        if place is None:
            return 0.0
        return haversine_km(anchor.lat, anchor.lon, place.lat, place.lon)

    spare = sorted(
        (
            c
            for c in state.candidates.values()
            if usable(c) and distance_km(c) <= MAX_FILL_KM
        ),
        key=lambda c: (RANK.get(c.verdict, 3), round(distance_km(c), 1), c.name),
    )

    for candidate in spare:
        if scheduled >= target:
            break
        verdict = candidate.verdict or "unknown"
        items.append(
            {
                "time": "",
                "place_id": candidate.place_id,
                "name": candidate.name,
                "kind": "activity",
                "duration_min": int(candidate.brief.get("duration_min") or 90),
                "accessibility": verdict,
                # No note. The traveller asked for this many stops a day, so a
                # filled one is what they requested rather than something to
                # apologise for, and an unverified venue already has its own
                # line under "Confirm before you travel".
                "note": "",
            }
        )
        used_ids.add(candidate.place_id)
        scheduled += 1
        if verdict == "unknown":
            added_unknown.append((candidate.place_id, candidate.name, verdict))
        state.log(f"SchedulePlanner: filled day with {candidate.name}")
    return added_unknown


def _trim_past_day_end(
    state: RunState, items: list[dict[str, Any]], day_end: str | None
) -> None:
    """Drop trailing venues that would start after the day is meant to finish.

    Topping a day up can overshoot once travel time is added. A stop beginning
    after the traveller's day has ended is worse than a shorter day.
    """
    end = _parse_time(day_end)
    if end is None:
        return
    while items:
        start = _parse_time(items[-1].get("time"))
        if start is None or start < end or not _is_real_activity(items[-1], state):
            break
        state.log(f"SchedulePlanner: dropped {items[-1].get('name')} past day end")
        items.pop()


def _prune_not_scheduled(
    state: RunState,
    entries: list[Any],
    unreachable: list[tuple[str, str]],
) -> list[Any]:
    """Keep the rejections that earned a place in the response, and cap them.

    A live city search returns dozens of surplus venues, and the planner used
    to list every one it did not need -- forty entries whose reason amounted to
    "no evidence in the knowledge base". That is not a rejection: an unverified
    place is usable, and saying otherwise buries the few venues that really do
    have a concern the traveller needs to read.

    So an unverified candidate that simply was not needed is dropped, concerns
    come first, and the tail becomes a single count.
    """
    unreachable_ids = {place_id for place_id, _ in unreachable}

    def rank(entry: Any) -> int:
        if not isinstance(entry, dict):
            return 3
        candidate = state.candidates.get(str(entry.get("place_id") or ""))
        if candidate is not None and candidate.verdict == "flagged":
            return 0  # A real accessibility concern. Always worth the space.
        if str(entry.get("place_id") or "") in unreachable_ids:
            return 1
        return 2

    def worth_listing(entry: Any) -> bool:
        if rank(entry) < 2:
            return True
        if not isinstance(entry, dict):
            return True
        candidate = state.candidates.get(str(entry.get("place_id") or ""))
        # Surplus unverified candidates are not rejections; they are simply
        # places the itinerary did not need.
        return not (candidate is not None and candidate.verdict == "unknown")

    kept = sorted((e for e in entries if worth_listing(e)), key=rank)
    for entry in kept:
        if isinstance(entry, dict) and entry.get("reason"):
            entry["reason"] = _readable(str(entry["reason"]), state)

    dropped = len(entries) - len(kept)
    overflow = max(0, len(kept) - MAX_NOT_SCHEDULED)
    kept = kept[:MAX_NOT_SCHEDULED]

    remaining = dropped + overflow
    if remaining:
        kept.append(
            f"{remaining} further place(s) were reviewed and not needed for this "
            "itinerary. None of them was rejected for an accessibility concern."
        )
    return kept


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
    conditions = [str(value).strip() for value in (detail.get("conditions") or []) if value]
    if conditions:
        parts.append("Conditions: " + "; ".join(conditions[:2]))
    unmet = [str(value).replace("_", " ") for value in (detail.get("unmet_needs") or [])]
    if unmet:
        parts.append("Unmet needs: " + ", ".join(unmet[:3]))
    return truncate(
        " ".join(parts) or "The accessibility evidence conflicts with the traveller's needs.",
        700,
    )


def _enforce_verdicts(
    state: RunState,
    itinerary: dict[str, Any],
    travel_estimates: list[dict[str, Any]] | None = None,
    router: Any = None,
    places: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scheduled_non_ok: list[tuple[str, str, str]] = []
    unreachable: list[tuple[str, str]] = []
    matrix_lookup = _travel_lookup(travel_estimates)
    profile = state.profile or {}
    shape = state.shape

    # Choose the base before laying out the days that start from it.
    _recentre_hotels(state, places)

    days = itinerary.get("days")
    if not isinstance(days, list):
        days = []

    clean_days = []
    for index, day in enumerate(days, start=1):
        if not isinstance(day, dict):
            continue
        day_number = day.get("day") or index
        day_shape = next(
            (
                value
                for value in (shape.get("days") or [])
                if value.get("day") == day_number
            ),
            {},
        )
        planned_location = str(day_shape.get("location") or "").strip()
        items = day.get("items") if isinstance(day.get("items"), list) else []
        clean_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            place_id = item.get("place_id")
            kind = str(item.get("kind") or "activity")
            candidate = state.candidates.get(str(place_id or ""))
            candidate_city = str((candidate.brief if candidate else {}).get("city") or "")
            if (
                candidate is not None
                and candidate_city
                and planned_location
                and candidate_city.casefold() != planned_location.casefold()
            ):
                state.log(
                    f"SchedulePlanner: dropped {candidate.name} from day {day_number}; "
                    f"it belongs to {candidate_city}, not {planned_location}"
                )
                continue
            if _is_generic_placeholder(item, candidate):
                # Generic downtime has no venue and therefore no accessibility
                # claim. Also detach any candidate ID the model borrowed.
                item["place_id"] = _canonical_placeholder_id(item)
                item["accessibility"] = "n/a"
            elif (
                candidate is not None
                and candidate.kind == "hotel"
                and not _is_accommodation_change(item, state, candidate)
            ):
                # The hotel is where the traveller sleeps, not a stop on the
                # tour. It has its own "Where you'll stay" section, so putting
                # it in a day both duplicates it and -- with the zero duration
                # the planner gives it -- collides with the next item's start.
                state.log(f"SchedulePlanner: dropped hotel row {candidate.name}")
                continue
            elif candidate is not None and _normalise_label(item.get("kind")) == "stay":
                # Moving hotel is a logistics step, not a visit. Its own
                # accessibility verdict belongs in "Where you'll stay", which
                # is where a traveller looks for it; labelling the move as
                # though it were an attraction only muddies both.
                if candidate.verdict == "flagged":
                    continue
                item["accessibility"] = "n/a"
                if not str(item.get("name") or "").strip():
                    item["name"] = f"Move to {candidate.name}"
            elif candidate is not None:
                verdict = candidate.verdict or "unknown"
                if verdict == "flagged":
                    # Defense in depth: even if the planner ignored its prompt,
                    # a venue with known concerns cannot reach the itinerary.
                    continue
                item["accessibility"] = verdict
                conditions = candidate.verdict_detail.get("conditions") or []
                if conditions:
                    # A supported place can still depend on a condition that
                    # the current profile satisfies, such as having a companion.
                    item["note"] = "Access condition: " + " ".join(
                        str(condition) for condition in conditions[:2]
                    )
                if verdict == "unknown":
                    # The candidate's name, never the place_id: a traveller
                    # cannot act on "gmp:ChIJWT0gUBz2wokRNcAxVUphAAs".
                    scheduled_non_ok.append(
                        (candidate.place_id, candidate.name or str(item.get("name") or ""), verdict)
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
        # Venues we cannot locate go before anything else: they must not
        # become the origin of the next hop's distance.
        unreachable.extend(_drop_unreachable(state, clean_items, places))
        clean_items = _compact_breaks(clean_items)
        day["items"] = clean_items
        day["day"] = day_number
        day["_target"] = max(1, int(day_shape.get("activities") or 1))
        day["_location"] = planned_location
        clean_days.append(day)

    # Every day is now clean, so we know which candidates the plan actually
    # used and can hand the leftovers to whichever days came up short.
    used_ids = {
        str(item.get("place_id") or "")
        for day in clean_days
        for item in day["items"]
        if str(item.get("place_id") or "") in state.candidates
    }
    for day in clean_days:
        day_number = day["day"]
        # A change of hotel comes first: it happens before the day's sightseeing
        # and it consumes time that must not be handed to an attraction.
        _ensure_hotel_move(state, day["items"], day_number)
        target = day.pop("_target")
        location = day.pop("_location")
        # Fill, route, and discard anything that turned out to be hours away;
        # a second pass lets a nearer candidate take the vacated slot.
        for _ in range(2):
            scheduled_non_ok.extend(
                _fill_short_day(
                    state, day["items"], target, location, used_ids, places
                )
            )
            _clear_travel(day["items"])
            _attach_travel_options(
                state, day["items"], matrix_lookup, router, places, profile
            )
            _attach_hotel_departure(
                state, day["items"], day_number, router, places, profile
            )
            too_far = _drop_distant_stops(state, day["items"])
            if not too_far:
                break
            unreachable.extend(too_far)
            for place_id, _name in too_far:
                used_ids.discard(place_id)
        # After every drop and refill, make sure the day still opens with
        # something to do rather than with lunch.
        _lead_with_something_to_do(state, day["items"])
        _clear_travel(day["items"])
        _attach_travel_options(
            state, day["items"], matrix_lookup, router, places, profile
        )
        _attach_hotel_departure(state, day["items"], day_number, router, places, profile)
        _align_item_times(day["items"], shape.get("day_start"))
        _trim_past_day_end(state, day["items"], shape.get("day_end"))

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
        readable
        for c in (itinerary.get("things_to_confirm") or [])
        if not any(term in str(c).casefold() for term in generic_confirmation_terms)
        if (readable := _readable(str(c), state))
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
    itinerary["things_to_confirm"] = [_readable(entry, state) for entry in confirm[:12]]

    if not isinstance(itinerary.get("warnings"), list):
        itinerary["warnings"] = []
    itinerary.setdefault("summary", "")

    # Preserve model-provided spare reasons, then deterministically add every
    # rejected candidate with the validator's concrete explanation. The model
    # never gets to hide a concern by omitting it from not_scheduled.
    not_scheduled = [
        entry
        for entry in (itinerary.get("not_scheduled") or [])
        if not (
            isinstance(entry, dict)
            and str(entry.get("place_id") or "") in state.selected_hotel_ids
            and state.candidates.get(str(entry.get("place_id") or "")) is not None
            and state.candidates[str(entry.get("place_id"))].verdict != "flagged"
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
    # A venue removed for being unreachable is still a venue the traveller was
    # offered and then did not get. Saying why beats a silent disappearance.
    for place_id, name in unreachable:
        if place_id in listed_ids:
            continue
        listed_ids.add(place_id)
        not_scheduled.append(
            {
                "place_id": place_id,
                "name": name,
                "reason": (
                    "The place provider no longer returns a location for this venue, "
                    "so there is no way to say how far away it is or how to reach it. "
                    "It was left out rather than scheduled without directions."
                ),
            }
        )

    itinerary["not_scheduled"] = _prune_not_scheduled(state, not_scheduled, unreachable)
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
    # The matrix is the cheap local estimate, and it exists only so the planner
    # can group each day geographically. The per-mode options the traveller
    # actually reads are routed afterwards, for the handful of pairs that made
    # it into the plan.
    matrix = travel_matrix(places)
    places_by_id = {place.id: place for place in places}

    # A fixed ceiling truncated long trips: a two-week plan is roughly seven
    # times the JSON of a two-day one. Scale with the trip, then cap.
    trip_days = max(1, int((state.profile or {}).get("trip_days") or 3))
    budget = ctx.settings.budget
    output_tokens = min(budget.planner_max_output_tokens, 1200 + 500 * trip_days)

    result = ctx.llm.complete_json(
        SCHEDULE_PLANNER,
        PLANNER_SYSTEM,
        planner_user_prompt(
            profile=state.profile or {},
            candidates=[c.to_planner_dict() for c in candidates],
            travel_matrix=matrix,
            instruction=instruction or "Build the itinerary now.",
            plan_shape=state.shape,
            selected_hotel_stays=state.selected_hotel_stays,
        ),
        max_tokens=output_tokens,
    )

    state.itinerary = _enforce_verdicts(
        state,
        result,
        matrix,
        router=build_router(ctx.settings),
        places=places_by_id,
    )
    day_count = len(state.itinerary.get("days", []))
    state.log(f"SchedulePlanner: produced a {day_count}-day itinerary")
