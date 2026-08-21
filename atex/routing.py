"""Travel options between two scheduled venues.

Two providers behind one interface:

* `LocalRouter` derives distance and duration from coordinates alone -- the
  straight-line estimate this system has always used, now expressed per travel
  mode instead of silently assuming public transport.
* `GoogleRoutesRouter` asks the Google Routes API for real routed durations and
  falls back to the local estimate for any pair or mode it cannot answer.

Only the consecutive pairs that survive into the final itinerary are routed, so
the live provider is billed for a handful of elements per run rather than a
quadratic matrix over every candidate the finder considered.

Which modes are offered is a product decision, not a routing one: suggesting a
5 km push to a manual wheelchair user is worse than useless, so
`offered_modes()` filters by what the traveller's profile can actually do.
"""

from __future__ import annotations

from typing import Any

from .httpjson import HttpError, post_json
from .util import haversine_km

COMPUTE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

MODES = ("wheelchair_walk", "accessible_transit", "accessible_taxi")

MODE_LABELS = {
    "wheelchair_walk": "wheelchair or on foot",
    "accessible_transit": "accessible public transport",
    "accessible_taxi": "wheelchair-accessible taxi",
}

GOOGLE_TRAVEL_MODE = {
    "wheelchair_walk": "WALK",
    "accessible_transit": "TRANSIT",
    "accessible_taxi": "DRIVE",
}

MODE_SPEED_KMH = {
    "wheelchair_walk": 3.5,
    "accessible_transit": 14.0,
    "accessible_taxi": 20.0,
}

# Waiting, boarding, hailing. A self-powered trip has none of it.
MODE_OVERHEAD_MIN = {
    "wheelchair_walk": 0.0,
    "accessible_transit": 8.0,
    "accessible_taxi": 5.0,
}

DETOUR_FACTOR = 1.35  # typical street-network detour over a straight line

# Google's WALK duration assumes an able-bodied pedestrian at roughly 4.8 km/h.
# A manual wheelchair, a walker, or a cane is slower, and quoting Google's
# number unadjusted would understate every journey for exactly the travellers
# this system exists to serve.
WHEELCHAIR_WALK_FACTOR = 1.37

# How far the traveller can reasonably cover under their own power. Beyond
# this, `wheelchair_walk` is not offered at all.
SELF_POWERED_LIMIT_KM = {
    "powered": 3.0,
    "scooter": 3.0,
    "manual": 1.5,
    "walking_limited": 0.8,
    "default": 2.0,
}

# Below this, a tram or a taxi is theatre; walking is the honest answer.
TRANSIT_MIN_KM = 0.4

# Ceiling on live routing per run, so a long trip cannot turn into a surprise
# Google bill or blow the wall-clock budget.
MAX_GOOGLE_ELEMENTS = 40
GOOGLE_TIMEOUT_S = 8.0


def _round_up_5(minutes: float) -> int:
    """Itinerary times read better in five-minute steps, and rounding up never
    promises the traveller time they do not have."""
    return max(5, int(5 * -(-minutes // 5)))


def self_powered_limit_km(profile: dict[str, Any] | None) -> float:
    mobility = (profile or {}).get("mobility") or {}
    wheelchair = str(mobility.get("wheelchair") or "unknown").lower()
    if wheelchair in ("powered", "scooter"):
        return SELF_POWERED_LIMIT_KM["powered"]
    if wheelchair == "manual":
        return SELF_POWERED_LIMIT_KM["manual"]
    if mobility.get("walking_limited"):
        return SELF_POWERED_LIMIT_KM["walking_limited"]
    return SELF_POWERED_LIMIT_KM["default"]


def _is_self_powered_slow(profile: dict[str, Any] | None) -> bool:
    mobility = (profile or {}).get("mobility") or {}
    wheelchair = str(mobility.get("wheelchair") or "unknown").lower()
    return wheelchair in ("manual", "powered", "scooter") or bool(
        mobility.get("walking_limited")
    )


def preferred_mode(profile: dict[str, Any] | None) -> str | None:
    """The traveller's stated way of getting around, if they named one."""
    value = str((profile or {}).get("preferred_transport") or "").strip().lower()
    return value if value in MODES else None


def offered_modes(profile: dict[str, Any] | None, km: float) -> list[str]:
    """Modes worth showing for a journey of this length to this traveller.

    Always returns at least one mode: an accessible taxi covers any distance.
    """
    stated = preferred_mode(profile)
    if stated:
        return [stated]

    modes = []
    if km <= self_powered_limit_km(profile):
        modes.append("wheelchair_walk")
    if km >= TRANSIT_MIN_KM:
        modes.append("accessible_transit")
    modes.append("accessible_taxi")
    return modes


def planning_option(options: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The option the schedule's arithmetic is built on.

    The slowest offered option, so the itinerary still holds together whichever
    one the traveller actually picks.
    """
    if not options:
        return None
    return max(options, key=lambda option: option.get("minutes") or 0)


class LocalRouter:
    """Coordinate-only estimates. No network, no key, deterministic."""

    name = "estimate"

    def options(self, origin, destination, profile=None) -> list[dict[str, Any]]:
        km = round(
            haversine_km(origin.lat, origin.lon, destination.lat, destination.lon)
            * DETOUR_FACTOR,
            2,
        )
        return [self._estimate(mode, km) for mode in offered_modes(profile, km)]

    @staticmethod
    def _estimate(mode: str, km: float) -> dict[str, Any]:
        minutes = (km / MODE_SPEED_KMH[mode]) * 60.0 + MODE_OVERHEAD_MIN[mode]
        return {
            "mode": mode,
            "label": MODE_LABELS[mode],
            "km": km,
            "minutes": _round_up_5(minutes),
            "source": "estimate",
        }


class GoogleRoutesRouter(LocalRouter):
    """Real routed durations, with the local estimate as the safety net.

    Every failure mode -- no key, disabled API, no transit in this city, a
    quota error, a timeout -- degrades to the estimate for that one pair rather
    than losing the itinerary.
    """

    name = "google_routes"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}
        self._elements = 0

    def options(self, origin, destination, profile=None) -> list[dict[str, Any]]:
        estimates = super().options(origin, destination, profile)
        slow_on_foot = _is_self_powered_slow(profile)

        live_by_mode = {
            option["mode"]: self._route(origin, destination, option["mode"])
            for option in estimates
        }
        # A mode Google could not answer must not be estimated from a shorter
        # straight line than the modes it could. Mixing the two produced a
        # 1.65 km hop whose walk came out faster than its taxi.
        basis_km = next(
            (live["km"] for live in live_by_mode.values() if live and live["km"]),
            None,
        )

        merged = []
        for option in estimates:
            live = live_by_mode.get(option["mode"])
            if live is None:
                merged.append(
                    self._estimate(option["mode"], basis_km)
                    if basis_km is not None
                    else option
                )
                continue
            minutes = live["minutes"]
            if option["mode"] == "wheelchair_walk" and slow_on_foot:
                minutes *= WHEELCHAIR_WALK_FACTOR
            merged.append(
                {
                    **option,
                    "km": live["km"],
                    "minutes": _round_up_5(minutes),
                    "source": "google",
                }
            )
        return merged

    def _route(self, origin, destination, mode: str) -> dict[str, Any] | None:
        key = (origin.id, destination.id, mode)
        if key in self._cache:
            return self._cache[key]
        if self._elements >= MAX_GOOGLE_ELEMENTS:
            return None
        self._elements += 1

        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
        }
        body = {
            "origin": _waypoint(origin),
            "destination": _waypoint(destination),
            "travelMode": GOOGLE_TRAVEL_MODE[mode],
            "units": "METRIC",
        }
        try:
            payload = post_json(
                COMPUTE_ROUTES_URL,
                body,
                headers=headers,
                timeout=GOOGLE_TIMEOUT_S,
                max_retries=0,
            )
        except (HttpError, OSError, TimeoutError, ValueError):
            # Routing is an enhancement. Losing it must never lose the trip.
            self._cache[key] = None
            return None

        parsed = _parse_route(payload)
        self._cache[key] = parsed
        return parsed


def _waypoint(place) -> dict[str, Any]:
    return {
        "location": {
            "latLng": {"latitude": float(place.lat), "longitude": float(place.lon)}
        }
    }


def _parse_route(payload: Any) -> dict[str, Any] | None:
    routes = (payload or {}).get("routes") if isinstance(payload, dict) else None
    if not isinstance(routes, list) or not routes:
        return None
    route = routes[0]
    if not isinstance(route, dict):
        return None

    raw_duration = str(route.get("duration") or "").strip()
    if not raw_duration.endswith("s"):
        return None
    try:
        seconds = float(raw_duration[:-1])
        meters = float(route.get("distanceMeters") or 0.0)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return {"minutes": seconds / 60.0, "km": round(meters / 1000.0, 2)}


def build_router(settings) -> LocalRouter:
    """Google Routes when a Maps key is configured, the estimator otherwise."""
    if getattr(settings, "google_maps_api_key", ""):
        return GoogleRoutesRouter(settings.google_maps_api_key)
    return LocalRouter()


def describe_options(options: list[dict[str, Any]]) -> str:
    """One traveller-facing sentence listing every offered mode and its time."""
    parts = [
        f"{option['label']} ~{option['minutes']} min"
        for option in options
        if option.get("minutes")
    ]
    return "; ".join(parts)
