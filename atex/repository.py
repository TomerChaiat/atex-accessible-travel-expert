"""Curated place data: the Supabase-backed catalogue and its local mirror.

`Place.to_brief()` matters more than it looks. The ReAct agent sees only the
brief form, never the full record, which keeps the Finder's prompts small --
the single biggest lever on token cost in this system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from .config import SEED_DIR, Settings
from .httpjson import get_json
from .util import haversine_km

ACCESS_KEYS = (
    "step_free_entrance",
    "accessible_toilet",
    "lift_access",
    "wheelchair_rental",
    "accessible_parking",
    "quiet_space",
    "audio_guide_captioned",
    "tactile_or_braille",
    "assistance_animals",
)


@dataclass
class Place:
    id: str
    city: str
    name: str
    kind: str  # activity | hotel | restaurant
    lat: float = 0.0
    lon: float = 0.0
    country: str = ""
    area: str = ""
    categories: list[str] = field(default_factory=list)
    typical_duration_min: int = 90
    price_level: str = "mid"
    crowd_level: str = "unknown"
    noise_level: str = "unknown"
    accessibility_claims: dict[str, str] = field(default_factory=dict)
    accessibility_source: str = "unknown"
    source_url: str = ""
    last_verified: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Place":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def to_brief(self) -> dict[str, Any]:
        """Compact form for LLM prompts: identity + claims, nothing decorative."""
        claims = {k: v for k, v in self.accessibility_claims.items() if v != "unknown"}
        unknown = [k for k in ACCESS_KEYS if self.accessibility_claims.get(k, "unknown") == "unknown"]
        brief: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "categories": self.categories,
            "duration_min": self.typical_duration_min,
            "price": self.price_level,
            "claims": claims,
        }
        if self.area:
            brief["area"] = self.area
        if unknown:
            brief["unknown_claims"] = unknown
        return brief


class Repository(Protocol):
    name: str

    def list_cities(self) -> list[str]: ...

    def get_place(self, place_id: str) -> Place | None: ...

    def search_places(
        self,
        city: str,
        kind: str = "activity",
        categories: list[str] | None = None,
        needs: list[str] | None = None,
        limit: int = 6,
    ) -> list[Place]: ...


def _score(place: Place, categories: list[str], needs: list[str]) -> float:
    """Rank by interest overlap, then by how many required needs are claimed met.

    Places that merely lack information are not pushed to the bottom -- an
    unknown is not a negative, and surfacing it is the whole point of ATEX.
    """
    score = 0.0
    wanted = {c.lower() for c in categories}
    if wanted:
        overlap = wanted & {c.lower() for c in place.categories}
        score += 3.0 * len(overlap)
    for need in needs:
        value = place.accessibility_claims.get(need, "unknown")
        if value == "yes":
            score += 2.0
        elif value == "limited":
            score += 0.5
        elif value == "no":
            score -= 4.0
    if place.accessibility_source.startswith("osm"):
        score += 0.5
    return score


class LocalRepository:
    """Reads the JSON mirrors in data/seed/. Used whenever Supabase is unset."""

    name = "local"

    def __init__(self, seed_dir=SEED_DIR):
        self._seed_dir = seed_dir

    @property
    def _places(self) -> dict[str, Place]:
        return _load_seed(str(self._seed_dir))

    def list_cities(self) -> list[str]:
        return sorted({p.city for p in self._places.values()})

    def get_place(self, place_id: str) -> Place | None:
        return self._places.get(place_id)

    def search_places(
        self,
        city: str,
        kind: str = "activity",
        categories: list[str] | None = None,
        needs: list[str] | None = None,
        limit: int = 6,
    ) -> list[Place]:
        city_key = (city or "").strip().lower()
        matches = [
            p
            for p in self._places.values()
            if p.city.lower() == city_key and p.kind == kind
        ]
        ranked = sorted(
            matches,
            key=lambda p: _score(p, categories or [], needs or []),
            reverse=True,
        )
        return ranked[: max(1, limit)]


@lru_cache(maxsize=4)
def _load_seed(seed_dir: str) -> dict[str, Place]:
    from pathlib import Path

    places: dict[str, Place] = {}
    directory = Path(seed_dir)
    if not directory.exists():
        return places
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for raw in payload.get("places", []):
            place = Place.from_dict(raw)
            places[place.id] = place
    return places


class SupabaseRepository:
    """PostgREST client. Mirrors LocalRepository so the agents are unaware."""

    name = "supabase"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._headers = {
            "apikey": settings.supabase_service_key,
            "Authorization": f"Bearer {settings.supabase_service_key}",
        }

    def _select(self, query: str) -> list[dict[str, Any]]:
        url = f"{self._settings.supabase_url}/rest/v1/places?{query}"
        data = get_json(url, headers=self._headers, timeout=20.0)
        return data if isinstance(data, list) else []

    def list_cities(self) -> list[str]:
        rows = self._select("select=city")
        return sorted({r["city"] for r in rows if r.get("city")})

    def get_place(self, place_id: str) -> Place | None:
        rows = self._select(f"select=*&id=eq.{place_id}&limit=1")
        return Place.from_dict(rows[0]) if rows else None

    def search_places(
        self,
        city: str,
        kind: str = "activity",
        categories: list[str] | None = None,
        needs: list[str] | None = None,
        limit: int = 6,
    ) -> list[Place]:
        # Filter server-side by the cheap predicates, rank client-side so the
        # scoring rule stays identical to the local backend.
        rows = self._select(
            f"select=*&city=ilike.{city}&kind=eq.{kind}&limit={max(limit * 4, 24)}"
        )
        places = [Place.from_dict(r) for r in rows]
        ranked = sorted(
            places, key=lambda p: _score(p, categories or [], needs or []), reverse=True
        )
        return ranked[: max(1, limit)]


def build_repository(settings: Settings) -> Repository:
    if settings.repository_backend == "supabase":
        return SupabaseRepository(settings)
    return LocalRepository()


def travel_estimate(a: Place, b: Place, mode: str = "accessible_transit") -> dict[str, Any]:
    """Distance/duration estimate with no external API.

    Straight-line distance scaled by a detour factor, then divided by a mode
    speed. Deliberately approximate, and labelled as such in the output so the
    planner never presents it as a routed journey time.
    """
    km = haversine_km(a.lat, a.lon, b.lat, b.lon)
    road_km = km * 1.35  # typical street-network detour over straight line
    speeds = {
        "wheelchair_walk": 3.5,
        "accessible_transit": 14.0,
        "accessible_taxi": 20.0,
    }
    speed = speeds.get(mode, 14.0)
    minutes = (road_km / speed) * 60.0
    if mode != "wheelchair_walk":
        minutes += 8.0  # waiting/boarding overhead
    return {
        "from": a.id,
        "to": b.id,
        "mode": mode,
        "distance_km": round(road_km, 2),
        "duration_min": int(round(minutes)),
        "basis": "straight-line estimate, not a routed journey",
    }
