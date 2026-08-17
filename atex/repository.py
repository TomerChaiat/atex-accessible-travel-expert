"""Real-world place discovery plus a deterministic offline sample.

`Place.to_brief()` matters more than it looks. The ReAct agent sees only the
brief form, never the full record, which keeps the Finder's prompts small --
the single biggest lever on token cost in this system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol
from urllib.parse import quote

from .config import SEED_DIR, Settings
from .httpjson import HttpError, get_json, post_json
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

GOOGLE_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_PLACE_FIELDS = (
    "id,displayName,formattedAddress,location,types,primaryType,priceLevel,"
    "rating,accessibilityOptions,googleMapsUri,businessStatus"
)


class RepositoryError(RuntimeError):
    """A live place provider could not complete a repository operation."""


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
    """Reads data/seed for keyless local development and deterministic tests."""

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


class GooglePlacesRepository:
    """Global runtime place discovery backed by Google Places API (New)."""

    name = "google_places"

    def __init__(self, settings: Settings):
        self._api_key = settings.google_maps_api_key
        self._cache: dict[str, Place] = {}
        self._headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": f"places.{GOOGLE_PLACE_FIELDS.replace(',', ',places.')}",
        }

    def list_cities(self) -> list[str]:
        # The provider is global; there is no finite city allow-list.
        return []

    @staticmethod
    def _claim(options: dict[str, Any], key: str) -> str:
        if key not in options:
            return "unknown"
        return "yes" if options.get(key) is True else "no"

    @staticmethod
    def _price_level(value: str) -> str:
        return {
            "PRICE_LEVEL_FREE": "free",
            "PRICE_LEVEL_INEXPENSIVE": "low",
            "PRICE_LEVEL_MODERATE": "mid",
            "PRICE_LEVEL_EXPENSIVE": "high",
            "PRICE_LEVEL_VERY_EXPENSIVE": "high",
        }.get(value or "", "mid")

    def _to_place(self, raw: dict[str, Any], city: str, kind: str) -> Place | None:
        google_id = str(raw.get("id") or "").strip()
        display = raw.get("displayName") or {}
        name = str(display.get("text") if isinstance(display, dict) else display or "").strip()
        if not google_id or not name:
            return None

        location = raw.get("location") or {}
        options = raw.get("accessibilityOptions") or {}
        claims = {
            "step_free_entrance": self._claim(
                options, "wheelchairAccessibleEntrance"
            ),
            "accessible_toilet": self._claim(
                options, "wheelchairAccessibleRestroom"
            ),
            "accessible_parking": self._claim(
                options, "wheelchairAccessibleParking"
            ),
        }
        place = Place(
            id=f"gmp:{google_id}",
            city=city,
            name=name,
            kind=kind,
            lat=float(location.get("latitude") or 0.0),
            lon=float(location.get("longitude") or 0.0),
            area=str(raw.get("formattedAddress") or ""),
            categories=[str(value) for value in (raw.get("types") or [])][:12],
            typical_duration_min={"hotel": 0, "restaurant": 75}.get(kind, 90),
            price_level=self._price_level(str(raw.get("priceLevel") or "")),
            accessibility_claims=claims,
            accessibility_source="google-places:accessibilityOptions",
            source_url=str(raw.get("googleMapsUri") or ""),
            notes=(
                f"Google Places listing. Rating: {raw.get('rating')}. "
                "Accessibility options are preliminary and require ATEX evidence validation."
            ),
        )
        self._cache[place.id] = place
        return place

    @staticmethod
    def _text_query(city: str, kind: str, categories: list[str]) -> str:
        if kind == "hotel":
            subject = "hotels"
        elif kind == "restaurant":
            subject = "restaurants"
        else:
            interests = " ".join(c for c in categories if c).strip()
            subject = f"{interests} tourist attractions" if interests else "tourist attractions"
        return f"{subject} in {city}"

    def search_places(
        self,
        city: str,
        kind: str = "activity",
        categories: list[str] | None = None,
        needs: list[str] | None = None,
        limit: int = 6,
    ) -> list[Place]:
        city = (city or "").strip()
        if not city:
            return []
        page_size = max(1, min(max(limit * 2, 8), 20))
        try:
            payload = post_json(
                GOOGLE_TEXT_SEARCH_URL,
                {
                    "textQuery": self._text_query(city, kind, categories or []),
                    "pageSize": page_size,
                    "languageCode": "en",
                },
                headers=self._headers,
                timeout=20.0,
                max_retries=1,
            )
        except (HttpError, OSError, TimeoutError) as exc:
            raise RepositoryError(f"Google Places search failed: {exc}") from exc

        places = [
            place
            for raw in (payload.get("places") or [])
            if isinstance(raw, dict)
            and raw.get("businessStatus", "OPERATIONAL") != "CLOSED_PERMANENTLY"
            if (place := self._to_place(raw, city, kind)) is not None
        ]
        ranked = sorted(
            places,
            key=lambda place: _score(place, categories or [], needs or []),
            reverse=True,
        )
        return ranked[: max(1, limit)]

    def get_place(self, place_id: str) -> Place | None:
        cached = self._cache.get(place_id)
        if cached is not None:
            return cached
        if not place_id.startswith("gmp:"):
            return None

        google_id = quote(place_id[4:], safe="")
        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": GOOGLE_PLACE_FIELDS,
        }
        try:
            raw = get_json(
                f"https://places.googleapis.com/v1/places/{google_id}",
                headers=headers,
                timeout=20.0,
            )
        except (HttpError, OSError, TimeoutError) as exc:
            raise RepositoryError(f"Google Place details failed: {exc}") from exc
        if not isinstance(raw, dict):
            return None
        # Follow-up sessions may fetch details without retaining the original
        # search city/kind. The saved Candidate brief remains authoritative for
        # planning; these defaults only restore coordinates and identity.
        return self._to_place(raw, "", "activity")


def build_repository(settings: Settings) -> Repository:
    if settings.repository_backend == "google_places":
        return GooglePlacesRepository(settings)
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
