"""Harvest real accessibility data from OpenStreetMap into data/seed/.

OSM is the best keyless source for this project: `wheelchair=yes|limited|no`,
`toilets:wheelchair`, and `tactile_paving` are structured, crowd-maintained
tags on actual venues, which beats scraping prose from blogs.

    python scripts/harvest_osm.py Amsterdam
    python scripts/harvest_osm.py Amsterdam Barcelona Berlin --limit 60

Replaces the hand-authored demo seed file for that city. Anything OSM does not
tag is written as "unknown" rather than guessed -- which is exactly the value
the AccessibilityValidator needs in order to say "unverified" honestly.

Please respect the Overpass API's fair-use policy: this script sleeps between
cities and requests a modest number of elements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.config import SEED_DIR  # noqa: E402
from atex.util import slugify  # noqa: E402

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

QUERY = """
[out:json][timeout:120];
area["name"="{city}"]["boundary"="administrative"]->.searchArea;
(
  nwr(area.searchArea)["tourism"~"^(museum|attraction|gallery|zoo|aquarium|theme_park)$"];
  nwr(area.searchArea)["leisure"~"^(park|garden|nature_reserve)$"];
  nwr(area.searchArea)["historic"~"^(monument|memorial|castle)$"];
  nwr(area.searchArea)["tourism"="hotel"];
  nwr(area.searchArea)["amenity"~"^(restaurant|cafe)$"];
);
out center tags 400;
"""

WHEELCHAIR_MAP = {"yes": "yes", "limited": "limited", "no": "no", "designated": "yes"}

CATEGORY_TAGS = {
    "tourism": {
        "museum": ["museum", "indoor", "culture"],
        "gallery": ["museum", "art", "indoor", "culture"],
        "attraction": ["landmark", "sightseeing"],
        "zoo": ["family", "outdoor", "nature"],
        "aquarium": ["family", "indoor"],
        "theme_park": ["family", "outdoor"],
        "hotel": ["hotel"],
    },
    "leisure": {
        "park": ["park", "outdoor", "relaxed", "free", "nature"],
        "garden": ["garden", "outdoor", "relaxed", "nature"],
        "nature_reserve": ["nature", "outdoor", "quiet"],
    },
    "historic": {
        "monument": ["landmark", "history", "outdoor"],
        "memorial": ["landmark", "history", "outdoor"],
        "castle": ["landmark", "history", "culture"],
    },
    "amenity": {
        "restaurant": ["restaurant", "food"],
        "cafe": ["cafe", "food", "casual"],
    },
}

DURATIONS = {"activity": 90, "hotel": 0, "restaurant": 60}


def classify(tags: dict[str, str]) -> tuple[str, list[str]]:
    if tags.get("tourism") == "hotel":
        return "hotel", ["hotel"]
    if tags.get("amenity") in ("restaurant", "cafe"):
        return "restaurant", CATEGORY_TAGS["amenity"][tags["amenity"]]

    categories: list[str] = []
    for key, mapping in CATEGORY_TAGS.items():
        value = tags.get(key)
        if value in mapping:
            categories += mapping[value]
    return "activity", sorted(set(categories)) or ["sightseeing"]


def claims_from_tags(tags: dict[str, str]) -> dict[str, str]:
    wheelchair = WHEELCHAIR_MAP.get((tags.get("wheelchair") or "").lower(), "unknown")
    toilet = WHEELCHAIR_MAP.get((tags.get("toilets:wheelchair") or "").lower(), "unknown")

    tactile = "unknown"
    if tags.get("tactile_paving") in ("yes", "no"):
        tactile = tags["tactile_paving"]
    elif tags.get("braille") == "yes":
        tactile = "yes"

    parking = "unknown"
    if tags.get("capacity:disabled") not in (None, "", "0", "no"):
        parking = "yes"

    return {
        "step_free_entrance": wheelchair,
        "accessible_toilet": toilet,
        "lift_access": "yes" if tags.get("elevator") == "yes" else "unknown",
        "wheelchair_rental": "unknown",
        "accessible_parking": parking,
        "quiet_space": "unknown",
        "audio_guide_captioned": "unknown",
        "tactile_or_braille": tactile,
        "assistance_animals": "yes" if tags.get("dog") == "leashed" else "unknown",
    }


def harvest(city: str, limit: int) -> dict:
    print(f"  querying Overpass for {city}...")
    # Overpass takes form-encoded QL, not JSON, so this uses its own request
    # rather than atex.httpjson.post_json.
    data = _overpass(city)

    places, seen = [], set()
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        name = tags.get("name")
        if not name:
            continue

        lat = element.get("lat") or (element.get("center") or {}).get("lat")
        lon = element.get("lon") or (element.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue

        place_id = f"{slugify(city)[:3]}-{slugify(name)}"[:60]
        if place_id in seen:
            continue
        seen.add(place_id)

        kind, categories = classify(tags)
        places.append({
            "id": place_id,
            "city": city,
            "country": tags.get("addr:country", ""),
            "name": name,
            "kind": kind,
            "area": tags.get("addr:suburb", "") or tags.get("addr:city", ""),
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "categories": categories,
            "typical_duration_min": DURATIONS[kind],
            "price_level": "mid",
            "crowd_level": "unknown",
            "noise_level": "unknown",
            "accessibility_claims": claims_from_tags(tags),
            "accessibility_source": f"osm:{element.get('type')}/{element.get('id')}",
            "source_url": f"https://www.openstreetmap.org/{element.get('type')}/{element.get('id')}",
            "last_verified": None,
            "notes": tags.get("wheelchair:description", ""),
        })

    # Prefer entries that actually carry an accessibility tag; they are the ones
    # worth having, and OSM returns a long tail of untagged venues.
    places.sort(
        key=lambda p: (
            p["accessibility_claims"]["step_free_entrance"] == "unknown",
            p["kind"] != "activity",
            p["name"],
        )
    )
    return {
        "city": city,
        "_provenance": (
            "Harvested from OpenStreetMap via the Overpass API. Claims are derived from "
            "wheelchair=*, toilets:wheelchair=*, tactile_paving=* and related tags. "
            "OSM data is crowd-sourced and may be incomplete or outdated; 'unknown' means "
            "the tag is absent, not that the feature is missing."
        ),
        "_generated": date.today().isoformat(),
        "_license": "OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        "places": places[:limit],
    }


def _overpass(city: str) -> dict:
    import urllib.parse
    import urllib.request

    body = urllib.parse.urlencode({"data": QUERY.format(city=city)}).encode("utf-8")
    request = urllib.request.Request(
        OVERPASS_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "ATEX-course-project/0.1 (accessibility research)",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cities", nargs="+", help="city names exactly as tagged in OSM")
    parser.add_argument("--limit", type=int, default=40, help="max places per city")
    parser.add_argument("--out", default=str(SEED_DIR), help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, city in enumerate(args.cities):
        if index:
            time.sleep(5)  # Overpass fair use
        try:
            payload = harvest(city, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {city} failed: {type(exc).__name__}: {exc}")
            continue

        tagged = sum(
            1 for p in payload["places"]
            if p["accessibility_claims"]["step_free_entrance"] != "unknown"
        )
        path = out_dir / f"{slugify(city)}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {city}: {len(payload['places'])} places ({tagged} with a wheelchair tag)")
        print(f"    -> data/seed/{path.name}")


if __name__ == "__main__":
    main()
