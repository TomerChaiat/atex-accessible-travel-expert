"""Turn the scraped article CSVs into knowledge-base chunks.

Offline work, like harvest_osm.py: it reads data/*.csv and writes the same JSON
shape that data/kb/ already uses, so scripts/ingest_kb.py can embed the result
without knowing where it came from. Nothing here calls a network service, so it
needs no API keys.

    python scripts/chunk_articles.py --report        # inspect, write nothing
    python scripts/chunk_articles.py                 # the three seeded cities
    python scripts/chunk_articles.py --all-cities    # every article, ~16k chunks

The article scrapers stripped paragraph breaks (median newline count is zero),
so splitting is sentence-based rather than on blank lines. Every chunk carries
its article title as a prefix: chunk 7 of a Berlin guide is unintelligible to an
embedding model without it, and the city filter depends on it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.config import DATA_DIR, KB_DIR, SEED_DIR  # noqa: E402

# One row of wheelchairtraveling holds 136k characters. Without a cap a single
# article would dominate the index.
MAX_CHUNKS_PER_DOC = 40
TARGET_CHARS = 1000
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 120

SOURCES = {
    "wheelchairtravel": {
        "csv": "wheelchairtravel_articles.csv",
        "source": "wheelchairtravel.org",
        "provenance": "scraped-article",
        "title": "title",
        "text": "text",
        "url": "url",
        "date": "timestamp",
    },
    "wheelchairtraveling": {
        "csv": "wheelchairtraveling_articles.csv",
        "source": "wheelchairtraveling.com",
        "provenance": "scraped-article",
        "title": "title",
        "text": "text",
        "url": "url",
        "date": "last_updated",
    },
}

# Accents and short forms that the seed catalogue spells plainly. Without these
# "Sagrada Familia" never matches an article writing "Sagrada Família".
EXTRA_ALIASES = {
    "bcn-sagrada-familia": ["Sagrada Familia", "Sagrada Família", "Sagrada Familía"],
    "bcn-park-guell": ["Park Guell", "Park Güell", "Parc Güell", "Parc Guell"],
    "bcn-ciutadella-park": ["Parc de la Ciutadella", "Ciutadella Park", "Ciutadella"],
    "bcn-barceloneta-beach": ["Barceloneta"],
    "bcn-picasso-museum": ["Museu Picasso", "Picasso Museum"],
    "bcn-magic-fountain": ["Magic Fountain", "Font Magica", "Font Màgica"],
    "ams-van-gogh-museum": ["Van Gogh Museum", "Van Gogh"],
    "ams-rijksmuseum": ["Rijksmuseum", "Rijks Museum"],
    "ams-anne-frank-house": ["Anne Frank House", "Anne Frank Huis", "Anne Frank"],
    "ams-nemo-science-museum": ["NEMO Science Museum", "NEMO"],
    "ams-hortus-botanicus": ["Hortus Botanicus"],
    "ams-vondelpark": ["Vondelpark", "Vondel Park"],
    "ber-brandenburg-gate": ["Brandenburg Gate", "Brandenburger Tor"],
    "ber-neues-museum": ["Neues Museum"],
    "ber-east-side-gallery": ["East Side Gallery"],
    "ber-jewish-museum": ["Jewish Museum Berlin", "Jewish Museum", "Judisches Museum"],
    "ber-reichstag-dome": ["Reichstag Dome", "Reichstag"],
    "ber-tiergarten": ["Tiergarten"],
}

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+")
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Ave|No|vs|etc|e\.g|i\.e|approx|ft|Inc)\.$", re.I)
_WS = re.compile(r"\s+")


# Leftovers from the scrape: WordPress shortcodes, bare embed markup and stray
# URLs. They survive as tokens that mean nothing to an embedding model.
_ARTEFACTS = re.compile(
    r"\[[^\]]{0,80}(?:embed|caption|gallery|/vc_|shortcode)[^\]]{0,200}\]"
    r"|<[^>]{1,200}>"
    r"|https?://\S+",
    re.I,
)

# Concrete accessibility vocabulary. Used only by --require-signal.
SIGNAL = re.compile(
    r"step[- ]free|\bramps?\b|\blifts?\b|elevator|accessible (?:toilet|restroom|entrance|room)"
    r"|disabled toilet|wheelchair access|wide door|threshold|cobble|kerb|curb|handrail"
    r"|level access|no steps|too steep|gradient|barrier[- ]free|grab rail|roll[- ]in shower",
    re.I,
)


def _clean(text: str) -> str:
    return _WS.sub(" ", _ARTEFACTS.sub(" ", text or "").replace(" ", " ")).strip()


def split_sentences(text: str) -> list[str]:
    """Sentence split that does not break on common abbreviations.

    Deliberately simple: the corpus is web prose, and an occasional bad split
    costs far less than a dependency would.
    """
    parts = _SENTENCE_END.split(text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def pack(
    text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS
) -> Iterator[str]:
    """Pack sentences into ~`target`-char windows with a sentence-aligned tail.

    The overlap is carried as whole trailing sentences rather than a raw
    character slice, so no chunk begins mid-clause.
    """
    window: list[str] = []
    size = 0
    for sentence in split_sentences(text):
        # A sentence longer than the window (missing punctuation in the scrape)
        # is hard-split, otherwise it would never fit and stall the loop.
        while len(sentence) > target:
            if window:
                yield " ".join(window)
                window, size = [], 0
            cut = sentence.rfind(" ", 0, target) or target
            yield sentence[:cut].strip()
            sentence = sentence[cut:].strip()

        if size + len(sentence) + 1 > target and window:
            yield " ".join(window)
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(window):
                if tail_size + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + 1
            window, size = tail, tail_size
        window.append(sentence)
        size += len(sentence) + 1

    if window:
        yield " ".join(window)


def load_gazetteer() -> list[tuple[str, str, re.Pattern[str]]]:
    """(place_id, city, pattern) for every seeded place, longest name first.

    Matching longest-first stops "Anne Frank" from claiming a chunk that names
    the "Anne Frank House" specifically.
    """
    entries: list[tuple[str, str, re.Pattern[str]]] = []
    for path in sorted(SEED_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        city = payload.get("city", "")
        for place in payload.get("places", []):
            names = list(EXTRA_ALIASES.get(place["id"], []))
            plain = re.sub(r"\s*\(.*?\)", "", place.get("name", "")).strip()
            if plain and plain not in names:
                names.append(plain)
            names = [n for n in names if len(n) >= 5]
            if not names:
                continue
            pattern = re.compile(
                r"\b(?:%s)\b" % "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)),
                re.I,
            )
            entries.append((place["id"], city, pattern))
    return entries


def detect_city(title: str, body: str, cities: list[str], min_mentions: int) -> str:
    """The city an article is *about*, not merely one it name-drops.

    A single mention is worthless as a signal: a business-class flight review
    that transits Schiphol mentions Amsterdam once and is otherwise about
    airline seating. Requiring the city in the title, or `min_mentions` times in
    the body, is what separates a city guide from a passing reference. The
    strongest city wins so an article naming two cities is filed under the one
    it actually covers.
    """
    best, best_score = "", 0
    for city in cities:
        pattern = re.compile(rf"\b{re.escape(city)}\b", re.I)
        in_title = len(pattern.findall(title))
        in_body = len(pattern.findall(body))
        if not in_title and in_body < min_mentions:
            continue
        # A title mention outranks any amount of body repetition.
        score = in_body + (1000 if in_title else 0)
        if score > best_score:
            best, best_score = city, score
    return best


def normalise_date(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def age_years(iso: str) -> int:
    if not iso:
        return -1
    try:
        then = date.fromisoformat(iso)
    except ValueError:
        return -1
    return max(0, (date.today() - then).days // 365)


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(
    key: str,
    cities: list[str],
    all_cities: bool,
    gazetteer: list[tuple[str, str, re.Pattern[str]]],
    min_mentions: int = 3,
    require_signal: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    spec = SOURCES[key]
    rows = read_csv(DATA_DIR / spec["csv"])

    # wheelchairtravel repeats 239 articles across category pages; the URL is
    # the identity, so the first occurrence wins.
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        # Not _clean(): that strips URLs out of prose, which would blank the
        # dedup key and collapse every row onto one entry.
        unique.setdefault((row[spec["url"]] or "").strip(), row)

    stats = {
        "rows": len(rows),
        "unique": len(unique),
        "kept": 0,
        "chunks": 0,
        "placed": 0,
        "dropped_no_signal": 0,
    }
    chunks: list[dict[str, Any]] = []

    for url, row in unique.items():
        title = _clean(row[spec["title"]])
        body = _clean(row[spec["text"]])
        if not body:
            continue

        city = detect_city(title, body, cities, min_mentions)
        if not city and not all_cities:
            continue
        stats["kept"] += 1

        retrieved = normalise_date(row.get(spec["date"], ""))
        age = age_years(retrieved)
        doc_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]

        for index, body_chunk in enumerate(pack(body)):
            if index >= MAX_CHUNKS_PER_DOC:
                break
            if len(body_chunk) < MIN_CHUNK_CHARS:
                continue
            if require_signal and not SIGNAL.search(body_chunk):
                stats["dropped_no_signal"] += 1
                continue

            text = f"{title}. {body_chunk}" if title else body_chunk

            place_id = ""
            for candidate_id, place_city, pattern in gazetteer:
                if (not city or place_city == city) and pattern.search(text):
                    place_id = candidate_id
                    break
            if place_id:
                stats["placed"] += 1

            chunks.append(
                {
                    "id": f"{key[:3]}-{doc_id}-{index}",
                    "place_id": place_id,
                    "city": city,
                    "text": text,
                    "source": spec["source"],
                    "source_url": url,
                    "provenance": spec["provenance"],
                    "retrieved_at": retrieved,
                    "doc_id": doc_id,
                    "age_years": age,
                }
            )
        stats["chunks"] = len(chunks)

    return chunks, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Amsterdam", "Barcelona", "Berlin"],
        help="cities to keep; articles naming none of them are skipped",
    )
    parser.add_argument(
        "--all-cities", action="store_true", help="keep every article, not just --cities"
    )
    parser.add_argument(
        "--city-mentions",
        type=int,
        default=3,
        help="body mentions needed to file an article under a city (title always wins)",
    )
    parser.add_argument(
        "--require-signal",
        action="store_true",
        help="drop chunks with no concrete accessibility vocabulary",
    )
    parser.add_argument("--report", action="store_true", help="print stats, write nothing")
    parser.add_argument("--out", default=str(KB_DIR), help="output directory")
    args = parser.parse_args()

    gazetteer = load_gazetteer()
    print(f"Gazetteer: {len(gazetteer)} places from data/seed/")

    out_dir = Path(args.out)
    total = 0
    for key in SOURCES:
        chunks, stats = build(
            key,
            args.cities,
            args.all_cities,
            gazetteer,
            min_mentions=args.city_mentions,
            require_signal=args.require_signal,
        )
        total += len(chunks)
        print(f"\n{key}")
        print(f"  rows {stats['rows']} -> {stats['unique']} unique -> {stats['kept']} in scope")
        print(f"  chunks: {len(chunks)}  ({stats['placed']} matched a place_id)")
        if stats["dropped_no_signal"]:
            print(f"  dropped for no accessibility signal: {stats['dropped_no_signal']}")
        if chunks:
            lengths = sorted(len(c["text"]) for c in chunks)
            print(
                f"  chunk chars min/median/max: "
                f"{lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}"
            )

        if args.report or not chunks:
            continue

        payload = {
            "_provenance": (
                f"Scraped from {SOURCES[key]['source']}. Each chunk records source_url and "
                "retrieved_at; verify before presenting any claim as current."
            ),
            "_generated": date.today().isoformat(),
            "chunks": chunks,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {path}")

    print(f"\nTotal: {total} chunks")
    if not args.report:
        print(
            "\nNote: ingest_kb.py reads every *.json in data/kb/, including the synthetic "
            "demo-corpus.json. Move it aside before ingesting real data."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
