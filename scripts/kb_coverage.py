"""Report how well the accessibility knowledge base covers a city.

Most "unverified" results are a coverage gap, not a judgement: across observed
runs, 71-93% of candidates had no retrievable passage at all. This measures
that directly, so an example prompt can be chosen for a city the corpus
actually knows something about rather than by guesswork.

    python scripts/kb_coverage.py                       # a default shortlist
    python scripts/kb_coverage.py Rome Paris Barcelona  # any cities you like

Needs PINECONE_API_KEY, PINECONE_INDEX_HOST and LLMOD_API_KEY (for the query
embedding). Read-only: it queries the index and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.config import settings  # noqa: E402
from atex.embeddings import build_embedder  # noqa: E402
from atex.vectorstore import build_vector_store  # noqa: E402

# Cities worth checking first: the three the examples use today, plus the large
# tourist destinations an accessibility corpus is most likely to cover.
DEFAULT_CITIES = [
    "Amsterdam", "New York City", "Rome",
    "London", "Paris", "Barcelona", "Berlin", "Madrid", "Vienna", "Prague",
    "Lisbon", "Dublin", "Copenhagen", "Stockholm", "Venice", "Florence",
    "Los Angeles", "San Francisco", "Chicago", "Boston", "Washington",
    "Toronto", "Sydney", "Tokyo", "Singapore", "Dubai",
]

PROBE = (
    "wheelchair accessibility step free entrance accessible toilet lift "
    "museum attraction opening hours"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cities", nargs="*", default=[], help="cities to probe")
    parser.add_argument("--top-k", type=int, default=200, help="passages to sample")
    args = parser.parse_args()

    cfg = settings(refresh=True)
    if cfg.vector_backend != "pinecone":
        print(
            "No Pinecone credentials found, so this would measure the local "
            "knowledge base instead. Set PINECONE_API_KEY and "
            "PINECONE_INDEX_HOST to probe the real corpus."
        )
        return 1

    embedder = build_embedder(cfg)
    vectors = build_vector_store(cfg, embedder)
    probe = embedder.embed([PROBE])[0]

    cities = args.cities or DEFAULT_CITIES
    print(f"Index: {cfg.pinecone_index_host}  namespace: {cfg.pinecone_namespace!r}")
    print(f"Sampling the top {args.top_k} passages per city.\n")
    print(f"{'city':<20} {'passages':>9}  {'named venues (sample)'}")
    print("-" * 78)

    scored: list[tuple[int, str, list[str]]] = []
    for city in cities:
        try:
            matches = vectors.query(probe, top_k=args.top_k, flt={"city": city})
        except Exception as exc:  # noqa: BLE001 - a probe should not abort a sweep
            print(f"{city:<20} {'error':>9}  {exc}")
            continue
        # A passage that names a venue is what actually produces a verdict; a
        # general city note cannot verify anything on its own.
        venues = []
        for match in matches:
            name = str(match.metadata.get("entity_name") or "").strip()
            if name and name not in venues:
                venues.append(name)
        scored.append((len(matches), city, venues))
        print(f"{city:<20} {len(matches):>9}  {', '.join(venues[:4])[:46]}")

    scored.sort(reverse=True)
    print("\nBest covered:")
    for count, city, venues in scored[:5]:
        print(f"  {city} - {count} passages, {len(venues)} distinct venues named")
    if scored and scored[0][0] == 0:
        print("  (nothing matched: check that PINECONE_NAMESPACE matches ingestion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
