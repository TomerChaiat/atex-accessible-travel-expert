# Test fixtures

Data used **only by the test suite**. Never read at runtime.

- `seed/` — a small place catalogue for three cities
- `kb/` — accessibility passages keyed to those places

These began life as the project's offline demo catalogue and were moved here
when `data/` stopped shipping invented content. Their value now is narrow and
real: they let the whole suite run with no API keys, no network and no
non-determinism, which is worth keeping.

Read them as fixtures, not as facts:

- Attraction entries name real institutions, but every accessibility claim is
  hand-authored and unverified.
- Hotel and restaurant entries are synthetic and match no real business.
- Every knowledge-base passage is synthetic. Nothing is quoted or derived from
  WheelchairTravel.org, TripAdvisor, or any other real source.

Tests point the keyless backends here by setting `ATEX_SEED_DIR` and
`ATEX_KB_DIR` before importing `atex`. Nothing else should ever set them to
this directory.
