# Local place catalogue

**Empty on purpose.** This repository does not ship a place catalogue.

Production discovers places live through the Google Places API. This directory
is the keyless fallback, and the output target of the harvester:

```bash
python scripts/harvest_osm.py Amsterdam Barcelona Berlin --limit 60
```

That writes one `<city>.json` per city from OpenStreetMap, where
`wheelchair=yes|limited|no`, `toilets:wheelchair` and `tactile_paving` are
structured tags on real venues, and an absent tag becomes `unknown` rather than
a guess. No API key is needed.

With this directory empty and no `GOOGLE_MAPS_API_KEY` set, place search returns
no results and says so. That is deliberate: the alternative is inventing venues,
which is the one thing this system exists not to do.

The test suite has its own catalogue under `tests/fixtures/seed/`. It is test
data, scoped to the tests, and is never read at runtime.
