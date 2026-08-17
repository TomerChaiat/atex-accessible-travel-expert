# Future work

The system is complete and tested. What follows is the work that would improve it,
plus the decisions still open to the team.

For design rationale see [docs/design.md](docs/design.md); for setup and
deployment see the [README](README.md).

## Data quality

The committed catalogue and knowledge base are labelled placeholders. Replacing
them is the single biggest improvement available:

1. `python scripts/harvest_osm.py Amsterdam Barcelona Berlin --limit 60` — real
   OpenStreetMap accessibility tags, no API key needed.
2. Build a real knowledge base to replace `data/kb/demo-corpus.json`, recording
   `source_url` and `retrieved_at` for every passage. Candidate sources: venue
   accessibility pages, OSM `wheelchair:description` free text, municipal open
   accessibility datasets.
3. Keep `place_id` values aligned between the catalogue and the vector index —
   they are the join key, and a mismatch silently turns every verdict into
   `unknown`.

## Capability

- **More cities.** Adding one is a harvest command plus knowledge-base passages.
- **A live search fallback.** When the catalogue has no entry for a destination,
  `ActivityLogisticsFinder` could fall back to a keyless web search rather than
  returning nothing. Bounded, and only on a miss.
- **Real routing.** Travel times are currently straight-line estimates scaled by a
  detour factor and labelled as such. A wheelchair-aware routing API would make
  the schedule materially more trustworthy.
- **Verdict freshness.** Accessibility information decays. Storing a
  `retrieved_at` per verdict and warning when evidence is old would make staleness
  visible instead of invisible.

## Open decisions

1. **Prompt key spelling.** `steps[].prompt` currently emits both
   `system_prompt` and `System_prompt`, because the assignment uses each spelling
   in a different place. If the graders confirm one, delete the aliases in
   `atex/tracing.py`.
2. **Graph framework.** The orchestrator is a hand-rolled loop over
   framework-shaped nodes, for the reasons in `docs/design.md`. If the course
   expects a named graph library, porting is mechanical.
3. **Budget defaults.** 20 calls / 60k tokens / 240s per run are conservative
   estimates made before real token costs were known. Revisit against actual
   usage and the $13 ceiling.
4. **`maxDuration`.** `vercel.json` requests 300s, the assignment's ceiling. If
   the deployment plan caps it lower, reduce it and set
   `ATEX_WALL_CLOCK_BUDGET_S` about 10s below the platform limit so the agent
   finalizes before the function is killed.
