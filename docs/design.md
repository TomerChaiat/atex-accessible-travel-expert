# ATEX — design and rationale

This document explains *why* ATEX is built the way it is. For how to run,
configure and deploy it, see the [README](../README.md).

Module names used here are the canonical ones from `atex/__init__.py`, identical
to those in the architecture diagram and in every `steps[].module` entry.

---

## The problem

Planning a trip as a disabled traveller means assembling information that no
single source holds. Accessibility details are scattered across venue pages,
blogs, forum threads and community groups; they go stale silently; and mainstream
travel sites compress accessibility into a single yes/no flag.

That flag is the core failure. "Accessible" is not one property. A step-free
entrance is worth little without an accessible toilet. A lift to every floor does
not help someone whose barrier is noise and crowding. A beach with an amphibious
chair is accessible only during staffed hours in season.

And the cost of being wrong is asymmetric. A missing restaurant reservation is an
inconvenience; a venue that turns out to have six steps at the entrance can end
the day. Wrong information is worse than absent information — which is why the
central design commitment of this project is that **"we don't know" is a
first-class answer**, surfaced rather than smoothed over.

## Who it is for

- Travellers with disabilities planning independently
- Families travelling with a disabled member, coordinating mixed needs
- Travel agents assembling accessible itineraries

## Why a multi-agent architecture

The task decomposes cleanly into four jobs that share almost nothing: reading a
person's needs out of prose, searching a catalogue, judging evidence, and
arranging a schedule under constraints.

A single prompt doing all four would be long, would mix retrieval with judgement,
and would make failure hard to localise — when the itinerary is wrong you cannot
tell whether the profile was misread, the search was poor, or the accessibility
call was wrong. Splitting the work means each module has a short prompt, a
narrow contract, a separate place in the trace, and can be improved in isolation.

The decisive reason is the accessibility judgement. Confining that to one module
lets us forbid it everywhere else: every other prompt explicitly states that it
may not assert accessibility. A monolithic agent has no such seam.

---

## The modules

| Module | Pattern | Responsibility |
|---|---|---|
| `Supervisor` | LLM router | Chooses the next module each turn, resolves conflicts, decides when to replan or ask the user |
| `UserProfileAgent` | Single extraction call | Prose → structured profile: mobility, sensory needs, pace, budget, interests |
| `ActivityLogisticsFinder` | ReAct | Thought → Action → Observation over the catalogue; shortlists activities, hotels, restaurants |
| `AccessibilityValidator` | RAG | Retrieves evidence per place and returns `supported` / `flagged` / `unknown` with citations |
| `SchedulePlanner` | Single generation call | Day-by-day itinerary: geographic grouping, pacing, rest |

---

## Design decisions

### 1. An autonomous supervisor, with invariants enforced in code

`Supervisor` is a real LLM decision on every turn. It is not a fixed pipeline: it
can revisit a module, send the finder back for alternatives after a poor batch of
verdicts, or stop and ask the user a question.

Full autonomy alone, though, is not safe to deploy against a metered budget and a
300-second platform limit. So the loop around it enforces *legality*, not a
route. If the model picks a module that would break an invariant — planning
before every place has a verdict, re-searching a third time — the choice is
corrected to the required module, and the correction is recorded in the trace.

The distinction matters: the prompt *requests* an ordering, the code
*guarantees* it. A prompt is a hope; an invariant is a property.

### 2. Live place discovery, with a deterministic offline fallback

The finder calls Google Places API (New) at request time, through a small
provider interface. This removes the previous three-city catalogue boundary:
destinations such as Rome can produce real attractions and hotels without a
manual seed import. The same interface uses a bundled JSON sample when the
Google key is absent, which keeps local tests deterministic.

Reasons, in order of weight:

The live search is kept safe for Vercel by a strict field mask, short HTTP
timeouts, one retry, result caps, and the bounded ReAct loop. The
finder stores compact result briefs in the prompt rather than full provider
records.

The tools keep JSON schemas shaped for MCP compatibility, but production uses a
normal HTTPS API because a Vercel function cannot inherit a developer's local
MCP connections or signed-in browser session.

Google accessibility options are treated only as preliminary discovery hints.
The AccessibilityValidator still decides the final label from the independent
Pinecone evidence corpus; missing evidence stays `unknown`.

### 3. Accessibility honesty is a property of the code, not the prompt

Three mechanisms, none of which depends on the model cooperating:

- **No evidence, no model call.** If retrieval returns nothing for a place, the
  verdict is `unknown` deterministically. The honest answer and the cheapest one
  coincide.
- **Citations are verified.** Evidence ids the model did not receive are dropped,
  and a `supported` verdict citing nothing is downgraded to `unknown`.
- **The planner cannot upgrade a verdict.** After the itinerary returns, every
  accessibility label is overwritten from the recorded verdicts, and any flagged
  or unknown place that was scheduled is added to *Confirm before you travel*.

Retrieval is filtered by `place_id` rather than ranked across the whole corpus.
An unranked top-k search can crowd out a venue's own passages and produce a false
`unknown` — the one failure mode this system exists to prevent. There is a
regression test asserting that every place with a knowledge-base entry retrieves
it.

### 4. Travel between venues: routed late, and only for what survives

An itinerary that says "3.42 km" and nothing else has told the traveller
nothing. What they need is how to make the hop and how long it takes — and for
this audience, *which* ways are even possible.

Two decisions follow.

**Modes are filtered by the profile, not offered blindly.** Suggesting a 5 km
push to a manual wheelchair user is worse than offering no suggestion, so
`offered_modes()` drops self-powered travel past a per-profile ceiling: 3.0 km
for a powered chair or scooter, 1.5 km manual, 0.8 km for a walker or cane.
An accessible taxi has no ceiling, so there is always at least one answer. When
the traveller names how they want to get around, that single mode replaces the
choice.

**Routing happens after planning, not before.** The planner still receives the
cheap coordinate-only matrix, because all it needs is enough geography to avoid
zig-zagging. Live routing then runs over the handful of consecutive pairs that
actually reached the itinerary — a few elements per run instead of a quadratic
matrix over every candidate the finder looked at, which is the difference
between cents and dollars on the Google Routes bill.

`GoogleRoutesRouter` degrades per pair and per mode: no key, a disabled API, no
transit in this city, a quota error or a timeout each fall back to the local
estimate for that one hop. Losing routing must never lose the trip.

One correction is applied on top of Google's numbers. Routes has no wheelchair
mode, and its `WALK` duration assumes an able-bodied pedestrian at roughly
4.8 km/h. Quoting it unadjusted would understate every journey for precisely
the travellers this system exists to serve, so walking durations are scaled for
anyone with a wheelchair or a stated walking limit.

The schedule is then laid out on the **slowest** offered option, so the day
holds together whichever one the traveller picks, and each row states both the
distance and what the schedule allowed. Times stay addable: no gap appears that
the itinerary has not already explained.

### 5. The shape of a day is a decision, not a constant

How many attractions belong in a day started life as a lookup table: relaxed
two, moderate three, packed four, keyed off one word the profile agent chose.
It produced exactly the failure you would predict. A traveller who asked for
"more than two attractions per day and finishing the day late" got two stops
ending at 13:40, because the table had already decided before anyone read the
sentence.

Worse, the invented number was indistinguishable from a stated one. Downstream
modules could not tell "the traveller asked for three" from "a table guessed
three", so neither could override it sensibly.

Now `UserProfileAgent` records only what was actually said — `null` when the
traveller did not say — and the Supervisor decides `plan_shape`
(`activities_per_day`, `day_start`, `day_end`) once, from the request itself.
That decision flows to `ActivityLogisticsFinder`, which sizes its search by
`trip_days × activities_per_day`, and to `SchedulePlanner`, which is told to
*hit* that number rather than stay under it.

The reasoning belongs in the Supervisor because it is the module that sees the
whole request rather than a summarised field, and because it is already the
system's judgement layer. `normalize_plan_shape()` clamps the answer into
plannable bounds and derives a fallback if the field is missing — a guard, not
a policy.

### 6. Token and call efficiency

| Technique | Effect |
|---|---|
| Places passed as compact `brief` forms, never full records | Keeps finder prompts small |
| `Supervisor` sees a lossy state summary, not a transcript | Per-turn prompt stays ~constant as the trip grows |
| Verdicts batched, five places per call | 50 validator calls → 10 |
| Empty retrieval short-circuits the model | Free verdicts for unknown places |
| ReAct observations windowed to the last three | A long loop cannot inflate the prompt |
| Two empty searches end the finder loop early | A destination with no coverage costs 2 calls, not 8 |
| Final response rendered by code, not an LLM | Saves a call and cannot contradict the verdicts |

A three-day trip costs roughly 12–14 calls and ~15k tokens; a two-week trip
nearer 35–45 calls and ~90k tokens.

### 7. Guardrails with reserved headroom

Limits on supervisor turns, total calls, tokens and wall-clock live in one
`Budget` object. Each keeps a **reserve**.

Crossing a soft limit is not an error. It ends the supervisor loop and hands
control to *forced finalize*, which runs `SchedulePlanner` from the reserve and
marks anything unchecked as `unknown`. A run that exhausts its budget still
returns a complete, honest itinerary instead of an error or a timeout.

This also fixed a real defect found in testing: when the validation cap sat below
the candidate count, one place stayed permanently unvalidated and `Supervisor`
routed back to the validator until it burned every turn. Unchecked places are now
settled as `unknown`, which both terminates the loop and states the truth.

### 8. Conversation state on a stateless platform

Serverless functions keep nothing between requests, but the response schema is
fixed to four top-level fields, so a session id cannot be returned in it.

The browser generates the session id instead. A caller sending only
`{"prompt": "..."}` gets a clean one-shot run; the GUI sends an extra
`session_id` and gets follow-up turns. What persists is a compact state object —
profile, verdicts, itinerary draft — not a transcript. Verdicts are deliberately
reused across turns: they cost LLM calls and do not change between them, which
makes a follow-up turn roughly a fifth the cost of the first.

### 9. Every external service has an offline fallback

The LLM, embeddings, vector store and database each resolve to a local
implementation when credentials are absent. The system runs end-to-end, and the
whole test suite passes, with no API keys and nothing installed.

This was not a convenience. It let the entire architecture be built and tested
before any key existed, and it keeps the test suite free to run — no key, no
quota, no network.

### 10. A hand-rolled orchestrator over a graph framework

Nodes are written in the shape a graph framework expects —
`(ctx, state, instruction) -> None`, mutating shared state — but the loop driving
them is about 120 lines of plain Python.

The budget checks, invariant corrections and finalize path all need to inspect
state between every hop, which is awkward to express as conditional edges. Plain
control flow is also easier to read, and it keeps the serverless bundle small:
runtime dependencies are `fastapi` and `pydantic`, with all outbound HTTP on
stdlib `urllib`. Porting the same functions into a `StateGraph` is mechanical.

---

## What changed from the original proposal

| Proposed | Built | Why |
|---|---|---|
| Supervisor coordinating four agents | Unchanged | The decomposition held up |
| Live Maps APIs, web search, MCP tools | Curated catalogue via in-process tools | Latency, cost, the 300s cap, reproducibility |
| Knowledge base from named accessibility websites | Provenance recorded per passage; OSM harvester provided | We will not attribute claims to sources we have not actually ingested |
| One verdict per place | Verdicts batched five per call | Cuts validator calls by four fifths |
| "Never invent an answer" as a principle | Enforced by citation checks and verdict overwriting | A principle the code guarantees rather than requests |

---

## Data provenance

Nothing is committed. `data/seed/` and `data/kb/` ship empty, and the product
holds no place catalogue and no accessibility evidence of its own.

This is the provenance rule taken to its conclusion. A repository that ships
hand-authored accessibility claims has, by construction, a path where invented
content reaches a traveller — a demo left running, a fallback that fires in
production, a placeholder nobody replaced. Removing the content removes the
path. Keyless, place search returns nothing and every verdict is `unverified`,
which is exactly what this system promises to say when it does not know.

Each knowledge-base chunk records `source`, `source_url`, `provenance` and
`retrieved_at`, so the distinction between real and placeholder content survives
ingestion instead of being lost at the boundary. `scripts/harvest_osm.py` pulls
real OpenStreetMap tags into `data/seed/` without an API key.

The test suite keeps a small hand-authored catalogue and corpus under
`tests/fixtures/`, pointed at by `ATEX_SEED_DIR` and `ATEX_KB_DIR`. Scoping it
to the tests is what makes it safe: it buys a keyless, deterministic, offline
suite, and it cannot reach a runtime code path.

---

## Future work

See [PLAN.md](../PLAN.md).
