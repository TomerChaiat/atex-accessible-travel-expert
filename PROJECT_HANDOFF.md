I need your help continuing development of my project:

`atex-accessible-travel-expert-main`

It is an agentic AI accessible-travel planner deployed on Vercel. Please inspect the repository and its current Git status before changing anything. Do not assume previous changes were committed.

Important distinction: the application itself intentionally uses LLMs. Do not remove its LLM functionality. We only wanted to remove development artifacts or references suggesting that Claude Code, ChatGPT, or another coding assistant collaborated in building the repository.

## Project architecture

- UserProfileAgent extracts the traveller's profile.
- ActivityLogisticsFinder discovers real venues using Google Places.
- AccessibilityValidator searches Pinecone evidence and evaluates each venue against the traveller's needs.
- SchedulePlanner creates the itinerary.
- Supervisor coordinates the agents.
- LLMod supplies the text and embedding models.
- The production application does not use Supabase.
- The repository ships no place catalogue and no accessibility evidence: `data/seed/` and `data/kb/` are empty.
- The fake LLM backend and the keyless local backends are development fallbacks, never a production data source.
- `tests/fixtures/` holds a small catalogue and corpus used only by the test suite, so it runs without API keys.

## Production environment variables

```env
LLMOD_API_KEY=
LLMOD_BASE_URL=https://api.llmod.ai/v1
PINECONE_API_KEY=
PINECONE_INDEX_HOST=
PINECONE_NAMESPACE=
GOOGLE_MAPS_API_KEY=
```

The Pinecone index is `accessibility-knowledge`. It contains approximately 15,359 vectors generated from the enriched travel CSV files. Verify that `PINECONE_NAMESPACE` in Vercel exactly matches the namespace used during ingestion. Do not expose API keys in source code.

The `.env.example` file should exist locally for documentation but remain ignored by Git.

## Current product rules

1. Focus primarily on wheelchair users and people with walking or mobility disabilities.
2. At least one example request must be located in the USA.
3. Google Places is used to discover hotels and activities worldwide.
4. Pinecone evidence is used to validate accessibility.
5. Accessibility classifications:
   - `[verified accessible]` — green
   - `[NOT VERIFIED]` — yellow
   - `[accessibility concerns]` — red
6. Never recommend a venue with accessibility concerns in the main itinerary.
7. Venues with concerns should appear under "Considered but not scheduled," with a short, complete explanation.
8. Unverified venues are allowed in the itinerary when verified alternatives are insufficient. "Not verified" does not mean inaccessible.
9. Generic items such as lunch breaks, rest breaks, and hotel rest must not receive accessibility labels.
10. Lunch already counts as a break.
11. Remove rest breaks immediately beside lunch and at the end of a day.
12. Keep no more than one meaningful explicit rest per day.
13. Traveller context matters. For example, a venue requiring a helper may be unsuitable for a solo traveller but conditionally supported when the traveller has an appropriate companion.
14. Travel between consecutive venues must be explained, not just numbered.
    Name the origin and call the distance an estimate, for example: `The
    estimated distance from X is about 1.29 km.` The destination is the line
    directly above, so do not repeat it. Then list the travel modes that suit
    the traveller, each with a time. Never describe any of it as an accessible
    route.
    - Only offer modes the traveller can actually manage. Self-powered travel
      is capped by profile: 3.0 km powered chair or scooter, 1.5 km manual,
      0.8 km walker or cane. An accessible taxi is always available, so there
      is always at least one option.
    - When the traveller names a preferred way of getting around, show only
      that one.
    - Live times come from the Google Routes API; every failure falls back to
      the local coordinate estimate for that hop. Google's walking time is
      scaled up for wheelchair and limited-walking travellers, because Routes
      has no wheelchair mode.
15. Itinerary times must be arithmetically contiguous, and travel time counts.
    Each start equals the previous item's end plus the travel shown on the new
    item's own row. For example:
    - 09:30 activity lasting 90 minutes
    - 11:00 lunch lasting 60 minutes
    - 12:00 plus 20 minutes of travel → 12:20 next activity

    The schedule is laid out on the slowest offered mode, so it holds whichever
    the traveller picks. Do not insert gaps the itinerary has not explained.
16. Google Place IDs must be copied exactly from provider observations. Invalid, altered, or obsolete IDs should be skipped without crashing the itinerary.
17. Do not expose Pinecone confidence fields or `classification_version` in the application or UI.
18. Accessibility evidence explanations should be concise but complete and must not end with a truncated sentence.
19. A hotel is never an attraction. It belongs in "Where you'll stay". A
    multi-location trip has one selected hotel per contiguous location range,
    and the response names the location and inclusive day range for every
    stay. A genuine hotel change may also appear once on its move-in day as
    `kind: "stay"`. Every hotel still passes through normal verdict enforcement,
    so one with accessibility concerns is replaced when possible and never
    presented as the selected stay.
20. The GUI must offer a one-click download of the full run logs, covering
    every turn in the conversation.
21. How full a day is, is the Supervisor's decision, not a fixed table. It
    reads the request and sets `plan_shape.days` once, with an independent
    location and attraction target for every day plus `day_start` and
    `day_end`. Arrival, transfer, and recovery days can be lighter than the
    others. The finder sizes discovery per location and the planner must honor
    each individual target rather than repeat one number across the trip.
22. Never show a provider place ID to the traveller. "Confirm before you
    travel" carries the venue name and what to check, nothing else.
23. "Considered but not scheduled" is for places actively rejected — a stated
    accessibility concern, an unreachable location, a duplicate. A merely
    unverified place that was not needed is surplus, not a rejection, and must
    not be listed. Cap the section and collapse the remainder into one count.
24. Unless the traveller explicitly says to remain only in the named
    location(s), the Supervisor may devote days to realistic nearby cities or
    regional day trips. Explicit destinations must all be covered, and
    `requested_locations_only` is a hard boundary. Discovery is capped at four
    Finder rounds and is repeated while a planned location lacks attractions
    or hotel coverage.
25. A day's attraction target is enforced in code, not left to the planner.
    After the itinerary comes back, any day short of its target is topped up
    from checked candidates nobody scheduled — verified first, then
    unverified — matching that day's location, and stopping at `day_end`. A
    Los Angeles run asked for three a day and returned two while 48 checked
    candidates sat unused.
26. Public transport is not offered for a hop over 40 minutes. Distance gates
    walking; time gates transit, because a short hop with three changes costs
    as much as a long one. The same Los Angeles run offered a 145-minute bus
    ride, and since the schedule is laid out on the slowest option that hop
    swallowed two and a half hours of the day. At least one option always
    survives — the traveller is never told there is no way to get there.

27. A follow-up names only what it changes. Fields the new extraction leaves
    empty are refilled from the saved profile, so "I want a different hotel"
    keeps the city, the trip length, the wheelchair and the stated needs. An
    explicitly stated value, including `false`, always wins over the old one.
    An update naming no location at all is not a destination change and must
    not discard paid-for candidates and verdicts.
28. Asking for a different hotel releases the current selection before
    discovery runs. Replanning alone cannot honour it — the itinerary is
    rebuilt from the same candidates, so the same hotel wins again. The
    rejected hotel stays in `candidates`, which is what stops the next search
    offering it straight back.

29. Location names are matched loosely. "Los Angels" and "Los Angeles" are
    the same city; treating them as two split a fourteen-day stay into days
    1-13 and day 14 with a separate hotel for each. Genuinely different
    cities, including short similar ones, stay separate.
30. A traveller can change hotel without changing city. The Supervisor may
    give `plan_shape.hotel_stays` explicitly — "one hotel the first week and
    a different one the second". A split is used only when it covers every day
    exactly once, in order, with no gap or overlap; anything malformed falls
    back to the geographic derivation rather than being patched up.
31. The change of hotel appears in the day it happens, with time for checking
    out, travelling with luggage and checking in. That row is logistics, not a
    visit, so it carries no accessibility label — the hotel's verdict belongs
    in "Where you'll stay".
32. Each day allows time to get from the hotel to its first stop. Days used to
    begin at `day_start` sharp at the first attraction, as though the traveller
    woke up inside it. Prefer a hotel central to that stay's attractions.
33. A request that is not about travel is declined by the Supervisor on its
    first turn, before any other module runs, and the loop stops there —
    including the forced-finalize path. Declining costs one model call. A
    vague travel request is not out of scope; that is what ASK_USER is for.
    - The reply is fixed wording (`OUT_OF_SCOPE_MESSAGE` in `atex/render.py`)
      and never varies. The off-topic subject is not echoed back, and the
      model's reason stays in the trace rather than reaching the traveller.

34. `search_locations` holds one entry per distinct place, not one per day.
    Appending unconditionally made a single-city fortnight title itself
    "Los Angeles, Los Angeles, Los Angeles, ..." fourteen times over.
35. A filled stop carries no note. The traveller asked for that many stops a
    day, so a filled one is what they requested rather than something to
    apologise for, and an unverified venue already has its own line under
    "Confirm before you travel".
36. No hop between two stops on the same day may exceed 75 minutes. The
    transit cap does not cover this: once transit is dropped for being slow,
    the schedule is laid out on the taxi instead, and a 165-minute taxi ride
    became the plan. Days are topped up nearest-first, within 30 km, and a
    stop that still comes out too far is dropped and one pass of refilling
    runs in its place.
37. Each stay is re-based on the hotel closest to the attractions planned for
    its days. The finder chooses hotels before the itinerary exists, so it
    cannot know where the traveller will actually spend their time — a Los
    Angeles fortnight based at the airport is why getting anywhere took hours.
    A verified hotel outranks a closer unverified one, and a hotel another
    stay already holds is never taken, so a requested split cannot collapse
    onto one building.

## Quick examples

- "We are a family of four visiting Amsterdam for three days. Our daughter uses a manual wheelchair. We prefer a relaxed pace, no more than two activities per day. We need a verified accessible hotel."
- "Four days in New York City. I use a powered wheelchair and need step-free entrances and accessible toilets. I prefer no more than two attractions per day."
- "Two days in Rome with my father, who uses a walker and cannot walk long distances. We need step-free places, short distances, and a relaxed pace."

## Working preferences

- Inspect the existing implementation before proposing changes.
- Preserve unrelated and uncommitted work.
- Implement requested fixes directly when authorized.
- Add regression tests for important behavior.
- Run the complete test suite and syntax checks after changes.
- Do not commit or push for me.
- When finished, tell me exactly which files changed and provide the precise `git add`, `git commit`, and `git push` commands so I can push them myself.
- Keep explanations practical and easy to follow.
