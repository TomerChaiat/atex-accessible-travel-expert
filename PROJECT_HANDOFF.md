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
- The bundled local catalogue and fake backend are only offline fallbacks, not the intended production data source.

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
19. A hotel is never an activity. It belongs in "Where you'll stay", so a trip
    that keeps one hotel throughout has no hotel row in any day — scheduling
    it duplicates the section and, with its zero duration, collides with the
    next item's start time. The one exception is a genuine change of
    accommodation: a hotel other than the selected one, on a row explicitly
    marked `kind: "stay"`, is kept as the day the traveller moves in. A move
    still passes through normal verdict enforcement, so a hotel with
    accessibility concerns cannot enter the itinerary this way.
    - Known limitation: `selected_hotel_id` is a single value and the finder
      returns one hotel, so a multi-hotel trip is not yet plannable
      end-to-end. The `stay` row is the seam that change would build on.
20. The GUI must offer a one-click download of the full run logs, covering
    every turn in the conversation.
16. Google Place IDs must be copied exactly from provider observations. Invalid, altered, or obsolete IDs should be skipped without crashing the itinerary.
17. Do not expose Pinecone confidence fields or `classification_version` in the application or UI.
18. Accessibility evidence explanations should be concise but complete and must not end with a truncated sentence.

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

