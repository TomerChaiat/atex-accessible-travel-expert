# ATEX Project Handoff Prompt

Copy the prompt below into a new chat when continuing development of this project.

---

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
14. The user-facing itinerary should show only the distance between consecutive venues, for example: `1.29 km`. Do not show calculated travel time or describe it as an accessible route.
15. Itinerary times must be arithmetically contiguous. For example:
    - 09:30 activity lasting 90 minutes
    - 11:00 lunch lasting 60 minutes
    - 12:00 next activity

    Do not insert unexplained hidden gaps.
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

