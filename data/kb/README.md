# Accessibility knowledge base

**Empty on purpose.** This repository does not ship accessibility evidence.

Production retrieves evidence from Pinecone (index `accessibility-knowledge`).
This directory is the keyless fallback and the ingestion source:

```bash
python scripts/ingest_kb.py
```

Each chunk must record `source`, `source_url`, `provenance` and `retrieved_at`,
so the distinction between real and placeholder content survives ingestion
instead of being lost at the boundary. Keep `place_id` values aligned with the
place catalogue — they are the join key, and a mismatch silently turns every
verdict into `unknown`.

With this directory empty and no Pinecone credentials, retrieval finds nothing
and every verdict comes back `unverified`. That is the correct answer when there
is no evidence, and exactly what this system promises to say.

The test suite has its own corpus under `tests/fixtures/kb/`. It is test data,
scoped to the tests, and is never read at runtime.
