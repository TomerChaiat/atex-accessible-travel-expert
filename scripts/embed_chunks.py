"""Embed knowledge-base chunks into a local vector cache.

Splits the expensive half of ingestion away from the Pinecone half. Embedding is
paid work that produces a durable artefact; upserting is cheap and repeatable.
Fusing them, as ingest_kb.py originally did, means every retry, namespace change
or metadata fix re-pays for embeddings that have not changed.

    python scripts/embed_chunks.py             # embed anything not cached
    python scripts/embed_chunks.py --force     # re-embed everything

Then either:

    python scripts/ingest_kb.py --from-cache   # upload without re-embedding
    python scripts/devserver.py                # local runs reuse the cache too

Cache layout is one JSONL file per knowledge-base file, in data/kb/vectors/.
Each line records the chunk id, a hash of the exact text embedded, and the
vector as base64 float32 -- about 8x smaller than JSON floats and exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.config import KB_DIR, settings  # noqa: E402
from atex.embeddings import build_embedder  # noqa: E402
from atex.vectorcache import (  # noqa: E402
    cache_path,
    encode_vector,
    load_cache,
    text_digest,
)

BATCH = 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-embed even if cached")
    parser.add_argument("--kb", default=str(KB_DIR), help="knowledge-base directory")
    args = parser.parse_args()

    cfg = settings(refresh=True)
    embedder = build_embedder(cfg)
    print(f"Embedder: {embedder.name} ({embedder.dimensions} dims)")
    if embedder.name == "fake":
        print(
            "  WARNING: no LLMOD_API_KEY, so this caches hash-based vectors. They are\n"
            "  not semantic and must not be uploaded to Pinecone."
        )

    kb_dir = Path(args.kb)
    total_new = 0

    for path in sorted(kb_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [c for c in payload.get("chunks", []) if c.get("id") and c.get("text")]
        if not chunks:
            continue

        out = cache_path(path)
        cached = {} if args.force else load_cache(out, embedder.name, embedder.dimensions)

        # A chunk is stale when its text changed since it was embedded, which is
        # what makes re-running the chunker safe.
        pending = [c for c in chunks if cached.get(c["id"], {}).get("hash") != text_digest(c["text"])]

        print(f"\n{path.name}: {len(chunks)} chunks, {len(pending)} to embed")
        if not pending:
            continue

        lines = {
            cid: entry["line"]
            for cid, entry in cached.items()
            if entry.get("line")
        }
        for start in range(0, len(pending), BATCH):
            batch = pending[start : start + BATCH]
            vectors = embedder.embed([c["text"] for c in batch])
            if len(vectors) != len(batch):
                print(f"  ! expected {len(batch)} vectors, got {len(vectors)}; aborting")
                return 1
            for chunk, vector in zip(batch, vectors):
                lines[chunk["id"]] = json.dumps(
                    {
                        "id": chunk["id"],
                        "hash": text_digest(chunk["text"]),
                        "vector": encode_vector(vector),
                    }
                )
            done = min(start + BATCH, len(pending))
            print(f"  embedded {done}/{len(pending)}")

            # Written after every batch so an interrupted run keeps its work.
            out.parent.mkdir(parents=True, exist_ok=True)
            header = json.dumps(
                {"_model": embedder.name, "_dimensions": embedder.dimensions}
            )
            out.write_text(
                "\n".join([header, *lines.values()]) + "\n", encoding="utf-8"
            )

        total_new += len(pending)
        print(f"  -> {out}")

    print(f"\nDone. {total_new} new vectors cached.")
    if total_new:
        print("Next: python scripts/ingest_kb.py --from-cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
