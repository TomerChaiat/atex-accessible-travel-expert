"""Embed the accessibility knowledge base and upsert it into Pinecone.

Run once after changing data/kb/. This is offline work: it does not touch the
per-request budget, and it is the only place embeddings are generated in bulk.

    python scripts/ingest_kb.py            # needs PINECONE_* and LLMOD_API_KEY
    python scripts/ingest_kb.py --dry-run  # parse and report, call nothing

Index requirements: dimension 1536, metric cosine (text-embedding-3-small).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.config import KB_DIR, settings  # noqa: E402
from atex.embeddings import build_embedder  # noqa: E402
from atex.vectorstore import VectorRecord, build_vector_store, chunk_metadata  # noqa: E402

BATCH = 64


def load_chunks() -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for path in sorted(KB_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for chunk in payload.get("chunks", []):
            if not chunk.get("id") or not chunk.get("text"):
                print(f"  ! skipping malformed chunk in {path.name}")
                continue
            records.append(
                VectorRecord(
                    id=chunk["id"],
                    text=chunk["text"],
                    # Same normalisation the in-memory store uses, so filters
                    # behave identically across backends.
                    metadata=chunk_metadata(chunk),
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = settings(refresh=True)
    records = load_chunks()
    if not records:
        print("No chunks found in data/kb/. Nothing to do.")
        return 1

    synthetic = sum(1 for r in records if r.metadata.get("provenance") == "synthetic-demo")
    print(f"Loaded {len(records)} chunks from data/kb/")
    if synthetic:
        print(
            f"  WARNING: {synthetic} of them are synthetic demo text. Ingesting these into a "
            "production index would put invented accessibility claims behind real verdicts."
        )

    print(f"  embeddings backend: {cfg.embedding_backend}")
    print(f"  vector backend:     {cfg.vector_backend}")

    if args.dry_run:
        print("Dry run: nothing was called.")
        return 0

    if cfg.vector_backend != "pinecone":
        print("\nPINECONE_API_KEY / PINECONE_INDEX_HOST are not set, so there is no index to")
        print("write to. The in-memory store reads data/kb/ directly at query time, so the")
        print("system already works offline without this step.")
        return 1

    embedder = build_embedder(cfg)
    store = build_vector_store(cfg, embedder)

    total = 0
    for start in range(0, len(records), BATCH):
        batch = records[start : start + BATCH]
        vectors = embedder.embed([r.text for r in batch])
        for record, vector in zip(batch, vectors):
            record.values = vector
        total += store.upsert(batch)
        print(f"  upserted {total}/{len(records)}")

    print(f"Done. {total} vectors in namespace '{cfg.pinecone_namespace}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
