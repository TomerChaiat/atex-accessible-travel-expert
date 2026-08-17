"""Vector storage for the accessibility knowledge base.

Two implementations behind one interface: Pinecone for production, and an
in-memory cosine index built from data/kb/ for offline work. The
AccessibilityValidator cannot tell them apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import KB_DIR, Settings
from .embeddings import Embedder, cosine_similarity
from .httpjson import post_json


@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    values: list[float] | None = None


@dataclass
class Match:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalise a KB chunk's metadata.

    `scope` exists because Pinecone drops null metadata values, so "this passage
    is about the city rather than a specific venue" cannot be expressed as
    place_id=null in a filter. An explicit scope survives the round trip.
    """
    place_id = chunk.get("place_id") or ""
    return {
        "place_id": place_id,
        "scope": "place" if place_id else "city",
        "city": chunk.get("city") or "",
        "source": chunk.get("source") or "unknown",
        "source_url": chunk.get("source_url") or "",
        "provenance": chunk.get("provenance", "unknown"),
        "retrieved_at": chunk.get("retrieved_at") or "",
    }


class VectorStore(Protocol):
    name: str

    def upsert(self, records: list[VectorRecord]) -> int: ...

    def query(
        self, vector: list[float], top_k: int, flt: dict[str, Any] | None = None
    ) -> list[Match]: ...


class InMemoryVectorStore:
    """Cosine-similarity index over the local knowledge base."""

    name = "memory"

    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._records: list[VectorRecord] = []
        self._loaded = False

    def upsert(self, records: list[VectorRecord]) -> int:
        pending = [r for r in records if r.values is None]
        if pending:
            vectors = self._embedder.embed([r.text for r in pending])
            for record, vector in zip(pending, vectors):
                record.values = vector
        by_id = {r.id: r for r in self._records}
        for record in records:
            by_id[record.id] = record
        self._records = list(by_id.values())
        return len(records)

    def load_local_kb(self) -> int:
        """Read data/kb/*.json once, lazily, on first query."""
        if self._loaded:
            return len(self._records)
        self._loaded = True
        if not KB_DIR.exists():
            return 0

        records: list[VectorRecord] = []
        for path in sorted(KB_DIR.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for chunk in payload.get("chunks", []):
                records.append(
                    VectorRecord(
                        id=chunk["id"],
                        text=chunk["text"],
                        metadata=chunk_metadata(chunk),
                    )
                )
        if records:
            self.upsert(records)
        return len(records)

    def query(
        self, vector: list[float], top_k: int, flt: dict[str, Any] | None = None
    ) -> list[Match]:
        self.load_local_kb()
        scored: list[Match] = []
        for record in self._records:
            if flt and not _matches_filter(record.metadata, flt):
                continue
            score = cosine_similarity(vector, record.values or [])
            scored.append(Match(record.id, score, record.text, record.metadata))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]


def _matches_filter(metadata: dict[str, Any], flt: dict[str, Any]) -> bool:
    for key, expected in flt.items():
        actual = metadata.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


class PineconeVectorStore:
    name = "pinecone"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._headers = {
            "Api-Key": settings.pinecone_api_key,
            "X-Pinecone-API-Version": "2025-01",
        }

    def upsert(self, records: list[VectorRecord]) -> int:
        vectors = [
            {
                "id": r.id,
                "values": r.values,
                # Pinecone rejects null metadata values, so drop them.
                "metadata": {
                    k: v for k, v in {**r.metadata, "text": r.text}.items() if v is not None
                },
            }
            for r in records
            if r.values
        ]
        if not vectors:
            return 0
        total = 0
        # Pinecone caps request size; 100 vectors per batch stays well under it.
        for start in range(0, len(vectors), 100):
            batch = vectors[start : start + 100]
            post_json(
                f"{self._settings.pinecone_index_host}/vectors/upsert",
                {"vectors": batch, "namespace": self._settings.pinecone_namespace},
                headers=self._headers,
                timeout=60.0,
            )
            total += len(batch)
        return total

    def query(
        self, vector: list[float], top_k: int, flt: dict[str, Any] | None = None
    ) -> list[Match]:
        payload: dict[str, Any] = {
            "vector": vector,
            "topK": top_k,
            "includeMetadata": True,
            "namespace": self._settings.pinecone_namespace,
        }
        if flt:
            payload["filter"] = flt
        data = post_json(
            f"{self._settings.pinecone_index_host}/query",
            payload,
            headers=self._headers,
            timeout=30.0,
        )
        matches = []
        for item in data.get("matches") or []:
            metadata = dict(item.get("metadata") or {})
            text = metadata.pop("text", "")
            matches.append(
                Match(item.get("id", ""), float(item.get("score", 0.0)), text, metadata)
            )
        return matches


def build_vector_store(settings: Settings, embedder: Embedder) -> VectorStore:
    if settings.vector_backend == "pinecone":
        return PineconeVectorStore(settings)
    return InMemoryVectorStore(embedder)
