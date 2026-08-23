"""Read/write helpers for the local embedding cache.

Lives in atex/ rather than scripts/ because both sides need it: the offline
embedder writes the cache, and InMemoryVectorStore reads it so local runs get
real semantic vectors without re-embedding the whole knowledge base on every
process start.

Vectors are stored as base64 float32. That is exact, roughly eight times smaller
than JSON floats, and needs nothing outside the standard library.
"""

from __future__ import annotations

import base64
import hashlib
import json
from array import array
from pathlib import Path
from typing import Any

CACHE_DIRNAME = "vectors"


def cache_path(kb_file: Path) -> Path:
    """data/kb/foo.json -> data/kb/vectors/foo.jsonl"""
    return kb_file.parent / CACHE_DIRNAME / f"{kb_file.stem}.jsonl"


def text_digest(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def encode_vector(vector: list[float]) -> str:
    return base64.b64encode(array("f", vector).tobytes()).decode("ascii")


def decode_vector(blob: str) -> list[float]:
    values = array("f")
    values.frombytes(base64.b64decode(blob))
    return list(values)


def load_cache(path: Path, model: str, dimensions: int) -> dict[str, dict[str, Any]]:
    """Cached entries by chunk id, or {} when the cache cannot be trusted.

    A cache built by a different embedder is discarded rather than mixed in: the
    fake embedder's 256-dim hash vectors and text-embedding-3-small's 1536-dim
    vectors are not comparable, and silently blending them would produce
    retrieval results that look plausible and are meaningless.
    """
    if not path.exists():
        return {}

    entries: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if not lines:
        return {}

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError:
        return {}
    if header.get("_model") != model or header.get("_dimensions") != dimensions:
        return {}

    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("id") and record.get("vector"):
            entries[record["id"]] = {
                "hash": record.get("hash", ""),
                "vector": record["vector"],
                "line": line,
            }
    return entries
