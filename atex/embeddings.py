"""Text embeddings, with an offline deterministic fallback.

The fake embedder is a hashed bag-of-words projection. It is not semantic, but
it does capture lexical overlap, which is enough for the RAG pipeline to
retrieve sensible accessibility passages while no API key is available.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings
from .httpjson import post_json

FAKE_DIMENSIONS = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class FakeEmbedder:
    name = "fake"
    dimensions = FAKE_DIMENSIONS

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_RE.findall((text or "").lower())
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            # Sign bit spreads tokens across the space instead of piling them up.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


class LLModEmbedder:
    name = "llmod"
    dimensions = 1536  # text-embedding-3-small

    def __init__(self, settings: Settings):
        self._settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = post_json(
            f"{self._settings.llmod_base_url}/embeddings",
            {"model": self._settings.embedding_model, "input": texts},
            headers={"Authorization": f"Bearer {self._settings.llmod_api_key}"},
            timeout=60.0,
        )
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "llmod":
        return LLModEmbedder(settings)
    return FakeEmbedder()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
