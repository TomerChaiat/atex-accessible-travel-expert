"""Dependency container passed to every module.

Building it in one place is what makes the offline mode work: swapping the four
backends here is the only difference between a keyless local run and production.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, settings as load_settings
from .embeddings import Embedder, build_embedder
from .llm import TracedLLM, build_llm_backend
from .repository import Repository, build_repository
from .tracing import RunTrace
from .vectorstore import VectorStore, build_vector_store


@dataclass
class AgentContext:
    settings: Settings
    trace: RunTrace
    llm: TracedLLM
    repo: Repository
    embedder: Embedder
    vectors: VectorStore

    @classmethod
    def build(cls, trace: RunTrace, settings: Settings | None = None) -> "AgentContext":
        cfg = settings or load_settings()
        embedder = build_embedder(cfg)
        return cls(
            settings=cfg,
            trace=trace,
            llm=TracedLLM(build_llm_backend(cfg), trace, cfg),
            repo=build_repository(cfg),
            embedder=embedder,
            vectors=build_vector_store(cfg, embedder),
        )
