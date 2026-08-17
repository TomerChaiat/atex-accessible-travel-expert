"""Configuration, backend selection, and run guardrails.

Two ideas drive this module:

1. Every external dependency (LLM, embeddings, vector DB, relational DB) is
   selected by name here and resolved to a *fake* offline implementation when
   its credentials are absent. The system therefore boots and runs a full
   itinerary with no API keys at all.
2. All cost/latency limits live in one `Budget` object rather than being
   scattered through the agents, so they can be tuned or tightened in one place
   and asserted against in tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEED_DIR = DATA_DIR / "seed"
KB_DIR = DATA_DIR / "kb"
SESSION_DIR = DATA_DIR / "sessions"
WEB_DIR = ROOT / "web"
STATIC_DIR = ROOT / "atex" / "static"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Budget:
    """Hard bounds on a single /api/execute run.

    Each limit has a *soft* value and a small reserve. The supervisor loop stops
    as soon as a soft limit trips; the reserve is what remains for the
    forced-finalize path, so a run that exhausts its budget still returns a
    real itinerary instead of an error or a timeout.
    """

    max_supervisor_turns: int = 8
    max_total_llm_calls: int = 20
    max_tokens_per_run: int = 60_000
    wall_clock_budget_s: float = 240.0

    react_max_iters: int = 4
    rag_top_k: int = 5
    max_candidates_per_search: int = 6
    max_validations_per_run: int = 12

    # Places judged per AccessibilityValidator call. Batching is the largest
    # single saving in the system: one call for three places instead of three.
    validation_batch_size: int = 3

    # Headroom kept back for SchedulePlanner after a soft limit trips.
    reserve_llm_calls: int = 3
    reserve_wall_clock_s: float = 30.0
    reserve_tokens: int = 8_000

    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    @classmethod
    def from_env(cls) -> "Budget":
        d = cls()
        return cls(
            max_supervisor_turns=_env_int("ATEX_MAX_SUPERVISOR_TURNS", d.max_supervisor_turns),
            max_total_llm_calls=_env_int("ATEX_MAX_TOTAL_LLM_CALLS", d.max_total_llm_calls),
            max_tokens_per_run=_env_int("ATEX_MAX_TOKENS_PER_RUN", d.max_tokens_per_run),
            wall_clock_budget_s=_env_float("ATEX_WALL_CLOCK_BUDGET_S", d.wall_clock_budget_s),
            react_max_iters=_env_int("ATEX_REACT_MAX_ITERS", d.react_max_iters),
            rag_top_k=_env_int("ATEX_RAG_TOP_K", d.rag_top_k),
            max_candidates_per_search=_env_int(
                "ATEX_MAX_CANDIDATES", d.max_candidates_per_search
            ),
            max_validations_per_run=_env_int(
                "ATEX_MAX_VALIDATIONS", d.max_validations_per_run
            ),
            validation_batch_size=_env_int(
                "ATEX_VALIDATION_BATCH", d.validation_batch_size
            ),
            llm_timeout_s=_env_float("ATEX_LLM_TIMEOUT_S", d.llm_timeout_s),
        )


def _resolve(explicit: str, has_credentials: bool, real: str, fake: str) -> str:
    """Pick a backend: an explicit env override wins, otherwise auto-detect."""
    choice = (explicit or "auto").lower()
    if choice in (real, fake):
        return choice
    return real if has_credentials else fake


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    embedding_backend: str
    vector_backend: str
    repository_backend: str

    llmod_api_key: str
    llmod_base_url: str
    text_model: str
    embedding_model: str

    pinecone_api_key: str
    pinecone_index_host: str
    pinecone_namespace: str

    supabase_url: str
    supabase_service_key: str

    budget: Budget

    @property
    def offline(self) -> bool:
        """True when no external service is reachable/configured."""
        return self.llm_backend == "fake"

    def backend_summary(self) -> dict[str, str]:
        return {
            "llm": self.llm_backend,
            "embeddings": self.embedding_backend,
            "vector_store": self.vector_backend,
            "repository": self.repository_backend,
        }


def load_settings() -> Settings:
    llmod_key = _env("LLMOD_API_KEY")
    pinecone_key = _env("PINECONE_API_KEY")
    pinecone_host = _env("PINECONE_INDEX_HOST")
    supabase_url = _env("SUPABASE_URL")
    supabase_key = _env("SUPABASE_SERVICE_KEY")

    return Settings(
        llm_backend=_resolve(_env("ATEX_LLM_BACKEND"), bool(llmod_key), "llmod", "fake"),
        embedding_backend=_resolve(
            _env("ATEX_EMBEDDING_BACKEND"), bool(llmod_key), "llmod", "fake"
        ),
        vector_backend=_resolve(
            _env("ATEX_VECTOR_BACKEND"),
            bool(pinecone_key and pinecone_host),
            "pinecone",
            "memory",
        ),
        repository_backend=_resolve(
            _env("ATEX_REPOSITORY_BACKEND"),
            bool(supabase_url and supabase_key),
            "supabase",
            "local",
        ),
        llmod_api_key=llmod_key,
        llmod_base_url=_env("LLMOD_BASE_URL", "https://api.llmod.ai/v1").rstrip("/"),
        text_model=_env("LLMOD_TEXT_MODEL", "MB5R2CF-azure/gpt-5.4-mini"),
        embedding_model=_env(
            "LLMOD_EMBEDDING_MODEL", "MB5R2CF-azure/text-embedding-3-small"
        ),
        pinecone_api_key=pinecone_key,
        pinecone_index_host=pinecone_host.rstrip("/"),
        pinecone_namespace=_env("PINECONE_NAMESPACE", "atex-accessibility"),
        supabase_url=supabase_url.rstrip("/"),
        supabase_service_key=supabase_key,
        budget=Budget.from_env(),
    )


_cached: Settings | None = None


def settings(refresh: bool = False) -> Settings:
    """Process-wide settings. `refresh=True` re-reads the environment (tests)."""
    global _cached
    if _cached is None or refresh:
        _cached = load_settings()
    return _cached
