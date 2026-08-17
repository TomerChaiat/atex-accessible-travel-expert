"""AccessibilityValidator: the only module allowed to judge accessibility.

The retrieval step does real work before any model is consulted, and that has a
useful consequence: when nothing in the knowledge base mentions a place, the
verdict is `unknown` *deterministically*, with no LLM call at all. Honesty and
cost control point the same way here -- the cheapest answer is also the correct
one.
"""

from __future__ import annotations

from typing import Any

from .. import ACCESSIBILITY_VALIDATOR
from ..context import AgentContext
from ..prompts import VALIDATOR_SYSTEM, validator_user_prompt
from ..state import Candidate, RunState
from ..util import truncate

DEFAULT_NEEDS = ["step_free_entrance", "accessible_toilet"]
EVIDENCE_CHARS = 700


def _retrieval_query(candidate: Candidate, city: str, needs: list[str]) -> str:
    readable = " ".join(need.replace("_", " ") for need in needs)
    return f"{candidate.name} {city} accessibility {readable} wheelchair step free entrance toilet lift"


def retrieve_evidence(
    ctx: AgentContext, candidate: Candidate, city: str, needs: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (place-specific evidence, general city evidence).

    Two filtered queries share one embedding. Filtering by place_id rather than
    ranking the whole corpus guarantees that a venue's own passages are found:
    an unfiltered top-k can crowd them out and produce a false "unknown", which
    is the one failure mode this module must not have.
    """
    top_k = ctx.settings.budget.rag_top_k
    query_vector = ctx.embedder.embed([_retrieval_query(candidate, city, needs)])[0]

    def to_entry(match) -> dict[str, Any]:
        return {
            "id": match.id,
            "text": truncate(match.text, EVIDENCE_CHARS),
            "source": match.metadata.get("source", "unknown"),
            "provenance": match.metadata.get("provenance", "unknown"),
            "score": round(match.score, 4),
        }

    specific = [
        to_entry(m)
        for m in ctx.vectors.query(
            query_vector, top_k=top_k, flt={"place_id": candidate.place_id}
        )
    ]

    general = []
    if city:
        general = [
            to_entry(m)
            for m in ctx.vectors.query(
                query_vector, top_k=2, flt={"scope": "city", "city": city}
            )
        ]

    return specific, general[:2]


def _unknown(reason: str) -> dict[str, Any]:
    return {
        "verdict": "unknown",
        "confidence": 0.0,
        "met_needs": [],
        "unmet_needs": [],
        "concerns": [],
        "evidence_ids": [],
        "summary": reason,
    }


def _sanitize(result: dict[str, Any], evidence_ids: set[str]) -> dict[str, Any]:
    """Hold the model to the rules rather than trusting it to follow them.

    A verdict with no cited evidence is downgraded to `unknown`, and cited ids
    that were never supplied are dropped. This is the enforcement point for the
    project's central promise.
    """
    verdict = str(result.get("verdict", "")).lower()
    if verdict not in {"supported", "flagged", "unknown"}:
        verdict = "unknown"

    cited = [i for i in (result.get("evidence_ids") or []) if i in evidence_ids]
    if verdict == "supported" and not cited:
        verdict = "unknown"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "verdict": verdict,
        "confidence": 0.0 if verdict == "unknown" else max(0.0, min(confidence, 1.0)),
        "met_needs": [str(n) for n in (result.get("met_needs") or [])][:8],
        "unmet_needs": [str(n) for n in (result.get("unmet_needs") or [])][:8],
        "concerns": [str(c) for c in (result.get("concerns") or [])][:5],
        "evidence_ids": cited,
        "summary": truncate(str(result.get("summary") or ""), 300),
    }


def _assign(candidate: Candidate, detail: dict[str, Any]) -> None:
    candidate.verdict = detail["verdict"]
    candidate.verdict_detail = detail


def validate_batch(
    ctx: AgentContext, state: RunState, batch: list[Candidate]
) -> None:
    """Judge up to `validation_batch_size` places in a single LLM call.

    Places with no retrieved evidence never reach the model at all: their
    verdict is `unknown` by definition, so spending a call on them would buy
    nothing.
    """
    needs = state.required_needs or DEFAULT_NEEDS
    city = state.destination or (state.profile or {}).get("destination") or ""

    items: list[dict[str, Any]] = []
    with_evidence: list[Candidate] = []
    evidence_ids: set[str] = set()

    for candidate in batch:
        specific, general = retrieve_evidence(ctx, candidate, city, needs)
        if not specific:
            ctx.trace.note(
                f"AccessibilityValidator: no evidence for {candidate.place_id}; "
                "returned unknown without an LLM call"
            )
            _assign(
                candidate,
                _unknown("No accessibility information for this place in the knowledge base."),
            )
            continue

        evidence = specific + general
        evidence_ids |= {e["id"] for e in evidence}
        items.append({"place": candidate.brief, "evidence": evidence})
        with_evidence.append(candidate)

    if not with_evidence:
        return

    result = ctx.llm.complete_json(
        ACCESSIBILITY_VALIDATOR,
        VALIDATOR_SYSTEM,
        validator_user_prompt(items, needs),
        max_tokens=400 + 320 * len(with_evidence),
    )

    by_id = {
        str(entry.get("place_id")): entry
        for entry in (result.get("verdicts") or [])
        if isinstance(entry, dict)
    }
    for candidate in with_evidence:
        entry = by_id.get(candidate.place_id)
        if entry is None:
            # The model skipped this place; unknown is the only honest default.
            _assign(candidate, _unknown("The validator returned no verdict for this place."))
            ctx.trace.note(
                f"AccessibilityValidator: no verdict returned for {candidate.place_id}"
            )
            continue
        _assign(candidate, _sanitize(entry, evidence_ids))


def run(ctx: AgentContext, state: RunState, instruction: str = "") -> None:
    budget = ctx.settings.budget
    pending = state.unvalidated()
    if not pending:
        state.log("AccessibilityValidator: nothing left to check")
        return

    allowance = max(0, budget.max_validations_per_run - state.validation_count)
    if allowance <= 0:
        # Returning with candidates still unvalidated would livelock the
        # supervisor, which would route straight back here. Settle them as
        # unknown instead: it is honest, and it lets the run make progress.
        for candidate in pending:
            _assign(candidate, _unknown("Not checked: the per-run validation cap was reached."))
        ctx.trace.note(
            f"AccessibilityValidator: validation cap ({budget.max_validations_per_run}) "
            f"reached; {len(pending)} place(s) settled as unknown"
        )
        state.log(f"AccessibilityValidator: cap reached, {len(pending)} left unknown")
        return

    pending = pending[:allowance]

    size = max(1, budget.validation_batch_size)
    checked = 0
    for start in range(0, len(pending), size):
        # Leave headroom so the planner can still run after this module.
        if ctx.trace.soft_exhausted():
            break
        batch = pending[start : start + size]
        validate_batch(ctx, state, batch)
        checked += len(batch)
        state.validation_count += len(batch)

    # Anything past the allowance, or skipped because the budget tripped, is
    # settled here for the same anti-livelock reason.
    for candidate in state.unvalidated():
        _assign(candidate, _unknown("Not checked: the planning budget was reached first."))

    counts = {v: len(state.by_verdict(v)) for v in ("supported", "flagged", "unknown")}
    state.log(f"AccessibilityValidator: checked {checked}; totals {counts}")
