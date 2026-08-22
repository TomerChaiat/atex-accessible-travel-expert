"""AccessibilityValidator: the only module allowed to judge accessibility.

The retrieval step does real work before any model is consulted, and that has a
useful consequence: when nothing in the knowledge base mentions a place, the
verdict is `unknown` *deterministically*, with no LLM call at all. Honesty and
cost control point the same way here -- the cheapest answer is also the correct
one.
"""

from __future__ import annotations

import re
from typing import Any

from .. import ACCESSIBILITY_VALIDATOR
from ..context import AgentContext
from ..prompts import VALIDATOR_SYSTEM, validator_user_prompt
from ..state import Candidate, RunState
from ..util import truncate

DEFAULT_NEEDS = ["step_free_entrance", "accessible_toilet"]
EVIDENCE_CHARS = 700
SEMANTIC_CANDIDATE_POOL = 80


def _normalise_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def _candidate_aliases(name: str) -> list[str]:
    """Return conservative names that can prove a passage is about a place."""
    without_note = re.sub(r"\([^)]*\)", " ", name or "")
    aliases = {
        _normalise_name(name),
        _normalise_name(without_note),
        _normalise_name(re.sub(r"\baccessible\b", " ", without_note, flags=re.I)),
    }
    return sorted((alias for alias in aliases if len(alias) >= 4), key=len, reverse=True)


def _mentions_candidate(match: Any, candidate: Candidate) -> bool:
    """Require semantic fallback evidence to explicitly identify the candidate."""
    if str(match.metadata.get("place_id") or "") == candidate.place_id:
        return True
    searchable = [
        match.text,
        match.metadata.get("entity_name", ""),
        match.metadata.get("title", ""),
        match.metadata.get("thread_title", ""),
    ]
    haystack = _normalise_name(" ".join(str(value or "") for value in searchable))
    return any(alias in haystack for alias in _candidate_aliases(candidate.name))


def _candidate_excerpt(text: str, candidate: Candidate) -> str:
    """Keep the candidate mention inside the passage shown to the validator."""
    value = text or ""
    for alias in _candidate_aliases(candidate.name):
        tokens = alias.split()
        pattern = r"\b" + r"\W+".join(re.escape(token) for token in tokens) + r"\b"
        match = re.search(pattern, value, flags=re.I)
        if not match:
            continue
        start = max(0, match.start() - 180)
        excerpt = value[start : start + EVIDENCE_CHARS]
        if start:
            excerpt = "…" + excerpt
        if start + EVIDENCE_CHARS < len(value):
            excerpt += "…"
        return excerpt
    return truncate(value, EVIDENCE_CHARS)


def _retrieval_query(candidate: Candidate, city: str, needs: list[str]) -> str:
    readable = " ".join(need.replace("_", " ") for need in needs)
    return f"{candidate.name} {city} accessibility {readable} wheelchair step free entrance toilet lift"


def retrieve_evidence(
    ctx: AgentContext, candidate: Candidate, city: str, needs: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (place-specific evidence, general city evidence).

    Exact catalogue IDs are preferred. The enriched corpus contains many useful
    legacy records without catalogue place IDs, so a larger semantic search is
    used as a fallback. A fallback passage is accepted as place-specific only
    when its text or identity metadata explicitly names the candidate.
    """
    top_k = ctx.settings.budget.rag_top_k
    query_vector = ctx.embedder.embed([_retrieval_query(candidate, city, needs)])[0]

    def to_entry(match, *, place_specific: bool = False) -> dict[str, Any]:
        return {
            "id": match.id,
            "text": (
                _candidate_excerpt(match.text, candidate)
                if place_specific
                else truncate(match.text, EVIDENCE_CHARS)
            ),
            "source": match.metadata.get("source", "unknown"),
            "provenance": match.metadata.get("provenance", "unknown"),
        }

    exact = ctx.vectors.query(
        query_vector, top_k=top_k, flt={"place_id": candidate.place_id}
    )
    specific_matches = list(exact)
    seen = {match.id for match in specific_matches}

    if len(specific_matches) < top_k:
        # Google Place IDs cannot equal IDs from the older CSV corpus. Search
        # inside the destination first so a venue article is not crowded out
        # by the other ~15k vectors, then retain only passages that explicitly
        # name this candidate. The unfiltered pass covers records whose city
        # metadata is absent while keeping the same identity check.
        semantic_pools = []
        if city:
            semantic_pools.append(
                ctx.vectors.query(
                    query_vector,
                    top_k=max(SEMANTIC_CANDIDATE_POOL, top_k * 12),
                    flt={"city": city},
                )
            )
        for pool in semantic_pools:
            for match in pool:
                if match.id in seen or not _mentions_candidate(match, candidate):
                    continue
                specific_matches.append(match)
                seen.add(match.id)
                if len(specific_matches) >= top_k:
                    break

        # Only pay for the global fallback when destination-scoped retrieval
        # did not already fill the evidence batch.
        if len(specific_matches) < top_k:
            for match in ctx.vectors.query(
                query_vector,
                top_k=max(SEMANTIC_CANDIDATE_POOL, top_k * 12),
            ):
                if match.id in seen or not _mentions_candidate(match, candidate):
                    continue
                specific_matches.append(match)
                seen.add(match.id)
                if len(specific_matches) >= top_k:
                    break

    specific = [
        to_entry(match, place_specific=True) for match in specific_matches[:top_k]
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
        "met_needs": [],
        "unmet_needs": [],
        "concerns": [],
        "conditions": [],
        "evidence_ids": [],
        "summary": reason,
    }


def _sanitize(
    result: dict[str, Any],
    evidence_ids: set[str],
    required_needs: list[str] | None = None,
    traveler_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hold the model to the rules rather than trusting it to follow them.

    A verdict with no cited evidence is downgraded to `unknown`, and cited ids
    that were never supplied are dropped. This is the enforcement point for the
    project's central promise.
    """
    verdict = str(result.get("verdict", "")).lower()
    if verdict not in {"supported", "flagged", "unknown"}:
        verdict = "unknown"

    cited = [i for i in (result.get("evidence_ids") or []) if i in evidence_ids]
    met_needs = [str(n) for n in (result.get("met_needs") or [])][:8]
    unmet_needs = [str(n) for n in (result.get("unmet_needs") or [])][:8]
    concerns = [truncate(str(c), 180) for c in (result.get("concerns") or [])][:5]
    conditions = [truncate(str(c), 180) for c in (result.get("conditions") or [])][:4]
    summary = truncate(str(result.get("summary") or ""), 300)

    if verdict == "supported" and not cited:
        verdict = "unknown"

    required = set(required_needs or [])
    met = set(met_needs)
    core_required = required & {
        "step_free_entrance",
        "accessible_toilet",
        "lift_access",
    }
    barrier_text = " ".join([summary, *concerns, *conditions]).casefold()
    barrier_markers = (
        "not accessible",
        "few steps",
        "stairs",
        "narrow entrance",
        "narrow doorway",
        "advance arrangement",
        "advance registration",
    )
    helper_markers = (
        "with a helper",
        "visit with a helper",
        "visiting with a helper",
        "with a companion",
        "companion required",
        "requires a companion",
        "requires assistance",
    )
    helper_required = any(marker in barrier_text for marker in helper_markers)
    companion_available = bool((traveler_context or {}).get("companion_available"))
    unresolved_markers = (
        "not confirmed",
        "unconfirmed",
        "not directly confirmed",
        "not addressed",
        "not available",
        "no accessible toilet",
    )
    has_other_problem = bool(unmet_needs) or any(
        marker in barrier_text for marker in (*barrier_markers, *unresolved_markers)
    )

    if helper_required and not conditions:
        conditions.append("Visit with a companion or helper.")

    # Partial, cited evidence is useful but not a full verification. Present it
    # as a concern rather than erasing it into the same bucket as no evidence.
    if verdict == "unknown" and cited and met:
        verdict = "flagged"
    if (
        verdict == "flagged"
        and helper_required
        and companion_available
        and not has_other_problem
        and (not core_required or core_required.issubset(met))
    ):
        # The only condition is already satisfied by this traveller's profile.
        # Keep it visible, but do not reject an otherwise supported venue.
        verdict = "supported"
    if verdict == "supported" and (
        unmet_needs
        or (core_required and not core_required.issubset(met))
        or any(marker in barrier_text for marker in barrier_markers)
        or (helper_required and not companion_available)
    ):
        verdict = "flagged"

    return {
        "verdict": verdict,
        "met_needs": met_needs,
        "unmet_needs": unmet_needs,
        "concerns": concerns,
        "conditions": conditions,
        "evidence_ids": cited,
        "summary": summary,
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
    primary_city = state.destination or (state.profile or {}).get("destination") or ""
    profile = state.profile or {}
    mobility = profile.get("mobility") or {}
    party_size = int(profile.get("party_size") or 1)
    assistant_present = mobility.get("assistant_present")
    traveler_context = {
        "party_size": party_size,
        "assistant_present": assistant_present,
        "companion_available": assistant_present is True or party_size > 1,
        "travelling_solo": party_size == 1 and assistant_present is not True,
    }

    items: list[dict[str, Any]] = []
    with_evidence: list[Candidate] = []
    evidence_ids_by_place: dict[str, set[str]] = {}

    for candidate in batch:
        # Multi-city candidates must be retrieved inside their own destination;
        # using the primary city here would hide evidence for every day trip.
        city = str(candidate.brief.get("city") or primary_city)
        specific, general = retrieve_evidence(ctx, candidate, city, needs)
        if not specific:
            ctx.trace.note(
                f"AccessibilityValidator: no evidence for {candidate.name} "
                f"({candidate.place_id}); "
                "returned unknown without an LLM call"
            )
            _assign(
                candidate,
                _unknown("No accessibility information for this place in the knowledge base."),
            )
            continue

        evidence = specific + general
        evidence_ids_by_place[candidate.place_id] = {e["id"] for e in evidence}
        items.append({"place": candidate.brief, "evidence": evidence})
        with_evidence.append(candidate)

    if not with_evidence:
        return

    result = ctx.llm.complete_json(
        ACCESSIBILITY_VALIDATOR,
        VALIDATOR_SYSTEM,
        validator_user_prompt(items, needs, traveler_context),
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
        _assign(
            candidate,
            _sanitize(
                entry,
                evidence_ids_by_place[candidate.place_id],
                needs,
                traveler_context,
            ),
        )


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
