"""The run state shared by every module, plus its compact supervisor view.

`RunState` is the graph's single mutable object (LangGraph-shaped: each node
takes it and returns it). `summary_for_supervisor` is deliberately lossy -- the
supervisor sees verdict counts and identifiers, never full place records or a
conversation transcript, which keeps its per-turn prompt roughly constant in
size no matter how large the trip gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tracing import RunTrace
from .util import truncate

VERDICTS = ("supported", "flagged", "unknown")

# Bounds on the shape the Supervisor chooses. Not a policy about how full a day
# should be -- that is the Supervisor's judgement, made per request -- only a
# guard so a misread number cannot produce an unplannable day.
MIN_ACTIVITIES_PER_DAY = 1
MAX_ACTIVITIES_PER_DAY = 6
MIN_DAY_MINUTES = 4 * 60

PACE_FALLBACK = {"relaxed": 2, "moderate": 3, "packed": 4}


def _parse_hhmm(value: Any) -> int | None:
    try:
        hours, minutes = str(value).strip().split(":", 1)
        hours_int, minutes_int = int(hours), int(minutes)
    except (TypeError, ValueError):
        return None
    if not 0 <= hours_int <= 23 or not 0 <= minutes_int <= 59:
        return None
    return hours_int * 60 + minutes_int


def _hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def normalize_plan_shape(
    raw: Any, profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Clamp the Supervisor's plan shape, and derive one if it gave none.

    How many stops a day holds and how late it runs are judgements about a
    specific request -- "more than two attractions per day and finishing late"
    means something a fixed table cannot express. The Supervisor decides;
    this only keeps the answer inside plannable bounds and supplies a
    reasonable shape if the decision is missing.
    """
    raw = raw if isinstance(raw, dict) else {}
    profile = profile or {}

    per_day = raw.get("activities_per_day")
    if not isinstance(per_day, int):
        try:
            per_day = int(per_day)
        except (TypeError, ValueError):
            # The traveller's own stated limit is the next best authority.
            per_day = profile.get("max_activities_per_day")
    if not isinstance(per_day, int):
        # Last resort only, for a Supervisor reply that omitted the field.
        # Not a policy table: the Supervisor's judgement is the primary path,
        # and it sees the whole request rather than one summarised word.
        per_day = PACE_FALLBACK.get(str(profile.get("pace") or ""), 3)
    per_day = max(MIN_ACTIVITIES_PER_DAY, min(per_day, MAX_ACTIVITIES_PER_DAY))

    start = _parse_hhmm(raw.get("day_start"))
    end = _parse_hhmm(raw.get("day_end"))
    if start is None:
        start = 9 * 60
    if end is None or end - start < MIN_DAY_MINUTES:
        # Long enough to hold the stops that were asked for, plus a meal.
        end = min(23 * 60, start + max(MIN_DAY_MINUTES, per_day * 120 + 60))

    return {
        "activities_per_day": per_day,
        "day_start": _hhmm(start),
        "day_end": _hhmm(end),
        "why": str(raw.get("why") or "")[:200],
    }


@dataclass
class Candidate:
    place_id: str
    name: str
    kind: str
    brief: dict[str, Any]
    verdict: str | None = None
    verdict_detail: dict[str, Any] = field(default_factory=dict)

    def to_planner_dict(self) -> dict[str, Any]:
        detail = self.verdict_detail
        return {
            "place_id": self.place_id,
            "name": self.name,
            "kind": self.kind,
            "categories": self.brief.get("categories", []),
            "duration_min": self.brief.get("duration_min", 90),
            "area": self.brief.get("area", ""),
            "accessibility": self.verdict or "unknown",
            "accessibility_summary": truncate(detail.get("summary", ""), 160),
            "concerns": detail.get("concerns", [])[:3],
            "conditions": detail.get("conditions", [])[:3],
        }


@dataclass
class RunState:
    request: str
    session_id: str | None = None
    turn_index: int = 0

    profile: dict[str, Any] | None = None
    # Decided once by the Supervisor from the request, then handed to the
    # finder (how many places to look for) and the planner (how to fill a day).
    plan_shape: dict[str, Any] | None = None
    candidates: dict[str, Candidate] = field(default_factory=dict)
    selected_hotel_id: str | None = None

    itinerary: dict[str, Any] | None = None
    clarification_question: str | None = None

    finder_rounds: int = 0
    validation_count: int = 0
    history: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- helpers
    @property
    def destination(self) -> str:
        return ((self.profile or {}).get("destination") or "").strip()

    @property
    def shape(self) -> dict[str, Any]:
        """The plan shape, derived on demand if the Supervisor has not set it."""
        if self.plan_shape is None:
            self.plan_shape = normalize_plan_shape(None, self.profile)
        return self.plan_shape

    @property
    def required_needs(self) -> list[str]:
        needs = (self.profile or {}).get("accessibility_needs") or []
        return [n for n in needs if isinstance(n, str)]

    def unvalidated(self) -> list[Candidate]:
        return [c for c in self.candidates.values() if c.verdict is None]

    def by_verdict(self, verdict: str) -> list[Candidate]:
        return [c for c in self.candidates.values() if c.verdict == verdict]

    def add_candidate(self, candidate: Candidate) -> None:
        existing = self.candidates.get(candidate.place_id)
        if existing is None:
            self.candidates[candidate.place_id] = candidate
        else:
            # Keep any verdict we already paid an LLM call to produce.
            existing.brief = candidate.brief

    def log(self, message: str) -> None:
        self.history.append(message)

    # --------------------------------------------------- supervisor's view
    def summary_for_supervisor(self, trace: RunTrace) -> dict[str, Any]:
        counts = {v: len(self.by_verdict(v)) for v in VERDICTS}
        budget = trace.budget
        profile = self.profile or {}
        return {
            "request": truncate(self.request, 400),
            "turn": trace.supervisor_turns + 1,
            "profile_ready": self.profile is not None,
            "profile_digest": {
                "destination": profile.get("destination"),
                "trip_days": profile.get("trip_days"),
                "party_size": profile.get("party_size"),
                "pace": profile.get("pace"),
                "max_activities_per_day": profile.get("max_activities_per_day"),
                "needs_hotel": profile.get("needs_hotel"),
                "accessibility_needs": profile.get("accessibility_needs", []),
            }
            if self.profile
            else None,
            "candidates": [
                {
                    "id": c.place_id,
                    "name": c.name,
                    "kind": c.kind,
                    "verdict": c.verdict or "not_yet_checked",
                }
                for c in list(self.candidates.values())[:20]
            ],
            "plan_shape": self.plan_shape,
            "verdict_counts": counts,
            "unvalidated_count": len(self.unvalidated()),
            "hotel_selected": self.selected_hotel_id,
            "itinerary_ready": self.itinerary is not None,
            "finder_rounds_used": self.finder_rounds,
            "recent_events": self.history[-4:],
            "budget_left": {
                "supervisor_turns": max(
                    0, budget.max_supervisor_turns - trace.supervisor_turns
                ),
                "llm_calls": max(0, budget.max_total_llm_calls - trace.llm_calls),
                "seconds": int(trace.remaining_s()),
            },
        }
