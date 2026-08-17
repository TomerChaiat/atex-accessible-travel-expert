"""Offline stand-in for the LLM, one handler per module.

This is not a mock that returns fixed strings. Each handler reads the same
prompt the real model would receive and produces a schema-valid response
derived from it, so the full supervisor loop, ReAct cycle, verdict logic,
guardrails and forced-finalize path all execute for real with no API key.

What it is NOT: a quality substitute. The handlers use keyword heuristics where
the real model reasons. When LLMOD_API_KEY is set, `build_llm_backend` returns
the real client instead and none of this code runs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import (
    ACCESSIBILITY_VALIDATOR,
    ACTIVITY_LOGISTICS_FINDER,
    SCHEDULE_PLANNER,
    SUPERVISOR,
    USER_PROFILE_AGENT,
)
from .llm import LLMResult

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

KNOWN_CITIES = ("amsterdam", "barcelona", "berlin")

INTEREST_KEYWORDS = {
    "museum": ["museum", "museums", "gallery"],
    "art": ["art", "painting", "van gogh", "picasso"],
    "history": ["history", "historic", "heritage", "war"],
    "park": ["park", "green", "garden", "outdoor"],
    "nature": ["nature", "botanical", "beach"],
    "food": ["food", "restaurant", "eat", "dining", "tapas"],
    "architecture": ["architecture", "gaudi", "building"],
    "family": ["family", "kids", "children"],
    "relaxed": ["relaxed", "slow", "calm", "easy"],
}

NEGATIVE_EVIDENCE = (
    "not suitable", "no lift", "not wheelchair accessible", "stairs-only",
    "steep", "too narrow", "no public accessible toilets", "no toilets",
    "not accessible", "is not wheelchair",
)
CONDITIONAL_EVIDENCE = (
    "varies", "seasonal", "only during", "on request", "must be arranged",
    "raised lip", "out of service", "irregular", "cannot be relied",
    "contact the specific", "registration in advance", "tight", "indirect",
    "hard work", "sometimes closed", "disorienting",
)
POSITIVE_EVIDENCE = (
    "step free", "step-free", "level", "lift", "accessible toilet",
    "roll-in shower", "grab rail", "ramp", "wide doorways", "flat",
)


def _int_in(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    token = match.group(1).lower()
    if token.isdigit():
        return int(token)
    return WORD_NUMBERS.get(token)


class FakeLLMBackend:
    name = "fake"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        module: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        json_object: bool = True,
        timeout: float = 60.0,
    ) -> LLMResult:
        handlers = {
            SUPERVISOR: self._supervisor,
            USER_PROFILE_AGENT: self._profile,
            ACTIVITY_LOGISTICS_FINDER: self._finder,
            ACCESSIBILITY_VALIDATOR: self._validator,
            SCHEDULE_PLANNER: self._planner,
        }
        handler = handlers.get(module)
        payload = handler(user_prompt) if handler else {"error": f"no fake for {module}"}
        return LLMResult(text=json.dumps(payload, ensure_ascii=False))

    # ------------------------------------------------------------ Supervisor
    def _supervisor(self, user_prompt: str) -> dict[str, Any]:
        state = _safe_json(user_prompt)
        budget = state.get("budget_left") or {}
        digest = state.get("profile_digest") or {}

        low_budget = (budget.get("llm_calls") or 99) <= 4 or (budget.get("seconds") or 999) <= 45

        if not state.get("profile_ready"):
            return _decision("UserProfileAgent", "Extract the traveller's profile.",
                             "No profile yet.")
        if not digest.get("destination"):
            return _decision(
                "ASK_USER", "", "No destination named.",
                question="Which city would you like to visit, and for how many days?",
            )
        if state.get("itinerary_ready"):
            return _decision("FINISH", "", "Itinerary is ready.")
        if low_budget and state.get("candidates"):
            return _decision("SchedulePlanner", "Budget is low; plan with what we have.",
                             "Budget nearly spent.")
        if not state.get("candidates"):
            return _decision("ActivityLogisticsFinder", "Find candidate places for this profile.",
                             "No candidates yet.")
        if (state.get("unvalidated_count") or 0) > 0:
            return _decision("AccessibilityValidator", "Check the unvalidated candidates.",
                             "Candidates need verdicts.")
        return _decision("SchedulePlanner", "Build the day-by-day itinerary.",
                         "All candidates have verdicts.")

    # ------------------------------------------------------ UserProfileAgent
    def _profile(self, user_prompt: str) -> dict[str, Any]:
        maybe = _safe_json(user_prompt)
        text = maybe.get("new_request") if maybe else None
        prior = maybe.get("previous_profile") if maybe else None
        if not text:
            text = user_prompt
        lowered = text.lower()

        destination = None
        for city in KNOWN_CITIES:
            if city in lowered:
                destination = city.capitalize()
                break
        if destination is None:
            match = re.search(r"\b(?:in|to|visiting|visit)\s+([A-Z][a-zA-Z]+)", text)
            if match:
                destination = match.group(1)

        days = _int_in(lowered, r"(\d+|one|two|three|four|five|six|seven)[\s-]*day")
        party = _int_in(lowered, r"(?:family|group|party)\s+of\s+(\d+|one|two|three|four|five|six)")
        per_day = _int_in(
            lowered, r"(?:no more than|at most|max(?:imum)?(?: of)?)\s+(\d+|one|two|three|four)"
        )

        wheelchair = "none"
        if re.search(r"powered|electric|motori[sz]ed", lowered):
            wheelchair = "powered"
        elif "scooter" in lowered:
            wheelchair = "scooter"
        elif "wheelchair" in lowered:
            wheelchair = "manual"

        autism = any(w in lowered for w in ("autism", "autistic", "sensory"))
        low_noise = autism or any(w in lowered for w in ("quiet", "noise", "loud"))

        pace = None
        if any(w in lowered for w in ("relaxed", "slow", "calm", "easy pace")):
            pace = "relaxed"
        elif any(w in lowered for w in ("packed", "intense", "as much as possible")):
            pace = "packed"

        budget_level = None
        for level, words in (
            ("low", ("budget", "cheap", "affordable")),
            ("high", ("luxury", "high-end", "upscale")),
        ):
            if any(w in lowered for w in words):
                budget_level = level
                break

        interests = [
            tag for tag, words in INTEREST_KEYWORDS.items() if any(w in lowered for w in words)
        ]

        needs = []
        if wheelchair != "none":
            needs = ["step_free_entrance", "accessible_toilet", "lift_access"]
        if low_noise:
            needs.append("quiet_space")

        missing = []
        if days is None:
            missing.append("trip length")
        if not interests:
            missing.append("preferred kinds of activity")

        profile = {
            "destination": destination or (prior or {}).get("destination"),
            "country": None,
            "trip_days": days or (prior or {}).get("trip_days"),
            "party_size": party or (prior or {}).get("party_size"),
            "mobility": {
                "wheelchair": wheelchair,
                "step_free_required": wheelchair != "none",
                "assistant_present": None,
            },
            "sensory": {
                "low_noise": low_noise,
                "low_crowd": autism,
                "autism_friendly": autism,
            },
            "pace": pace,
            "max_activities_per_day": per_day,
            "budget_level": budget_level,
            "interests": interests,
            "needs_hotel": "hotel" in lowered or "stay" in lowered,
            "accessibility_needs": needs,
            "missing_info": missing,
            "notes": "Extracted by the offline fake backend using keyword heuristics.",
        }
        return profile

    # ---------------------------------------------- ActivityLogisticsFinder
    def _finder(self, user_prompt: str) -> dict[str, Any]:
        payload = _safe_json(user_prompt)
        profile = payload.get("profile") or {}
        observations = payload.get("observations") or []
        turns_left = int(payload.get("turns_left") or 1)
        city = profile.get("destination") or ""

        searched = {obs.get("tool") for obs in observations}

        if "search_activities" not in searched:
            return {
                "thought": f"Look for activities in {city} matching the traveller's interests.",
                "action": {
                    "tool": "search_activities",
                    "args": {
                        "city": city,
                        "categories": profile.get("interests") or [],
                        "limit": 6,
                    },
                },
            }

        if profile.get("needs_hotel") and "search_hotels" not in searched and turns_left > 1:
            return {
                "thought": "The traveller needs a hotel, so search accommodation next.",
                "action": {"tool": "search_hotels", "args": {"city": city, "limit": 3}},
            }

        if "search_restaurants" not in searched and turns_left > 1:
            return {
                "thought": "Add a couple of meal options near the selected areas.",
                "action": {"tool": "search_restaurants", "args": {"city": city, "limit": 2}},
            }

        activities, hotels, restaurants = _partition_observations(observations)
        wanted = max(2, (profile.get("trip_days") or 3) * (profile.get("max_activities_per_day") or 2))
        return {
            "thought": "Enough candidates gathered; hand them to the validator.",
            "action": {
                "tool": "finish",
                "args": {
                    "selected_activity_ids": activities[: wanted + 2],
                    "selected_hotel_id": hotels[0] if (hotels and profile.get("needs_hotel")) else None,
                    "selected_restaurant_ids": restaurants[:2],
                },
            },
        }

    # ----------------------------------------------- AccessibilityValidator
    def _validator(self, user_prompt: str) -> dict[str, Any]:
        payload = _safe_json(user_prompt)
        needs = payload.get("required_needs") or []
        places = payload.get("places") or []
        return {"verdicts": [self._one_verdict(item, needs) for item in places]}

    def _one_verdict(self, item: dict[str, Any], needs: list[str]) -> dict[str, Any]:
        place = item.get("place") or {}
        place_id = place.get("id") or ""
        evidence = item.get("evidence") or []

        if not evidence:
            return _unknown_verdict(place_id, "No evidence passages were supplied.")

        blob = " ".join(str(e.get("text", "")).lower() for e in evidence)
        ids = [e.get("id") for e in evidence if e.get("id")]

        negatives = [p for p in NEGATIVE_EVIDENCE if p in blob]
        conditionals = [p for p in CONDITIONAL_EVIDENCE if p in blob]
        positives = [p for p in POSITIVE_EVIDENCE if p in blob]

        if negatives:
            verdict, confidence = "flagged", 0.75
            concerns = [f"Evidence mentions: {p}" for p in negatives[:3]]
            summary = "The evidence describes barriers that conflict with the stated needs."
        elif conditionals:
            verdict, confidence = "flagged", 0.6
            concerns = [f"Conditional: {p}" for p in conditionals[:3]]
            summary = "Access appears conditional, seasonal, or dependent on arranging ahead."
        elif positives:
            verdict, confidence = "supported", 0.7
            concerns = []
            summary = "The evidence directly describes step-free access and accessible facilities."
        else:
            return _unknown_verdict(
                place_id, "Retrieved passages do not address the required needs."
            )

        return {
            "place_id": place_id,
            "verdict": verdict,
            "confidence": confidence,
            "met_needs": needs if verdict == "supported" else [],
            "unmet_needs": needs[:1] if (verdict == "flagged" and negatives) else [],
            "concerns": concerns,
            "evidence_ids": ids,
            "summary": summary,
        }

    # ---------------------------------------------------- SchedulePlanner
    def _planner(self, user_prompt: str) -> dict[str, Any]:
        payload = _safe_json(user_prompt)
        profile = payload.get("profile") or {}
        candidates = payload.get("candidates") or []

        trip_days = max(1, int(profile.get("trip_days") or 2))
        per_day = max(1, int(profile.get("max_activities_per_day") or 2))

        activities = [c for c in candidates if c.get("kind") == "activity"]
        meals = [c for c in candidates if c.get("kind") == "restaurant"]

        days, used = [], set()
        cursor = 0
        for day_number in range(1, trip_days + 1):
            items, minutes = [], 9 * 60 + 30
            placed = 0

            while placed < per_day and cursor < len(activities):
                activity = activities[cursor]
                cursor += 1
                used.add(activity["place_id"])
                items.append({
                    "time": _clock(minutes),
                    "place_id": activity["place_id"],
                    "name": activity.get("name", activity["place_id"]),
                    "kind": "activity",
                    "duration_min": int(activity.get("duration_min") or 90),
                    "accessibility": activity.get("accessibility", "unknown"),
                    "note": _short_note(activity),
                })
                minutes += int(activity.get("duration_min") or 90) + 30
                placed += 1

                if placed == 1 and per_day > 1:
                    meal = meals[(day_number - 1) % len(meals)] if meals else None
                    items.append({
                        "time": _clock(max(minutes, 12 * 60 + 30)),
                        "place_id": meal["place_id"] if meal else "meal-break",
                        "name": meal.get("name") if meal else "Lunch break",
                        "kind": "meal",
                        "duration_min": 60,
                        "accessibility": meal.get("accessibility", "unknown") if meal else "n/a",
                        "note": "Unhurried stop before the afternoon.",
                    })
                    if meal:
                        used.add(meal["place_id"])
                    minutes = max(minutes, 12 * 60 + 30) + 60

            if items:
                items.append({
                    "time": _clock(minutes),
                    "place_id": "rest-break",
                    "name": "Rest and recharge",
                    "kind": "rest",
                    "duration_min": 45,
                    "accessibility": "n/a",
                    "note": "Deliberate downtime before the evening.",
                })

            days.append({
                "day": day_number,
                "theme": _theme(items),
                "items": items,
                "day_note": "Paced for a relaxed day with slack between stops."
                if per_day <= 2
                else "A fuller day; leave extra time between stops.",
            })

        not_scheduled = [
            {
                "place_id": c["place_id"],
                "name": c.get("name", c["place_id"]),
                "reason": "Kept as a spare; the daily activity limit was already met.",
            }
            for c in candidates
            if c["place_id"] not in used and c.get("kind") != "hotel"
        ]

        destination = profile.get("destination") or "your destination"
        return {
            "summary": (
                f"A {trip_days}-day plan for {destination} at {per_day} "
                f"{'activity' if per_day == 1 else 'activities'} per day, ordered so each day "
                "stays in one part of the city, with a rest built into every day."
            ),
            "days": days,
            "not_scheduled": not_scheduled[:6],
            "warnings": [
                "Travel times are straight-line estimates, not routed journeys.",
            ],
            "things_to_confirm": [],
        }


# --------------------------------------------------------------- utilities
def _safe_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _unknown_verdict(place_id: str, summary: str) -> dict[str, Any]:
    return {
        "place_id": place_id,
        "verdict": "unknown",
        "confidence": 0.0,
        "met_needs": [],
        "unmet_needs": [],
        "concerns": [],
        "evidence_ids": [],
        "summary": summary,
    }


def _decision(module: str, instruction: str, reasoning: str, question: str | None = None):
    return {
        "reasoning": reasoning,
        "next_module": module,
        "instruction": instruction,
        "clarification_question": question,
    }


def _partition_observations(observations: list[dict]) -> tuple[list[str], list[str], list[str]]:
    activities, hotels, restaurants = [], [], []
    for obs in observations:
        result = obs.get("result") or {}
        for row in result.get("results") or []:
            bucket = {
                "hotel": hotels,
                "restaurant": restaurants,
            }.get(row.get("kind"), activities)
            if row.get("id") and row["id"] not in bucket:
                bucket.append(row["id"])
    return activities, hotels, restaurants


def _clock(minutes: int) -> str:
    minutes = min(minutes, 22 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _short_note(candidate: dict[str, Any]) -> str:
    verdict = candidate.get("accessibility")
    if verdict == "unknown":
        return "Accessibility not verified - confirm before you go."
    if verdict == "flagged":
        concerns = candidate.get("concerns") or []
        return f"Check first: {concerns[0]}" if concerns else "Accessibility concerns found."
    return candidate.get("accessibility_summary") or "Verified against the knowledge base."


def _theme(items: list[dict[str, Any]]) -> str:
    names = [i["name"] for i in items if i.get("kind") == "activity"]
    if not names:
        return "Rest day"
    return names[0] if len(names) == 1 else f"{names[0]} and nearby"
