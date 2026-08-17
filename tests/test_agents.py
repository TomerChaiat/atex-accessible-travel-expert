"""Unit tests for the pieces that must be correct regardless of the model.

The recurring theme: an LLM may suggest anything, but code decides what is
allowed to reach the traveller.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atex.agents.accessibility_validator import (  # noqa: E402
    _candidate_excerpt,
    _mentions_candidate,
    _sanitize,
)
from atex.agents.activity_finder import _wants_restaurants  # noqa: E402
from atex.agents.supervisor import _legalize  # noqa: E402
from atex.agents.schedule_planner import _enforce_verdicts  # noqa: E402
from atex.agents.user_profile import normalize_profile  # noqa: E402
from atex.config import load_settings  # noqa: E402
from atex.graph import run_agent  # noqa: E402
from atex.httpjson import extract_json_object  # noqa: E402
from atex.repository import (  # noqa: E402
    GooglePlacesRepository,
    LocalRepository,
    travel_estimate,
)
from atex.state import Candidate, RunState  # noqa: E402
from atex.tools import ToolError, build_toolset  # noqa: E402
from atex.util import haversine_km  # noqa: E402
from atex.vectorstore import Match, _public_metadata  # noqa: E402


class TestProfileNormalisation(unittest.TestCase):
    def test_wheelchair_implies_step_free_and_toilet(self):
        profile = normalize_profile({"mobility": {"wheelchair": "manual"}})
        self.assertIn("step_free_entrance", profile["accessibility_needs"])
        self.assertIn("accessible_toilet", profile["accessibility_needs"])
        self.assertTrue(profile["mobility"]["step_free_required"])

    def test_sensory_needs_imply_quiet_space(self):
        profile = normalize_profile({"sensory": {"autism_friendly": True}})
        self.assertIn("quiet_space", profile["accessibility_needs"])

    def test_relaxed_pace_defaults_to_two_activities(self):
        self.assertEqual(normalize_profile({"pace": "relaxed"})["max_activities_per_day"], 2)

    def test_invented_needs_are_discarded(self):
        profile = normalize_profile({"accessibility_needs": ["teleportation", "lift_access"]})
        self.assertEqual(profile["accessibility_needs"], ["lift_access"])

    def test_absurd_values_are_clamped(self):
        profile = normalize_profile({"trip_days": 400, "max_activities_per_day": 99})
        self.assertLessEqual(profile["trip_days"], 7)
        self.assertLessEqual(profile["max_activities_per_day"], 5)

    def test_garbage_input_still_yields_a_usable_profile(self):
        profile = normalize_profile({"trip_days": "banana", "interests": None})
        self.assertIsInstance(profile["trip_days"], int)
        self.assertEqual(profile["interests"], [])


class TestVerdictSanitising(unittest.TestCase):
    def test_supported_without_evidence_is_downgraded(self):
        result = _sanitize({"verdict": "supported", "evidence_ids": []}, set())
        self.assertEqual(result["verdict"], "unknown")
        self.assertNotIn("confidence", result)


class TestValidatorEvidencePrivacy(unittest.TestCase):
    def test_internal_metadata_is_never_exposed(self):
        public = _public_metadata({
            "city": "Amsterdam",
            "location_confidence": "high",
            "entity_confidence": "medium",
            "confidence": 0.9,
            "classification_version": "atex-enrichment-v1",
        })
        self.assertEqual(public, {"city": "Amsterdam"})

    def test_semantic_fallback_requires_the_candidate_name(self):
        candidate = Candidate(
            "ams-nemo-science-museum",
            "NEMO Science Museum",
            "activity",
            {},
        )
        relevant = Match("1", 0.9, "NEMO Science Museum has lift access.", {})
        unrelated = Match("2", 0.95, "Amsterdam has many accessible museums.", {})
        self.assertTrue(_mentions_candidate(relevant, candidate))
        self.assertFalse(_mentions_candidate(unrelated, candidate))

    def test_evidence_excerpt_keeps_the_candidate_name_visible(self):
        candidate = Candidate(
            "ams-nemo-science-museum",
            "NEMO Science Museum",
            "activity",
            {},
        )
        text = "unrelated introduction " * 80 + "NEMO Science Museum has lift access."
        self.assertIn("NEMO Science Museum", _candidate_excerpt(text, candidate))

    def test_fabricated_evidence_ids_are_dropped(self):
        result = _sanitize(
            {"verdict": "supported", "evidence_ids": ["real-1", "invented-9"]},
            {"real-1"},
        )
        self.assertEqual(result["evidence_ids"], ["real-1"])
        self.assertEqual(result["verdict"], "supported")

    def test_unrecognised_verdict_becomes_unknown(self):
        self.assertEqual(
            _sanitize({"verdict": "probably fine", "evidence_ids": ["a"]}, {"a"})["verdict"],
            "unknown",
        )

    def test_confidence_is_discarded(self):
        result = _sanitize(
            {"verdict": "flagged", "confidence": 42, "evidence_ids": ["a"]}, {"a"}
        )
        self.assertNotIn("confidence", result)

    def test_partial_cited_evidence_becomes_a_concern(self):
        result = _sanitize(
            {
                "verdict": "unknown",
                "met_needs": ["quiet_space"],
                "evidence_ids": ["e1"],
            },
            {"e1"},
            ["step_free_entrance", "accessible_toilet", "quiet_space"],
        )
        self.assertEqual(result["verdict"], "flagged")

    def test_supported_requires_all_core_mobility_needs(self):
        result = _sanitize(
            {
                "verdict": "supported",
                "met_needs": ["step_free_entrance"],
                "evidence_ids": ["e1"],
            },
            {"e1"},
            ["step_free_entrance", "accessible_toilet"],
        )
        self.assertEqual(result["verdict"], "flagged")

    def test_missing_sensory_preference_does_not_erase_core_verification(self):
        result = _sanitize(
            {
                "verdict": "supported",
                "met_needs": [
                    "step_free_entrance",
                    "accessible_toilet",
                    "lift_access",
                ],
                "concerns": ["Quiet space is not addressed."],
                "evidence_ids": ["e1"],
            },
            {"e1"},
            [
                "step_free_entrance",
                "accessible_toilet",
                "lift_access",
                "quiet_space",
            ],
        )
        self.assertEqual(result["verdict"], "supported")

    def test_reported_barrier_downgrades_supported_to_concerns(self):
        result = _sanitize(
            {
                "verdict": "supported",
                "met_needs": ["step_free_entrance", "accessible_toilet"],
                "summary": "A small section is not accessible because of a few steps.",
                "evidence_ids": ["e1"],
            },
            {"e1"},
            ["step_free_entrance", "accessible_toilet"],
        )
        self.assertEqual(result["verdict"], "flagged")


class TestCandidateIntent(unittest.TestCase):
    def test_restaurants_are_not_requested_implicitly(self):
        state = RunState("Two days in Berlin with quiet parks and museums")
        state.profile = {"interests": ["history", "park"]}
        self.assertFalse(_wants_restaurants(state))

    def test_explicit_food_interest_requests_restaurants(self):
        state = RunState("I want local food and a quiet cafe")
        state.profile = {"interests": ["history"]}
        self.assertTrue(_wants_restaurants(state))


class TestGooglePlacesRepository(unittest.TestCase):
    def test_google_key_selects_live_repository_backend(self):
        with patch.dict(
            "os.environ",
            {"GOOGLE_MAPS_API_KEY": "configured-key"},
            clear=True,
        ):
            configured = load_settings()

        self.assertEqual(configured.repository_backend, "google_places")
        self.assertEqual(configured.google_maps_api_key, "configured-key")

    def test_live_search_maps_google_accessibility_fields(self):
        payload = {
            "places": [{
                "id": "ChIJ-test",
                "displayName": {"text": "Colosseum"},
                "formattedAddress": "Piazza del Colosseo, Rome",
                "location": {"latitude": 41.8902, "longitude": 12.4922},
                "types": ["historical_landmark", "tourist_attraction"],
                "businessStatus": "OPERATIONAL",
                "accessibilityOptions": {
                    "wheelchairAccessibleEntrance": True,
                    "wheelchairAccessibleRestroom": False,
                },
                "googleMapsUri": "https://maps.google.com/?cid=test",
            }]
        }
        repo = GooglePlacesRepository(
            SimpleNamespace(google_maps_api_key="test-google-key")
        )
        with patch("atex.repository.post_json", return_value=payload) as request:
            places = repo.search_places(
                "Rome", "activity", ["history"], ["step_free_entrance"], 6
            )

        self.assertEqual(len(places), 1)
        place = places[0]
        self.assertEqual(place.id, "gmp:ChIJ-test")
        self.assertEqual(place.name, "Colosseum")
        self.assertEqual(place.city, "Rome")
        self.assertEqual(place.accessibility_claims["step_free_entrance"], "yes")
        self.assertEqual(place.accessibility_claims["accessible_toilet"], "no")
        self.assertEqual(place.accessibility_claims["accessible_parking"], "unknown")
        self.assertIs(repo.get_place(place.id), place)

        _, kwargs = request.call_args
        self.assertEqual(kwargs["headers"]["X-Goog-Api-Key"], "test-google-key")
        self.assertIn("places.accessibilityOptions", kwargs["headers"]["X-Goog-FieldMask"])

    def test_permanently_closed_places_are_excluded(self):
        payload = {
            "places": [{
                "id": "closed",
                "displayName": {"text": "Closed Museum"},
                "businessStatus": "CLOSED_PERMANENTLY",
            }]
        }
        repo = GooglePlacesRepository(SimpleNamespace(google_maps_api_key="key"))
        with patch("atex.repository.post_json", return_value=payload):
            self.assertEqual(repo.search_places("Rome", "activity"), [])

    def test_empty_live_discovery_routes_to_planner_at_search_limit(self):
        state = RunState(request="Four days in Rome")
        state.profile = {"destination": "Rome", "trip_days": 4}
        state.finder_rounds = 2

        actual, corrected = _legalize(state, "ActivityLogisticsFinder")

        self.assertEqual(actual, "SchedulePlanner")
        self.assertEqual(corrected, "ActivityLogisticsFinder")


class TestPlannerCannotUpgradeVerdicts(unittest.TestCase):
    def _state(self):
        state = RunState(request="x")
        state.candidates["p1"] = Candidate(
            place_id="p1", name="Unknown Place", kind="activity", brief={},
            verdict="unknown", verdict_detail={"summary": "No evidence."},
        )
        state.candidates["p2"] = Candidate(
            place_id="p2", name="Flagged Place", kind="activity", brief={},
            verdict="flagged", verdict_detail={"summary": "Steep ramp.", "concerns": ["steps"]},
        )
        return state

    def test_claimed_accessibility_is_overwritten_from_the_verdict(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "p1", "name": "Unknown Place", "accessibility": "supported"},
                {"place_id": "p2", "name": "Flagged Place", "accessibility": "supported"},
            ]}],
        })
        labels = [i["accessibility"] for i in itinerary["days"][0]["items"]]
        self.assertEqual(labels, ["unknown", "flagged"])

    def test_non_supported_places_are_added_to_things_to_confirm(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "p1", "name": "Unknown Place", "accessibility": "supported"},
            ]}],
        })
        self.assertTrue(itinerary["things_to_confirm"])
        self.assertIn("UNVERIFIED", " ".join(itinerary["things_to_confirm"]))

    def test_a_place_scheduled_twice_is_only_flagged_once(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [
                {"day": 1, "items": [{"place_id": "p1", "name": "Unknown Place"}]},
                {"day": 2, "items": [{"place_id": "p1", "name": "Unknown Place"}]},
            ],
        })
        self.assertEqual(len(itinerary["things_to_confirm"]), 1)

    def test_invented_place_cannot_be_labelled_verified(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "hallucinated", "name": "Made Up Cafe", "accessibility": "supported"},
            ]}],
        })
        self.assertEqual(itinerary["days"][0]["items"][0]["accessibility"], "unknown")

    def test_generic_breaks_are_not_labelled(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "rest-break", "name": "Rest", "kind": "rest"},
                {"place_id": "meal-break", "name": "Lunch", "kind": "meal"},
            ]}],
        })
        self.assertEqual(
            [i["accessibility"] for i in itinerary["days"][0]["items"]], ["n/a", "n/a"]
        )

    def test_generic_rows_cannot_borrow_candidate_verdicts(self):
        state = self._state()
        state.candidates["hotel-1"] = Candidate(
            "hotel-1", "Named Hotel", "hotel", {}, verdict="unknown"
        )
        state.candidates["restaurant-1"] = Candidate(
            "restaurant-1", "Named Restaurant", "restaurant", {}, verdict="flagged"
        )
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "p1", "name": "Lunch break", "kind": "meal"},
                {"place_id": "hotel-1", "name": "Hotel rest", "kind": "rest"},
                {"place_id": "restaurant-1", "name": "Named Restaurant", "kind": "meal"},
            ]}],
            "things_to_confirm": [
                "Lunch break: verify access.",
                "Hotel rest: verify access.",
            ],
        })
        items = itinerary["days"][0]["items"]
        self.assertEqual(
            [item["accessibility"] for item in items],
            ["n/a", "n/a", "flagged"],
        )
        self.assertEqual(items[0]["place_id"], "meal-break")
        self.assertEqual(items[1]["place_id"], "rest-break")
        self.assertNotIn("Lunch break", " ".join(itinerary["things_to_confirm"]))
        self.assertNotIn("Hotel rest", " ".join(itinerary["things_to_confirm"]))


class TestTools(unittest.TestCase):
    def setUp(self):
        self.repo = LocalRepository()
        self.tools = build_toolset(self.repo, ["step_free_entrance"], 6)

    def test_search_returns_briefs_not_full_records(self):
        result = self.tools["search_activities"]({"city": "Amsterdam", "limit": 3})
        self.assertLessEqual(len(result["results"]), 3)
        for row in result["results"]:
            self.assertIn("id", row)
            self.assertNotIn("lat", row, "briefs must stay small for prompt efficiency")

    def test_unknown_city_explains_itself(self):
        result = self.tools["search_activities"]({"city": "Atlantis"})
        self.assertEqual(result["results"], [])
        self.assertIn("Amsterdam", result["note"])

    def test_missing_city_raises(self):
        with self.assertRaises(ToolError):
            self.tools["search_activities"]({})

    def test_unknown_place_id_raises(self):
        with self.assertRaises(ToolError):
            self.tools["get_place_details"]({"place_id": "nope"})

    def test_limit_is_capped(self):
        result = self.tools["search_activities"]({"city": "Amsterdam", "limit": 9999})
        self.assertLessEqual(len(result["results"]), 8)

    def test_places_explicitly_marked_no_rank_below_unknown(self):
        """An 'unknown' is not a negative; only an explicit 'no' should sink.

        Asserted against the repository rather than the tool, because the tool
        caps its result count and would truncate the tail we care about.
        """
        ranked = self.repo.search_places(
            city="Amsterdam", kind="activity", needs=["step_free_entrance"], limit=50
        )
        ids = [p.id for p in ranked]

        # Anne Frank House is the only Amsterdam entry with an explicit "no".
        self.assertEqual(ids[-1], "ams-anne-frank-house")

        # Places whose claims are merely unknown must not be pushed below it.
        unknown_claim = [
            p.id for p in ranked
            if p.accessibility_claims.get("step_free_entrance") == "unknown"
        ]
        for place_id in unknown_claim:
            self.assertLess(ids.index(place_id), ids.index("ams-anne-frank-house"))

    def test_tool_result_count_is_capped(self):
        results = self.tools["search_activities"]({"city": "Amsterdam", "limit": 8})["results"]
        self.assertLessEqual(len(results), 6)  # max_candidates_per_search
        self.assertNotIn(
            "ams-anne-frank-house",
            [r["id"] for r in results],
            "a place with an explicit 'no' should fall outside the capped shortlist",
        )


class TestTravelEstimates(unittest.TestCase):
    def test_haversine_against_a_known_distance(self):
        # Amsterdam centre to Berlin centre is ~577km.
        km = haversine_km(52.3676, 4.9041, 52.5200, 13.4050)
        self.assertAlmostEqual(km, 577, delta=15)

    def test_identical_points_are_zero(self):
        self.assertEqual(haversine_km(52.0, 4.0, 52.0, 4.0), 0.0)

    def test_walking_is_slower_than_transit(self):
        repo = LocalRepository()
        a = repo.get_place("ams-rijksmuseum")
        b = repo.get_place("ams-nemo-science-museum")
        walk = travel_estimate(a, b, "wheelchair_walk")
        transit = travel_estimate(a, b, "accessible_transit")
        self.assertGreater(walk["duration_min"], transit["duration_min"])

    def test_estimate_is_labelled_as_an_estimate(self):
        repo = LocalRepository()
        estimate = travel_estimate(
            repo.get_place("ams-rijksmuseum"), repo.get_place("ams-vondelpark")
        )
        self.assertIn("estimate", estimate["basis"])


class TestJsonExtraction(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced_block(self):
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_object_wrapped_in_prose(self):
        self.assertEqual(extract_json_object('Sure!\n{"a": 1}\nHope that helps.'), {"a": 1})

    def test_concatenated_objects_returns_first_action(self):
        response = (
            '{"thought":"search first","action":{"tool":"search_hotels","args":{}}}\n'
            '{"thought":"future turn","action":{"tool":"search_activities","args":{}}}'
        )
        self.assertEqual(
            extract_json_object(response),
            {"thought": "search first", "action": {"tool": "search_hotels", "args": {}}},
        )

    def test_unparseable_raises(self):
        with self.assertRaises(ValueError):
            extract_json_object("no json at all")


class TestRetrievalRecall(unittest.TestCase):
    """A place with evidence in the KB must never be reported as unverified.

    Regression test: an unfiltered top-k search used to crowd out a venue's own
    passages, producing a false 'unknown' -- the one mistake this system exists
    to avoid.
    """

    def test_every_place_with_a_chunk_retrieves_it(self):
        import json

        from atex.agents.accessibility_validator import retrieve_evidence
        from atex.config import KB_DIR, settings
        from atex.context import AgentContext
        from atex.tracing import RunTrace

        wanted: dict[str, str] = {}
        for path in KB_DIR.glob("*.json"):
            for chunk in json.loads(path.read_text(encoding="utf-8")).get("chunks", []):
                if chunk.get("place_id"):
                    wanted.setdefault(chunk["place_id"], chunk.get("city") or "")

        self.assertTrue(wanted, "no place-specific chunks found in data/kb/")

        cfg = settings()
        ctx = AgentContext.build(RunTrace(budget=cfg.budget), cfg)
        repo = LocalRepository()

        for place_id, city in wanted.items():
            with self.subTest(place=place_id):
                place = repo.get_place(place_id)
                self.assertIsNotNone(place, f"KB references unknown place {place_id}")
                candidate = Candidate(place_id, place.name, place.kind, place.to_brief())
                specific, _ = retrieve_evidence(
                    ctx, candidate, city, ["step_free_entrance", "accessible_toilet"]
                )
                self.assertTrue(specific, f"no evidence retrieved for {place_id}")

    def test_general_city_notes_never_pose_as_place_evidence(self):
        from atex.agents.accessibility_validator import retrieve_evidence
        from atex.config import settings
        from atex.context import AgentContext
        from atex.tracing import RunTrace

        cfg = settings()
        ctx = AgentContext.build(RunTrace(budget=cfg.budget), cfg)
        repo = LocalRepository()

        # This place deliberately has no chunk of its own.
        place = repo.get_place("ams-cafe-centrum")
        candidate = Candidate(place.id, place.name, place.kind, place.to_brief())
        specific, general = retrieve_evidence(ctx, candidate, "Amsterdam", ["step_free_entrance"])

        self.assertEqual(specific, [], "city notes must not count as venue evidence")
        self.assertTrue(general, "city-scope notes should still be retrievable")


class TestFollowUpTurns(unittest.TestCase):
    def test_second_turn_reuses_verdicts_and_costs_less(self):
        first = run_agent("Three days in Amsterdam, manual wheelchair, relaxed pace, need a hotel.")
        saved = first.session_state()
        saved["turn_index"] = first.state.turn_index

        second = run_agent("Swap one day for something quieter.", session_id="s1", saved_state=saved)

        self.assertIsNone(second.error)
        self.assertTrue(second.state.candidates)
        self.assertLess(
            second.usage["llm_calls"],
            first.usage["llm_calls"],
            "a follow-up should not re-validate everything from scratch",
        )


if __name__ == "__main__":
    unittest.main()
