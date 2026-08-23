"""Unit tests for the pieces that must be correct regardless of the model.

The recurring theme: an LLM may suggest anything, but code decides what is
allowed to reach the traveller.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The place catalogue and knowledge base under tests/fixtures/ exist only for
# these tests. They are deliberately not shipped in data/: the product does not
# invent accessibility content. Pointing the keyless backends here keeps the
# suite deterministic with no API keys and no network. Set before importing
# atex, because settings read the environment at call time.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_SEED = FIXTURES / "seed"
FIXTURE_KB = FIXTURES / "kb"
os.environ["ATEX_SEED_DIR"] = str(FIXTURE_SEED)
os.environ["ATEX_KB_DIR"] = str(FIXTURE_KB)

from atex.agents.accessibility_validator import (  # noqa: E402
    _candidate_excerpt,
    _mentions_candidate,
    _sanitize,
    retrieve_evidence,
    validate_batch,
)
from atex.agents.activity_finder import (  # noqa: E402
    _candidate_memory,
    _finish_selection,
    _wants_restaurants,
    activities_needed,
)
from atex.agents.supervisor import MAX_FINDER_ROUNDS, _legalize, decide  # noqa: E402
from atex.agents.schedule_planner import (  # noqa: E402
    MAX_NOT_SCHEDULED,
    _duration,
    _enforce_verdicts,
    _flagged_reason,
    _is_real_activity,
    _parse_time,
)
from atex.agents.user_profile import (  # noqa: E402
    MAX_TRIP_DAYS,
    _apply_profile_change,
    _carry_forward,
    _release_hotel_selection,
    _replacement_hotel_requested,
    _strict_location_requested,
    normalize_profile,
)
from atex.config import load_settings  # noqa: E402
from atex.graph import run_agent  # noqa: E402
from atex.httpjson import HttpError, extract_json_object  # noqa: E402
from atex.repository import (  # noqa: E402
    GooglePlacesRepository,
    LocalRepository,
    RepositoryError,
    travel_estimate,
)
from atex.render import render_itinerary, render_out_of_scope  # noqa: E402
from atex.routing import (  # noqa: E402
    DETOUR_FACTOR,
    GoogleRoutesRouter,
    LocalRouter,
    build_router,
    describe_options,
    drop_unreasonable,
    planning_option,
    self_powered_limit_km,
)
from atex.state import (  # noqa: E402
    MAX_ACTIVITIES_PER_DAY,
    Candidate,
    RunState,
    normalize_plan_shape,
    same_location,
)
from atex.tools import ToolError, build_toolset  # noqa: E402
from atex.tracing import RunTrace  # noqa: E402
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

    def test_an_unstated_activity_limit_is_left_for_the_supervisor(self):
        # Inventing a number here made it indistinguishable from something the
        # traveller actually said, and left the Supervisor nothing to decide.
        profile = normalize_profile({"pace": "relaxed"})
        self.assertIsNone(profile["max_activities_per_day"])
        self.assertEqual(profile["pace"], "relaxed")

    def test_a_stated_activity_limit_is_kept(self):
        self.assertEqual(
            normalize_profile({"max_activities_per_day": 4})["max_activities_per_day"], 4
        )

    def test_invented_needs_are_discarded(self):
        profile = normalize_profile({"accessibility_needs": ["teleportation", "lift_access"]})
        self.assertEqual(profile["accessibility_needs"], ["lift_access"])

    def test_absurd_values_are_clamped(self):
        profile = normalize_profile({"trip_days": 400, "max_activities_per_day": 99})
        self.assertLessEqual(profile["trip_days"], MAX_TRIP_DAYS)
        self.assertLessEqual(profile["max_activities_per_day"], MAX_ACTIVITIES_PER_DAY)

    def test_garbage_input_still_yields_a_usable_profile(self):
        profile = normalize_profile({"trip_days": "banana", "interests": None})
        self.assertIsInstance(profile["trip_days"], int)
        self.assertEqual(profile["interests"], [])

    def test_multiple_requested_locations_and_strict_scope_are_preserved(self):
        profile = normalize_profile(
            {
                "destination": "Haifa",
                "destinations": ["Haifa", "Tel Aviv", "haifa"],
                "requested_locations_only": True,
            }
        )
        self.assertEqual(profile["destinations"], ["Haifa", "Tel Aviv"])
        self.assertTrue(profile["requested_locations_only"])

    def test_explicit_no_day_trip_language_is_a_hard_location_boundary(self):
        self.assertTrue(_strict_location_requested("Stay only in Haifa; no day trips."))
        self.assertFalse(_strict_location_requested("Stay in Haifa and suggest nearby trips."))

    def test_a_wheelchair_implies_limited_walking(self):
        profile = normalize_profile({"mobility": {"wheelchair": "manual"}})
        self.assertTrue(profile["mobility"]["walking_limited"])

    def test_a_stated_walking_limit_survives_without_a_wheelchair(self):
        profile = normalize_profile(
            {"mobility": {"wheelchair": "none", "walking_limited": True}}
        )
        self.assertTrue(profile["mobility"]["walking_limited"])

    def test_no_stated_transport_preference_leaves_the_choice_open(self):
        self.assertIsNone(normalize_profile({})["preferred_transport"])

    def test_an_invented_transport_mode_is_discarded(self):
        profile = normalize_profile({"preferred_transport": "helicopter"})
        self.assertIsNone(profile["preferred_transport"])

    def test_a_real_transport_preference_is_kept(self):
        profile = normalize_profile({"preferred_transport": "accessible_taxi"})
        self.assertEqual(profile["preferred_transport"], "accessible_taxi")

    def test_two_weeks_stays_two_weeks(self):
        # A 7-day ceiling silently turned "two weeks in New York" into a
        # 7-day itinerary. Trip length is the traveller's to decide.
        self.assertEqual(normalize_profile({"trip_days": 14})["trip_days"], 14)

    def test_a_multi_day_trip_assumes_somewhere_to_sleep(self):
        # Silence is not a no: "two weeks in New York" plainly needs a hotel.
        self.assertTrue(normalize_profile({"trip_days": 14})["needs_hotel"])

    def test_an_explicit_no_hotel_is_respected(self):
        profile = normalize_profile({"trip_days": 14, "needs_hotel": False})
        self.assertFalse(profile["needs_hotel"])

    def test_a_single_day_trip_assumes_no_hotel(self):
        self.assertFalse(normalize_profile({"trip_days": 1})["needs_hotel"])

    def test_group_size_is_preserved_for_conditional_access(self):
        profile = normalize_profile({"party_size": 4})
        self.assertEqual(profile["party_size"], 4)


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

    def test_helper_requirement_depends_on_traveller_profile(self):
        verdict = {
            "verdict": "flagged",
            "met_needs": ["step_free_entrance", "accessible_toilet"],
            "summary": "Access is supported when visiting with a helper.",
            "evidence_ids": ["e1"],
        }
        solo = _sanitize(
            verdict,
            {"e1"},
            ["step_free_entrance", "accessible_toilet"],
            {"companion_available": False, "travelling_solo": True},
        )
        group = _sanitize(
            verdict,
            {"e1"},
            ["step_free_entrance", "accessible_toilet"],
            {"companion_available": True, "party_size": 4},
        )
        self.assertEqual(solo["verdict"], "flagged")
        self.assertEqual(group["verdict"], "supported")
        self.assertIn("companion", " ".join(group["conditions"]).lower())

    def test_companion_does_not_erase_an_unconfirmed_toilet(self):
        result = _sanitize(
            {
                "verdict": "flagged",
                "met_needs": ["step_free_entrance"],
                "unmet_needs": ["accessible_toilet"],
                "summary": (
                    "Access requires a companion, and an accessible toilet is not confirmed."
                ),
                "evidence_ids": ["e1"],
            },
            {"e1"},
            ["step_free_entrance", "accessible_toilet"],
            {"companion_available": True, "party_size": 4},
        )
        self.assertEqual(result["verdict"], "flagged")

    def test_multi_city_candidates_retrieve_evidence_in_their_own_city(self):
        state = RunState("Haifa and Tel Aviv")
        state.profile = {"destination": "Haifa"}
        candidates = [
            Candidate("h", "Haifa Place", "activity", {"city": "Haifa"}),
            Candidate("t", "Tel Aviv Place", "activity", {"city": "Tel Aviv"}),
        ]
        ctx = SimpleNamespace(trace=RunTrace(load_settings().budget))
        with patch(
            "atex.agents.accessibility_validator.retrieve_evidence",
            return_value=([], []),
        ) as retrieve:
            validate_batch(ctx, state, candidates)

        self.assertEqual(
            [call.args[2] for call in retrieve.call_args_list],
            ["Haifa", "Tel Aviv"],
        )


class TestCandidateIntent(unittest.TestCase):
    def test_restaurants_are_not_requested_implicitly(self):
        state = RunState("Two days in Berlin with quiet parks and museums")
        state.profile = {"interests": ["history", "park"]}
        self.assertFalse(_wants_restaurants(state))

    def test_explicit_food_interest_requests_restaurants(self):
        state = RunState("I want local food and a quiet cafe")
        state.profile = {"interests": ["history"]}
        self.assertTrue(_wants_restaurants(state))

    def test_finish_rejects_ids_not_returned_by_the_provider(self):
        observations = [
            {
                "tool": "search_activities",
                "result": {"results": [{"id": "gmp:ChIJ-valid"}]},
            }
        ]
        activities, restaurants, hotels = _finish_selection(
            {
                "selected_activity_ids": ["gmp:ChIJ-valid", "gmp:i_modified"],
                "selected_restaurant_ids": ["gmp:invented-restaurant"],
                "selected_hotel_id": "gmp:invented-hotel",
            },
            observations,
        )
        self.assertEqual(activities, ["gmp:ChIJ-valid"])
        self.assertEqual(restaurants, [])
        self.assertEqual(hotels, [])

    def test_finish_accepts_multiple_exact_observed_hotels(self):
        observations = [
            {
                "tool": "search_hotels",
                "args": {"city": "Haifa"},
                "result": {"results": [{"id": "hotel-haifa", "kind": "hotel"}]},
            },
            {
                "tool": "search_hotels",
                "args": {"city": "Tel Aviv"},
                "result": {"results": [{"id": "hotel-tel-aviv", "kind": "hotel"}]},
            },
        ]
        _, _, hotels = _finish_selection(
            {
                "selected_hotels": [
                    {"place_id": "hotel-haifa", "location": "Haifa"},
                    {"place_id": "hotel-tel-aviv", "location": "Tel Aviv"},
                    {"place_id": "invented", "location": "Jerusalem"},
                ]
            },
            observations,
        )
        self.assertEqual(
            hotels,
            [
                {"place_id": "hotel-haifa", "location": "Haifa"},
                {"place_id": "hotel-tel-aviv", "location": "Tel Aviv"},
            ],
        )

    def test_compact_memory_keeps_ids_from_older_city_searches(self):
        observations = [
            {
                "tool": "search_activities",
                "args": {"city": f"City {index}"},
                "result": {
                    "results": [
                        {"id": f"place-{index}", "name": f"Place {index}", "kind": "activity"}
                    ]
                },
            }
            for index in range(6)
        ]
        memory = _candidate_memory(observations)
        self.assertEqual(memory[0]["id"], "place-0")
        self.assertEqual(memory[-1]["city"], "City 5")


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

    def test_invalid_or_obsolete_google_id_is_skipped(self):
        repo = GooglePlacesRepository(SimpleNamespace(google_maps_api_key="key"))
        error = HttpError(400, '{"error":{"status":"INVALID_ARGUMENT"}}', "test-url")
        with patch("atex.repository.get_json", side_effect=error):
            self.assertIsNone(repo.get_place("gmp:i_modified"))

    def test_google_auth_error_still_surfaces(self):
        repo = GooglePlacesRepository(SimpleNamespace(google_maps_api_key="bad-key"))
        error = HttpError(403, '{"error":{"status":"PERMISSION_DENIED"}}', "test-url")
        with patch("atex.repository.get_json", side_effect=error):
            with self.assertRaises(RepositoryError):
                repo.get_place("gmp:ChIJ-valid")

    def test_empty_live_discovery_routes_to_planner_at_search_limit(self):
        state = RunState(request="Four days in Rome")
        state.profile = {"destination": "Rome", "trip_days": 4}
        state.finder_rounds = MAX_FINDER_ROUNDS

        actual, corrected = _legalize(state, "ActivityLogisticsFinder")

        self.assertEqual(actual, "SchedulePlanner")
        self.assertEqual(corrected, "ActivityLogisticsFinder")


class TestPlannerCannotUpgradeVerdicts(unittest.TestCase):
    def _state(self):
        state = RunState(request="x")
        # One stop a day, so the day-filling top-up stays out of tests that are
        # about verdict enforcement rather than how full a day is.
        state.plan_shape = normalize_plan_shape({"activities_per_day": 1}, None)
        state.candidates["p1"] = Candidate(
            place_id="p1", name="Unknown Place", kind="activity", brief={},
            verdict="unknown", verdict_detail={"summary": "No evidence."},
        )
        state.candidates["p2"] = Candidate(
            place_id="p2", name="Flagged Place", kind="activity", brief={},
            verdict="flagged", verdict_detail={"summary": "Steep ramp.", "concerns": ["steps"]},
        )
        return state

    def test_flagged_candidate_is_removed_even_if_planner_scheduled_it(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "p1", "name": "Unknown Place", "accessibility": "supported"},
                {"place_id": "p2", "name": "Flagged Place", "accessibility": "supported"},
            ]}],
        })
        labels = [i["accessibility"] for i in itinerary["days"][0]["items"]]
        self.assertEqual(labels, ["unknown"])
        rejected = {entry["place_id"]: entry for entry in itinerary["not_scheduled"]}
        self.assertIn("p2", rejected)
        self.assertIn("Steep ramp", rejected["p2"]["reason"])
        self.assertIn("steps", rejected["p2"]["reason"])

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

    def test_lunch_replaces_an_adjacent_rest_and_is_not_labelled(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "p1", "name": "Unknown Place", "kind": "activity"},
                {"place_id": "rest-break", "name": "Rest", "kind": "rest"},
                {"place_id": "meal-break", "name": "Lunch", "kind": "meal"},
            ]}],
        })
        self.assertEqual(
            [i["accessibility"] for i in itinerary["days"][0]["items"]],
            ["unknown", "n/a"],
        )

    def test_end_of_day_rest_is_removed_and_only_one_useful_rest_is_kept(self):
        state = self._state()
        state.candidates["p3"] = Candidate(
            "p3", "Second Place", "activity", {}, verdict="supported"
        )
        state.candidates["p4"] = Candidate(
            "p4", "Third Place", "activity", {}, verdict="supported"
        )
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "p1", "name": "Unknown Place", "kind": "activity"},
                {"place_id": "p3", "name": "Second Place", "kind": "activity"},
                {"place_id": "rest-break", "name": "Rest", "kind": "rest"},
                {"place_id": "p4", "name": "Third Place", "kind": "activity"},
                {"place_id": "rest-break", "name": "Another rest", "kind": "rest"},
                {"place_id": "p3", "name": "Second Place again", "kind": "activity"},
                {"place_id": "rest-break", "name": "Final rest", "kind": "rest"},
            ]}],
        })
        items = itinerary["days"][0]["items"]
        self.assertEqual(sum(item["place_id"] == "rest-break" for item in items), 1)
        self.assertNotEqual(items[-1]["place_id"], "rest-break")

    def test_pairwise_travel_estimate_is_attached_across_a_generic_lunch(self):
        state = self._state()
        state.candidates["p3"] = Candidate(
            "p3", "Second Place", "activity", {}, verdict="supported"
        )
        itinerary = _enforce_verdicts(
            state,
            {
                "days": [{"day": 1, "items": [
                    {
                        "time": "09:30", "duration_min": 90,
                        "place_id": "p1", "name": "Unknown Place", "kind": "activity",
                    },
                    {
                        "time": "11:30", "duration_min": 60,
                        "place_id": "meal-break", "name": "Lunch", "kind": "meal",
                    },
                    {
                        "time": "13:00", "duration_min": 90,
                        "place_id": "p3", "name": "Second Place", "kind": "activity",
                    },
                ]}],
            },
            [{"from": "p1", "to": "p3", "min": 14, "km": 1.5}],
        )
        destination = itinerary["days"][0]["items"][2]
        travel = destination["travel_from_previous"]
        self.assertEqual(travel["km"], 1.5)
        self.assertEqual(travel["min"], 14)
        self.assertEqual(travel["from_name"], "Unknown Place")

        # 09:30 + 90 = 11:00 lunch, + 60 = 12:00, + 14 minutes of travel.
        self.assertEqual(
            [item["time"] for item in itinerary["days"][0]["items"]],
            ["09:30", "11:00", "12:14"],
        )

        state.profile = {"destination": "Test City", "trip_days": 1}
        state.itinerary = itinerary
        settings = load_settings()
        rendered = render_itinerary(state, RunTrace(settings.budget), settings)
        # The destination is the line directly above, so it is not repeated.
        self.assertIn(
            "The estimated distance from Unknown Place is about 1.5 km.", rendered
        )
        self.assertNotIn("to Second Place is about", rendered)
        self.assertIn("The schedule allows 14 min.", rendered)
        self.assertNotIn("accessible route", rendered.lower())

    def test_the_hotel_is_never_scheduled_as_an_activity(self):
        # The hotel has its own section. Scheduling it duplicates the venue,
        # and its zero duration collides with the next item's start time.
        state = self._state()
        state.selected_hotel_id = "hotel-1"
        state.candidates["hotel-1"] = Candidate(
            "hotel-1", "Hotel De Hallen", "hotel", {}, verdict="unknown"
        )
        state.candidates["museum"] = Candidate(
            "museum", "Museum Het Schip", "activity", {}, verdict="unknown"
        )
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {
                    "time": "09:00", "duration_min": 0,
                    "place_id": "hotel-1", "name": "Hotel De Hallen", "kind": "activity",
                },
                {
                    "time": "09:00", "duration_min": 90,
                    "place_id": "museum", "name": "Museum Het Schip", "kind": "activity",
                },
            ]}],
        })
        items = itinerary["days"][0]["items"]
        self.assertEqual([item["place_id"] for item in items], ["museum"])
        self.assertEqual(items[0]["time"], "09:00")

    def _two_hotels(self):
        state = self._state()
        state.selected_hotel_id = "hotel-1"
        state.candidates["hotel-1"] = Candidate(
            "hotel-1", "First Hotel", "hotel", {}, verdict="supported"
        )
        state.candidates["hotel-2"] = Candidate(
            "hotel-2", "Second Hotel", "hotel", {}, verdict="supported"
        )
        return state

    def test_moving_to_a_different_hotel_is_kept_as_a_stay(self):
        # A trip that changes accommodation has to show the move somewhere.
        state = self._two_hotels()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 2, "items": [
                {
                    "time": "09:00", "duration_min": 0,
                    "place_id": "hotel-2", "name": "Second Hotel", "kind": "stay",
                },
                {
                    "time": "09:00", "duration_min": 90,
                    "place_id": "p1", "name": "Unknown Place", "kind": "activity",
                },
            ]}],
        })
        items = itinerary["days"][0]["items"]
        self.assertEqual([item["place_id"] for item in items], ["hotel-2", "p1"])
        # Moving hotel is logistics, not a visit: the row carries no verdict.
        # The hotel's own verdict is shown under "Where you'll stay".
        self.assertEqual(items[0]["accessibility"], "n/a")
        # A zero-duration stay must not collide with what follows it.
        self.assertNotEqual(items[0]["time"], items[1]["time"])

    def test_where_youll_stay_lists_every_location_and_day_range(self):
        state = RunState("Four days in Haifa and Tel Aviv")
        state.profile = {
            "destination": "Haifa",
            "destinations": ["Haifa", "Tel Aviv"],
            "trip_days": 4,
            "needs_hotel": True,
        }
        state.plan_shape = normalize_plan_shape(
            {
                "days": [
                    {"day": 1, "location": "Haifa", "activities": 2},
                    {"day": 2, "location": "Haifa", "activities": 3},
                    {"day": 3, "location": "Tel Aviv", "activities": 2},
                    {"day": 4, "location": "Tel Aviv", "activities": 1},
                ]
            },
            state.profile,
        )
        state.candidates["h1"] = Candidate(
            "h1", "Haifa Hotel", "hotel", {"city": "Haifa"}, verdict="supported"
        )
        state.candidates["h2"] = Candidate(
            "h2", "Tel Aviv Hotel", "hotel", {"city": "Tel Aviv"}, verdict="unknown"
        )
        state.selected_hotel_id = "h1"
        state.selected_hotel_stays = [
            {"place_id": "h1", "location": "Haifa", "start_day": 1, "end_day": 2},
            {"place_id": "h2", "location": "Tel Aviv", "start_day": 3, "end_day": 4},
        ]
        state.itinerary = {"days": [], "summary": "A two-city trip."}

        settings = load_settings()
        rendered = render_itinerary(state, RunTrace(settings.budget), settings)

        self.assertIn("Days 1–2 — Haifa: Haifa Hotel", rendered)
        self.assertIn("Days 3–4 — Tel Aviv: Tel Aviv Hotel", rendered)

    def test_a_stay_row_for_the_hotel_already_booked_is_still_dropped(self):
        state = self._two_hotels()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "hotel-1", "name": "First Hotel", "kind": "stay"},
                {"place_id": "p1", "name": "Unknown Place", "kind": "activity"},
            ]}],
        })
        items = itinerary["days"][0]["items"]
        self.assertEqual([item["place_id"] for item in items], ["p1"])

    def test_a_second_hotel_with_concerns_cannot_enter_as_a_stay(self):
        state = self._two_hotels()
        state.candidates["hotel-2"].verdict = "flagged"
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 2, "items": [
                {"place_id": "hotel-2", "name": "Second Hotel", "kind": "stay"},
                {"place_id": "p1", "name": "Unknown Place", "kind": "activity"},
            ]}],
        })
        items = itinerary["days"][0]["items"]
        self.assertEqual([item["place_id"] for item in items], ["p1"])
        self.assertIn(
            "hotel-2", {entry["place_id"] for entry in itinerary["not_scheduled"]}
        )

    def test_a_venue_with_no_resolvable_location_is_not_scheduled(self):
        # An obsolete Google ID yields no coordinates, so there is no way to
        # say how far away it is or how to reach it. A stop you cannot get to
        # is not a plan -- it is dropped, and the reason is stated.
        state = self._state()
        state.candidates["ghost"] = Candidate(
            "ghost", "Vanished Pier", "activity", {}, verdict="unknown"
        )
        places = {
            "p1": SimpleNamespace(id="p1", lat=52.0, lon=4.0),
            # Present in the map but with no usable location.
            "ghost": SimpleNamespace(id="ghost", lat=0.0, lon=0.0),
        }
        itinerary = _enforce_verdicts(
            state,
            {
                "days": [{"day": 1, "items": [
                    {"place_id": "p1", "name": "Unknown Place", "kind": "activity"},
                    {"place_id": "ghost", "name": "Vanished Pier", "kind": "activity"},
                ]}],
            },
            router=LocalRouter(),
            places=places,
        )
        items = itinerary["days"][0]["items"]
        self.assertEqual([item["place_id"] for item in items], ["p1"])
        reasons = {
            entry["place_id"]: entry["reason"] for entry in itinerary["not_scheduled"]
        }
        self.assertIn("ghost", reasons)
        self.assertIn("location", reasons["ghost"].lower())

    def test_a_details_outage_empties_travel_not_the_itinerary(self):
        # No locations at all means the provider failed, not that every venue
        # is unreachable. The plan must survive.
        state = self._state()
        itinerary = _enforce_verdicts(
            state,
            {
                "days": [{"day": 1, "items": [
                    {"place_id": "p1", "name": "Unknown Place", "kind": "activity"},
                ]}],
            },
            router=LocalRouter(),
            places={},
        )
        self.assertEqual(len(itinerary["days"][0]["items"]), 1)

    def test_travel_time_between_venues_moves_the_next_start(self):
        state = self._state()
        state.profile = {"mobility": {"wheelchair": "manual"}}
        state.candidates["p3"] = Candidate(
            "p3", "Second Place", "activity", {}, verdict="supported"
        )
        places = {
            "p1": SimpleNamespace(id="p1", lat=52.0, lon=4.0),
            "p3": SimpleNamespace(id="p3", lat=52.009, lon=4.0),
        }
        itinerary = _enforce_verdicts(
            state,
            {
                "days": [{"day": 1, "items": [
                    {
                        "time": "09:00", "duration_min": 90,
                        "place_id": "p1", "name": "Unknown Place", "kind": "activity",
                    },
                    {
                        "time": "10:30", "duration_min": 90,
                        "place_id": "p3", "name": "Second Place", "kind": "activity",
                    },
                ]}],
            },
            router=LocalRouter(),
            places=places,
        )
        items = itinerary["days"][0]["items"]
        travel = items[1]["travel_from_previous"]
        self.assertGreater(len(travel["options"]), 1)
        self.assertEqual(travel["min"], max(o["minutes"] for o in travel["options"]))

        # The planner said 10:30; the walk between them pushes it later, and
        # the itinerary shows exactly how much later.
        self.assertEqual(items[0]["time"], "09:00")
        self.assertEqual(_parse_time(items[1]["time"]), 10 * 60 + 30 + travel["min"])

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
        # Every generic row is stripped of the verdict it borrowed. A real
        # activity may follow it, because a day with no attractions at all
        # gets topped up; that is a different promise, tested separately.
        generic = [item for item in items if not _is_real_activity(item, state)]
        self.assertEqual([item["accessibility"] for item in generic], ["n/a"])
        self.assertEqual(items[0]["place_id"], "meal-break")
        self.assertNotIn("Lunch break", " ".join(itinerary["things_to_confirm"]))
        self.assertNotIn("Hotel rest", " ".join(itinerary["things_to_confirm"]))
        self.assertIn(
            "restaurant-1",
            {entry["place_id"] for entry in itinerary["not_scheduled"]},
        )

    def test_supported_condition_is_visible_in_the_itinerary_note(self):
        state = self._state()
        state.candidates["conditional"] = Candidate(
            "conditional",
            "Companion Venue",
            "activity",
            {},
            verdict="supported",
            verdict_detail={"conditions": ["Visit with a companion or helper."]},
        )
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [{
                "place_id": "conditional",
                "name": "Companion Venue",
                "kind": "activity",
                "note": "Original planner note.",
            }]}],
        })
        note = itinerary["days"][0]["items"][0]["note"]
        self.assertIn("companion", note.lower())

    def test_rejected_reason_keeps_complete_long_explanation(self):
        candidate = Candidate(
            "long",
            "Domus Aurea",
            "activity",
            {},
            verdict="flagged",
            verdict_detail={
                "summary": (
                    "General wheelchair access is described, but the accessible toilet "
                    "is not directly confirmed."
                ),
                "concerns": [
                    "The source recommends companion assistance throughout the underground route."
                ],
                "conditions": ["Bring the companion throughout the visit."],
            },
        )
        reason = _flagged_reason(candidate)
        self.assertIn("Bring the companion throughout the visit.", reason)
        self.assertFalse(reason.endswith("…"))


class TestTools(unittest.TestCase):
    def setUp(self):
        self.repo = LocalRepository(FIXTURE_SEED)
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

    def test_replacement_search_excludes_already_checked_places(self):
        first = self.tools["search_activities"]({"city": "Amsterdam", "limit": 2})
        excluded_id = first["results"][0]["id"]
        replacement_tools = build_toolset(
            self.repo,
            ["step_free_entrance"],
            6,
            exclude_place_ids={excluded_id},
        )
        replacements = replacement_tools["search_activities"](
            {"city": "Amsterdam", "limit": 2}
        )
        self.assertNotIn(excluded_id, [row["id"] for row in replacements["results"]])
        self.assertEqual(len(replacements["results"]), 2)


class TestTravelEstimates(unittest.TestCase):
    def test_haversine_against_a_known_distance(self):
        # Amsterdam centre to Berlin centre is ~577km.
        km = haversine_km(52.3676, 4.9041, 52.5200, 13.4050)
        self.assertAlmostEqual(km, 577, delta=15)

    def test_identical_points_are_zero(self):
        self.assertEqual(haversine_km(52.0, 4.0, 52.0, 4.0), 0.0)

    def test_walking_is_slower_than_transit(self):
        repo = LocalRepository(FIXTURE_SEED)
        a = repo.get_place("ams-rijksmuseum")
        b = repo.get_place("ams-nemo-science-museum")
        walk = travel_estimate(a, b, "wheelchair_walk")
        transit = travel_estimate(a, b, "accessible_transit")
        self.assertGreater(walk["duration_min"], transit["duration_min"])

    def test_estimate_is_labelled_as_an_estimate(self):
        repo = LocalRepository(FIXTURE_SEED)
        estimate = travel_estimate(
            repo.get_place("ams-rijksmuseum"), repo.get_place("ams-vondelpark")
        )
        self.assertIn("estimate", estimate["basis"])


class TestTravelOptions(unittest.TestCase):
    """Which ways of getting around are worth offering, and how long they take.

    The rule that matters: never suggest a journey the traveller's mobility
    cannot actually make.
    """

    @staticmethod
    def _pair(km: float):
        """Two places roughly `km` apart, allowing for the detour factor."""
        degrees = (km / DETOUR_FACTOR) / 111.0
        return (
            SimpleNamespace(id="a", lat=52.0, lon=4.0),
            SimpleNamespace(id="b", lat=52.0 + degrees, lon=4.0),
        )

    @staticmethod
    def _profile(wheelchair="none", walking_limited=False, preferred=None):
        return {
            "mobility": {"wheelchair": wheelchair, "walking_limited": walking_limited},
            "preferred_transport": preferred,
        }

    def _modes(self, km, profile):
        origin, destination = self._pair(km)
        return [o["mode"] for o in LocalRouter().options(origin, destination, profile)]

    def test_long_hop_is_never_offered_on_foot_to_a_wheelchair_user(self):
        modes = self._modes(5.0, self._profile("manual"))
        self.assertNotIn("wheelchair_walk", modes)
        self.assertIn("accessible_transit", modes)

    def test_short_hop_is_offered_on_foot(self):
        self.assertIn("wheelchair_walk", self._modes(0.6, self._profile("manual")))

    def test_powered_chair_covers_more_ground_than_a_manual_one(self):
        self.assertGreater(
            self_powered_limit_km(self._profile("powered")),
            self_powered_limit_km(self._profile("manual")),
        )

    def test_a_walker_gets_the_shortest_self_powered_range(self):
        self.assertLess(
            self_powered_limit_km(self._profile("none", walking_limited=True)),
            self_powered_limit_km(self._profile("none")),
        )

    def test_there_is_always_at_least_one_way_to_get_there(self):
        # A taxi covers any distance, so the traveller is never left stranded.
        for km in (0.05, 1.0, 25.0):
            modes = self._modes(km, self._profile("manual", walking_limited=True))
            self.assertTrue(modes)
            self.assertIn("accessible_taxi", modes)

    def test_a_stated_preference_replaces_the_choice(self):
        modes = self._modes(1.0, self._profile("manual", preferred="accessible_taxi"))
        self.assertEqual(modes, ["accessible_taxi"])

    def test_transit_is_not_offered_for_a_few_hundred_metres(self):
        self.assertNotIn("accessible_transit", self._modes(0.2, self._profile()))

    def test_the_schedule_is_built_on_the_slowest_option(self):
        chosen = planning_option(
            [
                {"mode": "accessible_taxi", "minutes": 7},
                {"mode": "wheelchair_walk", "minutes": 20},
                {"mode": "accessible_transit", "minutes": 15},
            ]
        )
        self.assertEqual(chosen["mode"], "wheelchair_walk")

    def test_every_option_carries_a_time_and_a_readable_label(self):
        origin, destination = self._pair(1.0)
        options = LocalRouter().options(origin, destination, self._profile("manual"))
        for option in options:
            self.assertGreater(option["minutes"], 0)
            self.assertTrue(option["label"])
        self.assertIn("~", describe_options(options))


class TestGoogleRoutesRouter(unittest.TestCase):
    ORIGIN = SimpleNamespace(id="a", lat=52.0, lon=4.0)
    DESTINATION = SimpleNamespace(id="b", lat=52.01, lon=4.0)

    def test_live_durations_replace_the_estimate(self):
        payload = {"routes": [{"duration": "780s", "distanceMeters": 1400}]}
        with patch("atex.routing.post_json", return_value=payload):
            options = GoogleRoutesRouter("key").options(
                self.ORIGIN, self.DESTINATION, {"preferred_transport": "accessible_taxi"}
            )
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["source"], "google")
        self.assertEqual(options[0]["km"], 1.4)
        self.assertEqual(options[0]["minutes"], 15)

    def test_a_provider_failure_falls_back_to_the_estimate(self):
        with patch("atex.routing.post_json", side_effect=HttpError(403, "denied", "u")):
            options = GoogleRoutesRouter("key").options(
                self.ORIGIN, self.DESTINATION, {"preferred_transport": "accessible_taxi"}
            )
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["source"], "estimate")
        self.assertGreater(options[0]["minutes"], 0)

    def test_a_route_google_cannot_find_falls_back_to_the_estimate(self):
        # No transit in this city, for example: `routes` comes back empty.
        with patch("atex.routing.post_json", return_value={"routes": []}):
            options = GoogleRoutesRouter("key").options(
                self.ORIGIN, self.DESTINATION, {"preferred_transport": "accessible_transit"}
            )
        self.assertEqual(options[0]["source"], "estimate")

    def test_googles_walking_time_is_slowed_for_a_wheelchair_user(self):
        payload = {"routes": [{"duration": "600s", "distanceMeters": 800}]}
        profile = {
            "mobility": {"wheelchair": "manual"},
            "preferred_transport": "wheelchair_walk",
        }
        able = {"mobility": {"wheelchair": "none"}, "preferred_transport": "wheelchair_walk"}
        with patch("atex.routing.post_json", return_value=payload):
            slow = GoogleRoutesRouter("key").options(self.ORIGIN, self.DESTINATION, profile)
            quick = GoogleRoutesRouter("key").options(self.ORIGIN, self.DESTINATION, able)
        self.assertGreater(slow[0]["minutes"], quick[0]["minutes"])

    def test_repeated_pairs_are_only_routed_once(self):
        payload = {"routes": [{"duration": "300s", "distanceMeters": 500}]}
        profile = {"preferred_transport": "accessible_taxi"}
        router = GoogleRoutesRouter("key")
        with patch("atex.routing.post_json", return_value=payload) as call:
            router.options(self.ORIGIN, self.DESTINATION, profile)
            router.options(self.ORIGIN, self.DESTINATION, profile)
        self.assertEqual(call.call_count, 1)

    def test_a_partly_answered_hop_stays_internally_consistent(self):
        # Google answered DRIVE but not WALK. Estimating the walk from the
        # short straight line while quoting Google's longer route distance
        # made walking look faster than a taxi over the same hop.
        far = SimpleNamespace(id="c", lat=52.03, lon=4.0)
        profile = {"mobility": {"wheelchair": "powered"}}

        def only_drive(url, body, **kwargs):
            if body["travelMode"] != "DRIVE":
                raise HttpError(404, "no route", url)
            return {"routes": [{"duration": "600s", "distanceMeters": 4200}]}

        with patch("atex.routing.post_json", side_effect=only_drive):
            options = GoogleRoutesRouter("key").options(self.ORIGIN, far, profile)

        by_mode = {o["mode"]: o for o in options}
        self.assertEqual(by_mode["accessible_taxi"]["source"], "google")
        self.assertIn("accessible_transit", by_mode)
        # Every option is measured against the same distance...
        self.assertEqual(
            {o["km"] for o in options}, {by_mode["accessible_taxi"]["km"]}
        )
        # ...so a slower mode can never come out quicker than a faster one.
        self.assertGreaterEqual(
            by_mode["accessible_transit"]["minutes"],
            by_mode["accessible_taxi"]["minutes"],
        )

    def test_a_two_hour_bus_ride_is_not_offered_as_an_option(self):
        # Los Angeles offered a 145-minute transit hop between two attractions
        # on the same day. Because the schedule is laid out on the slowest
        # option, that one hop swallowed two and a half hours of the day.
        options = drop_unreasonable([
            {"mode": "accessible_transit", "minutes": 145, "km": 30},
            {"mode": "accessible_taxi", "minutes": 20, "km": 30},
        ])
        self.assertEqual([o["mode"] for o in options], ["accessible_taxi"])

    def test_a_reasonable_bus_ride_is_kept(self):
        options = drop_unreasonable([
            {"mode": "accessible_transit", "minutes": 35, "km": 6},
            {"mode": "accessible_taxi", "minutes": 20, "km": 6},
        ])
        self.assertEqual(len(options), 2)

    def test_the_traveller_is_never_left_with_no_way_to_get_there(self):
        # If every option is unreasonable the least bad one still stands: the
        # venue is already in the day, so silence is not an answer.
        options = drop_unreasonable([
            {"mode": "accessible_transit", "minutes": 200, "km": 60},
        ])
        self.assertEqual(len(options), 1)

    def test_capping_transit_shortens_what_the_schedule_allows(self):
        kept = drop_unreasonable([
            {"mode": "accessible_transit", "minutes": 145, "km": 30},
            {"mode": "accessible_taxi", "minutes": 20, "km": 30},
        ])
        self.assertEqual(planning_option(kept)["minutes"], 20)

    def test_no_maps_key_selects_the_local_router(self):
        self.assertIsInstance(build_router(SimpleNamespace(google_maps_api_key="")), LocalRouter)
        self.assertIsInstance(
            build_router(SimpleNamespace(google_maps_api_key="k")), GoogleRoutesRouter
        )


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
        from atex.config import settings
        from atex.context import AgentContext
        from atex.tracing import RunTrace

        wanted: dict[str, str] = {}
        for path in FIXTURE_KB.glob("*.json"):
            for chunk in json.loads(path.read_text(encoding="utf-8")).get("chunks", []):
                if chunk.get("place_id"):
                    wanted.setdefault(chunk["place_id"], chunk.get("city") or "")

        self.assertTrue(wanted, "no place-specific chunks found in the fixture knowledge base")

        cfg = settings()
        ctx = AgentContext.build(RunTrace(budget=cfg.budget), cfg)
        repo = LocalRepository(FIXTURE_SEED)

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
        repo = LocalRepository(FIXTURE_SEED)

        # This place deliberately has no chunk of its own.
        place = repo.get_place("ams-cafe-centrum")
        candidate = Candidate(place.id, place.name, place.kind, place.to_brief())
        specific, general = retrieve_evidence(ctx, candidate, "Amsterdam", ["step_free_entrance"])

        self.assertEqual(specific, [], "city notes must not count as venue evidence")
        self.assertTrue(general, "city-scope notes should still be retrievable")

    def test_google_place_semantic_fallback_searches_destination_first(self):
        class DestinationAwareVectors:
            def __init__(self):
                self.filters = []

            def query(self, vector, top_k, flt=None):
                self.filters.append(flt)
                if flt == {"city": "Rome"}:
                    return [
                        Match(
                            "rome-capitoline",
                            0.8,
                            "Capitoline Museums provides lift access to the galleries.",
                            {"city": "Rome", "source": "guide"},
                        )
                    ]
                return []

        vectors = DestinationAwareVectors()
        ctx = SimpleNamespace(
            settings=SimpleNamespace(budget=SimpleNamespace(rag_top_k=5)),
            embedder=SimpleNamespace(embed=lambda texts: [[0.1]]),
            vectors=vectors,
        )
        candidate = Candidate(
            "gmp:google-id", "Capitoline Museums", "activity", {}
        )

        specific, _ = retrieve_evidence(
            ctx, candidate, "Rome", ["step_free_entrance", "lift_access"]
        )

        self.assertTrue(specific)
        self.assertIn({"city": "Rome"}, vectors.filters)


class TestPlanShape(unittest.TestCase):
    """How full a day is, is the Supervisor's call -- not a fixed table."""

    def test_the_supervisors_choice_wins_over_the_stated_limit(self):
        shape = normalize_plan_shape(
            {"activities_per_day": 4, "day_start": "09:00", "day_end": "21:00"},
            {"max_activities_per_day": 2},
        )
        self.assertEqual(shape["activities_per_day"], 4)
        self.assertEqual(shape["day_end"], "21:00")

    def test_a_late_finish_is_carried_through(self):
        shape = normalize_plan_shape({"day_start": "10:00", "day_end": "22:30"}, {})
        self.assertEqual(shape["day_start"], "10:00")
        self.assertEqual(shape["day_end"], "22:30")

    def test_a_missing_decision_falls_back_to_the_stated_limit(self):
        shape = normalize_plan_shape(None, {"max_activities_per_day": 5})
        self.assertEqual(shape["activities_per_day"], 5)

    def test_a_missing_decision_and_no_limit_still_yields_a_usable_day(self):
        shape = normalize_plan_shape(None, {"pace": "relaxed"})
        self.assertEqual(shape["activities_per_day"], 2)
        self.assertGreater(_parse_time(shape["day_end"]), _parse_time(shape["day_start"]))

    def test_nonsense_values_cannot_produce_an_unplannable_day(self):
        shape = normalize_plan_shape(
            {"activities_per_day": 99, "day_start": "25:99", "day_end": "01:00"}, {}
        )
        self.assertLessEqual(shape["activities_per_day"], MAX_ACTIVITIES_PER_DAY)
        self.assertGreater(_parse_time(shape["day_end"]), _parse_time(shape["day_start"]))

    def test_the_day_starts_when_the_supervisor_said(self):
        state = RunState("x")
        state.plan_shape = normalize_plan_shape({"day_start": "11:00"}, {})
        state.candidates["p1"] = Candidate("p1", "A Place", "activity", {}, verdict="supported")
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "p1", "name": "A Place", "kind": "activity", "duration_min": 90},
            ]}],
        })
        self.assertEqual(itinerary["days"][0]["items"][0]["time"], "11:00")

    def test_the_finder_is_asked_for_enough_places_to_fill_the_shape(self):
        state = RunState("x")
        state.profile = {"trip_days": 7}
        state.plan_shape = normalize_plan_shape({"activities_per_day": 4}, {})
        self.assertEqual(activities_needed(state), 28)

    def test_each_day_keeps_its_own_activity_target(self):
        profile = {"destination": "Haifa", "trip_days": 3}
        shape = normalize_plan_shape(
            {
                "days": [
                    {"day": 1, "location": "Haifa", "activities": 1},
                    {"day": 2, "location": "Haifa", "activities": 4},
                    {"day": 3, "location": "Tel Aviv", "activities": 2},
                ]
            },
            profile,
        )
        self.assertEqual([day["activities"] for day in shape["days"]], [1, 4, 2])
        state = RunState("x", profile=profile, plan_shape=shape)
        self.assertEqual(activities_needed(state), 7)

    def test_nearby_city_is_allowed_unless_the_user_said_only(self):
        raw = {
            "days": [
                {"day": 1, "location": "Haifa", "activities": 2},
                {"day": 2, "location": "Tel Aviv", "activities": 3},
            ]
        }
        flexible = normalize_plan_shape(
            raw,
            {"destination": "Haifa", "destinations": ["Haifa"], "trip_days": 2},
        )
        strict = normalize_plan_shape(
            raw,
            {
                "destination": "Haifa",
                "destinations": ["Haifa"],
                "requested_locations_only": True,
                "trip_days": 2,
            },
        )
        self.assertEqual(flexible["days"][1]["location"], "Tel Aviv")
        self.assertEqual(strict["days"][1]["location"], "Haifa")

    def test_every_explicitly_requested_location_gets_a_day(self):
        shape = normalize_plan_shape(
            {
                "days": [
                    {"day": 1, "location": "Haifa", "activities": 2},
                    {"day": 2, "location": "Haifa", "activities": 2},
                    {"day": 3, "location": "Haifa", "activities": 2},
                ]
            },
            {
                "destination": "Haifa",
                "destinations": ["Haifa", "Tel Aviv"],
                "trip_days": 3,
            },
        )
        self.assertIn("Tel Aviv", [day["location"] for day in shape["days"]])

    def test_multi_location_days_create_one_hotel_segment_per_location(self):
        shape = normalize_plan_shape(
            {
                "days": [
                    {"day": 1, "location": "Haifa", "activities": 2},
                    {"day": 2, "location": "Haifa", "activities": 3},
                    {"day": 3, "location": "Tel Aviv", "activities": 2},
                ]
            },
            {"destination": "Haifa", "trip_days": 3, "needs_hotel": True},
        )
        self.assertEqual(
            shape["hotel_stays"],
            [
                {"location": "Haifa", "start_day": 1, "end_day": 2},
                {"location": "Tel Aviv", "start_day": 3, "end_day": 3},
            ],
        )

    def test_planner_cannot_put_a_candidate_in_the_wrong_city_day(self):
        state = RunState("Two days in Haifa and Tel Aviv")
        state.profile = {"destination": "Haifa", "trip_days": 2}
        state.plan_shape = normalize_plan_shape(
            {
                "days": [
                    {"day": 1, "location": "Haifa", "activities": 1},
                    {"day": 2, "location": "Tel Aviv", "activities": 1},
                ]
            },
            state.profile,
        )
        state.candidates["tel-aviv-place"] = Candidate(
            "tel-aviv-place",
            "Tel Aviv Museum",
            "activity",
            {"city": "Tel Aviv"},
            verdict="supported",
        )
        itinerary = _enforce_verdicts(
            state,
            {
                "days": [
                    {
                        "day": 1,
                        "items": [
                            {
                                "place_id": "tel-aviv-place",
                                "name": "Tel Aviv Museum",
                                "kind": "activity",
                            }
                        ],
                    }
                ]
            },
        )
        self.assertEqual(itinerary["days"][0]["items"], [])


class TestDaysAreActuallyFilled(unittest.TestCase):
    """A stated target is a promise, so code keeps it rather than the prompt.

    A fourteen-day Los Angeles request asked for three attractions a day. The
    Supervisor set that shape and the planner was told it, but it returned two
    a day while forty-eight checked candidates went unused.
    """

    def _state(self, spare=6, per_day=3):
        state = RunState("x")
        state.profile = {"trip_days": 1, "destination": "Los Angeles"}
        state.plan_shape = normalize_plan_shape(
            {"activities_per_day": per_day, "day_start": "09:00", "day_end": "20:00"},
            state.profile,
        )
        for i in range(spare):
            state.candidates[f"a{i}"] = Candidate(
                f"a{i}", f"Attraction {i}", "activity", {"duration_min": 90},
                verdict="supported" if i % 2 else "unknown",
            )
        return state

    def test_a_short_day_is_topped_up_to_its_target(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [
                {"place_id": "a0", "name": "Attraction 0", "kind": "activity"},
            ]}],
        })
        items = itinerary["days"][0]["items"]
        activities = [i for i in items if _is_real_activity(i, state)]
        self.assertEqual(len(activities), 3)

    def test_verified_places_are_used_before_unverified_ones(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {"days": [{"day": 1, "items": []}]})
        activities = [
            i for i in itinerary["days"][0]["items"] if _is_real_activity(i, state)
        ]
        self.assertEqual(activities[0]["accessibility"], "supported")

    def test_a_day_is_never_padded_with_the_same_place_twice(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {"days": [{"day": 1, "items": []}]})
        ids = [i["place_id"] for i in itinerary["days"][0]["items"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_filling_never_reuses_a_place_from_another_day(self):
        state = self._state(spare=4)
        state.profile["trip_days"] = 2
        state.plan_shape = normalize_plan_shape(
            {"activities_per_day": 2, "day_start": "09:00", "day_end": "20:00"},
            state.profile,
        )
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": []}, {"day": 2, "items": []}],
        })
        seen = [
            item["place_id"]
            for day in itinerary["days"]
            for item in day["items"]
        ]
        self.assertEqual(len(seen), len(set(seen)))

    def test_a_flagged_place_is_never_used_to_fill_a_day(self):
        state = self._state(spare=1)
        state.candidates["bad"] = Candidate(
            "bad", "Steps Museum", "activity", {}, verdict="flagged"
        )
        itinerary = _enforce_verdicts(state, {"days": [{"day": 1, "items": []}]})
        ids = {i["place_id"] for i in itinerary["days"][0]["items"]}
        self.assertNotIn("bad", ids)

    def test_filling_stops_at_the_end_of_the_day(self):
        # Twelve 90-minute stops cannot fit between 09:00 and 20:00.
        state = self._state(spare=12, per_day=12)
        itinerary = _enforce_verdicts(state, {"days": [{"day": 1, "items": []}]})
        items = itinerary["days"][0]["items"]
        self.assertTrue(items)
        for item in items:
            self.assertLess(_parse_time(item["time"]), 20 * 60)

    def test_a_filled_unverified_place_still_needs_confirming(self):
        state = self._state(spare=2, per_day=2)
        itinerary = _enforce_verdicts(state, {"days": [{"day": 1, "items": []}]})
        unknown = [
            i for i in itinerary["days"][0]["items"]
            if i.get("accessibility") == "unknown"
        ]
        self.assertTrue(unknown)
        confirm = " ".join(itinerary["things_to_confirm"])
        for item in unknown:
            self.assertIn(item["name"], confirm)


class TestLocationSpelling(unittest.TestCase):
    """A typo is not a second city.

    "2 weeks in Los Angels" put thirteen days in "Los Angeles" and one in
    "Los Angels", which read as two destinations and produced two hotel
    segments: days 1-13 and day 14.
    """

    def test_a_misspelt_city_is_the_same_city(self):
        self.assertTrue(same_location("Los Angeles", "Los Angels"))
        self.assertTrue(same_location("Amsterdam", "amsterdam "))
        self.assertTrue(same_location("Tel-Aviv", "Tel Aviv"))

    def test_genuinely_different_cities_stay_different(self):
        self.assertFalse(same_location("Los Angeles", "Las Vegas"))
        self.assertFalse(same_location("Rome", "Bern"))
        self.assertFalse(same_location("Bath", "Bern"))

    def test_a_typo_does_not_split_the_hotel_stay(self):
        profile = {
            "trip_days": 14, "destination": "Los Angeles",
            "destinations": ["Los Angeles"], "needs_hotel": True,
        }
        shape = normalize_plan_shape(
            {"days": [
                {"day": d, "location": "Los Angeles" if d < 14 else "Los Angels",
                 "activities": 3}
                for d in range(1, 15)
            ]},
            profile,
        )
        self.assertEqual({d["location"] for d in shape["days"]}, {"Los Angeles"})
        self.assertEqual(len(shape["hotel_stays"]), 1)
        self.assertEqual(shape["hotel_stays"][0]["end_day"], 14)


class TestAccommodationSplit(unittest.TestCase):
    """One hotel the first week, another the second, in a single city."""

    PROFILE = {
        "trip_days": 14, "destination": "Los Angeles",
        "destinations": ["Los Angeles"], "needs_hotel": True,
    }
    DAYS = [
        {"day": d, "location": "Los Angeles", "activities": 3} for d in range(1, 15)
    ]

    def test_a_requested_split_is_honoured(self):
        shape = normalize_plan_shape(
            {"days": self.DAYS,
             "hotel_stays": [{"start_day": 1, "end_day": 7},
                             {"start_day": 8, "end_day": 14}]},
            self.PROFILE,
        )
        self.assertEqual(
            [(s["start_day"], s["end_day"]) for s in shape["hotel_stays"]],
            [(1, 7), (8, 14)],
        )
        self.assertTrue(all(s["location"] == "Los Angeles" for s in shape["hotel_stays"]))

    def test_a_split_with_a_gap_is_rejected_entirely(self):
        # A partial split would leave nights with nowhere to sleep.
        shape = normalize_plan_shape(
            {"days": self.DAYS,
             "hotel_stays": [{"start_day": 1, "end_day": 5},
                             {"start_day": 9, "end_day": 14}]},
            self.PROFILE,
        )
        self.assertEqual(len(shape["hotel_stays"]), 1)

    def test_a_split_that_overshoots_the_trip_is_rejected(self):
        shape = normalize_plan_shape(
            {"days": self.DAYS,
             "hotel_stays": [{"start_day": 1, "end_day": 20}]},
            self.PROFILE,
        )
        self.assertEqual(shape["hotel_stays"][0]["end_day"], 14)

    def test_no_split_given_leaves_one_stay(self):
        shape = normalize_plan_shape({"days": self.DAYS}, self.PROFILE)
        self.assertEqual(len(shape["hotel_stays"]), 1)


class TestHotelMoveAndDeparture(unittest.TestCase):
    def _state(self):
        state = RunState("x")
        state.profile = {
            "trip_days": 2, "destination": "Rome", "destinations": ["Rome"],
            "needs_hotel": True,
        }
        state.plan_shape = normalize_plan_shape(
            {"days": [{"day": 1, "location": "Rome", "activities": 1},
                      {"day": 2, "location": "Rome", "activities": 1}],
             "hotel_stays": [{"start_day": 1, "end_day": 1},
                             {"start_day": 2, "end_day": 2}],
             "day_start": "09:00", "day_end": "20:00"},
            state.profile,
        )
        state.candidates["h1"] = Candidate("h1", "First Hotel", "hotel", {}, verdict="supported")
        state.candidates["h2"] = Candidate("h2", "Second Hotel", "hotel", {}, verdict="supported")
        state.candidates["a1"] = Candidate(
            "a1", "Colosseum", "activity", {"duration_min": 90}, verdict="supported"
        )
        state.selected_hotel_id = "h1"
        state.selected_hotel_stays = [
            {"place_id": "h1", "location": "Rome", "start_day": 1, "end_day": 1},
            {"place_id": "h2", "location": "Rome", "start_day": 2, "end_day": 2},
        ]
        return state

    PLACES = {
        "h1": SimpleNamespace(id="h1", lat=41.900, lon=12.500),
        "h2": SimpleNamespace(id="h2", lat=41.920, lon=12.520),
        "a1": SimpleNamespace(id="a1", lat=41.890, lon=12.492),
    }

    def test_the_hotel_move_appears_in_the_day_it_happens(self):
        # Checking out, crossing the city with luggage and checking in costs
        # real time; a schedule that skips it hands that time to an attraction.
        state = self._state()
        itinerary = _enforce_verdicts(
            state,
            {"days": [{"day": 1, "items": []},
                      {"day": 2, "items": [
                          {"place_id": "a1", "name": "Colosseum", "kind": "activity"}]}]},
            router=LocalRouter(), places=self.PLACES,
        )
        day2 = itinerary["days"][1]["items"]
        self.assertEqual(day2[0]["place_id"], "h2")
        self.assertEqual(day2[0]["kind"], "stay")
        self.assertGreater(_duration(day2[0]), 0)

    def test_the_move_carries_no_accessibility_label(self):
        # It is logistics, not a visit. The hotel's verdict lives in
        # "Where you'll stay", which is where a traveller looks for it.
        state = self._state()
        itinerary = _enforce_verdicts(
            state,
            {"days": [{"day": 1, "items": []}, {"day": 2, "items": []}]},
            router=LocalRouter(), places=self.PLACES,
        )
        move = itinerary["days"][1]["items"][0]
        self.assertEqual(move["accessibility"], "n/a")

    def test_no_move_row_on_a_trip_that_never_changes_hotel(self):
        state = self._state()
        state.selected_hotel_stays = [
            {"place_id": "h1", "location": "Rome", "start_day": 1, "end_day": 2}
        ]
        itinerary = _enforce_verdicts(
            state,
            {"days": [{"day": 1, "items": []}, {"day": 2, "items": []}]},
            router=LocalRouter(), places=self.PLACES,
        )
        for day in itinerary["days"]:
            self.assertNotIn("stay", [i.get("kind") for i in day["items"]])

    def test_the_day_allows_time_to_get_out_of_the_hotel(self):
        # Days used to begin at day_start sharp at the first attraction, as
        # though the traveller woke up inside it.
        state = self._state()
        state.selected_hotel_stays = [
            {"place_id": "h1", "location": "Rome", "start_day": 1, "end_day": 2}
        ]
        itinerary = _enforce_verdicts(
            state,
            {"days": [{"day": 1, "items": [
                {"place_id": "a1", "name": "Colosseum", "kind": "activity"}]}]},
            router=LocalRouter(), places=self.PLACES,
        )
        first = itinerary["days"][0]["items"][0]
        travel = first["travel_from_previous"]
        self.assertEqual(travel["from_name"], "First Hotel")
        self.assertGreater(travel["min"], 0)
        # 09:00 plus the journey out of the hotel, not 09:00 sharp.
        self.assertEqual(_parse_time(first["time"]), 9 * 60 + travel["min"])


class TestOutOfScopeRequests(unittest.TestCase):
    """A question that is not about travel must cost one model call, not a run."""

    class _Supervisor:
        """An LLM that always declines, and counts what it was asked."""

        def __init__(self):
            self.calls = 0

        def complete_json(self, module, system, user, **kwargs):
            self.calls += 1
            return {
                "reasoning": "Not a travel request.",
                "next_module": "OUT_OF_SCOPE",
                "instruction": "That is a question about commodity prices, not a trip.",
                "clarification_question": None,
            }

    def test_an_unrelated_question_stops_after_one_call(self):
        llm = self._Supervisor()
        with patch("atex.context.AgentContext.build") as build:
            build.side_effect = lambda trace, cfg: SimpleNamespace(
                trace=trace, settings=cfg, llm=llm, repo=None, vectors=None, embedder=None
            )
            result = run_agent("what is the average price of tomato across the world")
        self.assertEqual(llm.calls, 1, "declining must not run the other modules")
        self.assertIsNone(result.state.itinerary)
        self.assertTrue(result.state.out_of_scope)

    def test_the_refusal_is_the_same_words_every_time(self):
        # Fixed wording: the one reply that should never vary must not be the
        # one reply nobody has reviewed.
        tomato = RunState("what is the average price of tomato across the world")
        tomato.out_of_scope = True
        code = RunState("write me a python script")
        code.out_of_scope = True
        self.assertEqual(render_out_of_scope(tomato), render_out_of_scope(code))

    def test_the_refusal_says_what_the_agent_does_instead(self):
        state = RunState("what is the average price of tomato across the world")
        state.out_of_scope = True
        rendered = render_out_of_scope(state)
        self.assertIn("not a question about planning an accessible trip", rendered)
        self.assertIn("ATEX plans accessible trips", rendered)
        # The off-topic subject is never echoed back.
        self.assertNotIn("tomato", rendered.lower())

    def test_declining_is_never_corrected_into_planning_work(self):
        state = RunState("what is the price of tomatoes")
        allowed, corrected = _legalize(state, "OUT_OF_SCOPE")
        self.assertEqual(allowed, "OUT_OF_SCOPE")
        self.assertIsNone(corrected)


class TestResponseTidiness(unittest.TestCase):
    """The response is read by a traveller, not by a developer."""

    def _state(self):
        state = RunState("x")
        state.candidates["gmp:ChIJWT0gUBz2wokRNcAxVUphAAs"] = Candidate(
            "gmp:ChIJWT0gUBz2wokRNcAxVUphAAs",
            "El Museo del Barrio",
            "activity",
            {},
            verdict="unknown",
        )
        return state

    def test_place_ids_never_reach_the_confirm_list(self):
        # The planner prefixes these lines with the raw Google ID it was
        # working from. A traveller cannot act on that.
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [{
                "place_id": "gmp:ChIJWT0gUBz2wokRNcAxVUphAAs",
                "name": "El Museo del Barrio",
                "kind": "activity",
            }]}],
            "things_to_confirm": [
                "gmp:ChIJWT0gUBz2wokRNcAxVUphAAs - El Museo del Barrio: confirm step-free entrance."
            ],
        })
        confirm = itinerary["things_to_confirm"]
        self.assertTrue(confirm)
        joined = " ".join(confirm)
        self.assertNotIn("gmp:", joined)
        self.assertNotIn("ChIJWT0gUBz2wokRNcAxVUphAAs", joined)
        self.assertIn("El Museo del Barrio", joined)
        self.assertIn("step-free entrance", joined)

    def test_a_confirm_line_we_generate_uses_the_venue_name(self):
        state = self._state()
        itinerary = _enforce_verdicts(state, {
            "days": [{"day": 1, "items": [{
                "place_id": "gmp:ChIJWT0gUBz2wokRNcAxVUphAAs",
                "kind": "activity",
            }]}],
        })
        joined = " ".join(itinerary["things_to_confirm"])
        self.assertIn("El Museo del Barrio", joined)
        self.assertNotIn("gmp:", joined)

    def test_surplus_unverified_places_are_not_listed_as_rejections(self):
        # An unverified place is usable, so "no evidence" is not a reason to
        # reject it. Forty such entries buried the ones that mattered.
        state = RunState("x")
        for i in range(30):
            state.candidates[f"u{i}"] = Candidate(
                f"u{i}", f"Museum {i}", "activity", {}, verdict="unknown"
            )
        state.candidates["bad"] = Candidate(
            "bad", "Steps Museum", "activity", {},
            verdict="flagged",
            verdict_detail={"summary": "The only entrance described has steps."},
        )
        itinerary = _enforce_verdicts(state, {
            "days": [],
            "not_scheduled": [
                {"place_id": f"u{i}", "name": f"Museum {i}",
                 "reason": "Unknown accessibility; no information in the knowledge base."}
                for i in range(30)
            ],
        })
        entries = itinerary["not_scheduled"]
        dicts = [e for e in entries if isinstance(e, dict)]
        self.assertLessEqual(len(dicts), MAX_NOT_SCHEDULED)
        # The one real concern survives, and is first.
        self.assertEqual(dicts[0]["place_id"], "bad")
        # The rest collapse into a single honest count.
        tail = [e for e in entries if isinstance(e, str)]
        self.assertEqual(len(tail), 1)
        self.assertIn("30", tail[0])

    def test_a_short_rejection_list_gets_no_overflow_line(self):
        state = RunState("x")
        state.candidates["bad"] = Candidate(
            "bad", "Steps Museum", "activity", {},
            verdict="flagged",
            verdict_detail={"summary": "Steps at the only entrance."},
        )
        itinerary = _enforce_verdicts(state, {"days": [], "not_scheduled": []})
        self.assertTrue(all(isinstance(e, dict) for e in itinerary["not_scheduled"]))


class TestRunCapacity(unittest.TestCase):
    """The limits have to fit the trips the system claims to plan."""

    def _profile(self, days, per_day=3):
        state = RunState("x")
        state.profile = {"trip_days": days, "max_activities_per_day": per_day}
        return state

    def test_a_two_week_trip_asks_for_enough_places_to_fill_it(self):
        self.assertEqual(activities_needed(self._profile(14, 3)), 42)

    def test_the_budget_can_validate_a_two_week_trip(self):
        # 42 activities plus a hotel and spares must not hit the per-run cap,
        # which is what left places "not checked" on the New York run.
        budget = load_settings().budget
        self.assertGreaterEqual(budget.max_validations_per_run, 50)

    def test_the_budget_allows_the_calls_a_long_trip_costs(self):
        budget = load_settings().budget
        needed = (
            1                                          # UserProfileAgent
            + budget.react_max_iters                   # one finder round
            + -(-50 // budget.validation_batch_size)   # validator batches
            + 1                                        # SchedulePlanner
            + budget.max_supervisor_turns
        )
        self.assertGreaterEqual(budget.max_total_llm_calls, needed)

    def test_the_wall_clock_stays_inside_the_platform_limit(self):
        # vercel.json requests 300s; finishing must happen before that.
        budget = load_settings().budget
        self.assertLessEqual(budget.wall_clock_budget_s, 290.0)
        self.assertGreater(budget.reserve_wall_clock_s, 0)


class TestConcernReplacementRouting(unittest.TestCase):
    @staticmethod
    def _ctx():
        return SimpleNamespace(trace=RunTrace(load_settings().budget))

    def _state(self, finder_rounds: int) -> RunState:
        state = RunState("Two days in Rome using a wheelchair")
        state.profile = {"destination": "Rome", "trip_days": 2}
        state.finder_rounds = finder_rounds
        state.candidates["gmp:flagged"] = Candidate(
            "gmp:flagged",
            "Museum With Steps",
            "activity",
            {},
            verdict="flagged",
            verdict_detail={"summary": "The only entrance described has steps."},
        )
        return state

    def test_concern_triggers_one_replacement_search(self):
        decision = decide(self._ctx(), self._state(finder_rounds=1))
        self.assertEqual(decision.next_module, "ActivityLogisticsFinder")
        self.assertIn("Museum With Steps", decision.instruction)

    def test_finder_has_four_rounds_and_revisits_when_day_targets_are_short(self):
        self.assertEqual(MAX_FINDER_ROUNDS, 4)
        state = RunState("Three days in Haifa")
        state.profile = {"destination": "Haifa", "trip_days": 3}
        state.plan_shape = normalize_plan_shape(
            {
                "days": [
                    {"day": 1, "location": "Haifa", "activities": 1},
                    {"day": 2, "location": "Haifa", "activities": 2},
                    {"day": 3, "location": "Tel Aviv", "activities": 1},
                ]
            },
            state.profile,
        )
        state.finder_rounds = 1
        state.candidates["one"] = Candidate(
            "one", "One Museum", "activity", {}, verdict="supported"
        )

        decision = decide(self._ctx(), state)

        self.assertEqual(decision.next_module, "ActivityLogisticsFinder")
        self.assertIn("2 in Haifa", decision.instruction)
        self.assertIn("Tel Aviv", decision.instruction)

    def test_an_empty_result_run_finishes_instead_of_replanning(self):
        # Discovery found nothing and the planner already said so. Re-running
        # a module cannot improve that; it just burns turns until the limit.
        state = RunState("Two weeks somewhere with no coverage")
        state.profile = {"destination": "Nowhere", "trip_days": 14}
        state.finder_rounds = MAX_FINDER_ROUNDS
        state.itinerary = {"days": [], "summary": "No candidates."}

        ctx = self._ctx()
        decision = decide(ctx, state)

        self.assertEqual(decision.next_module, "FINISH")
        self.assertEqual(ctx.trace.llm_calls, 0)

    def test_concern_does_not_create_an_unbounded_search_loop(self):
        decision = decide(self._ctx(), self._state(finder_rounds=MAX_FINDER_ROUNDS))
        self.assertEqual(decision.next_module, "SchedulePlanner")


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

    def test_a_new_destination_clears_the_previous_trip_state(self):
        first = run_agent(
            "Three days in Amsterdam, manual wheelchair, relaxed pace, need a hotel."
        )
        saved = first.session_state()
        saved["turn_index"] = first.state.turn_index

        second = run_agent(
            "I am travelling alone to Rome for six days and use a manual wheelchair.",
            session_id="same-browser-session",
            saved_state=saved,
        )

        self.assertIsNone(second.error)
        self.assertEqual(second.state.profile["destination"], "Rome")
        self.assertEqual(second.state.profile["destinations"], ["Rome"])
        self.assertNotIn("Amsterdam", second.state.shape["search_locations"])
        self.assertIn("Rome", second.response)
        self.assertNotIn("itinerary: Amsterdam", second.response)

    def test_same_trip_follow_up_refreshes_profile_without_losing_destination(self):
        first = run_agent("Three days in Amsterdam with a manual wheelchair.")
        saved = first.session_state()
        second = run_agent(
            "Make the second day quieter.",
            session_id="same-trip",
            saved_state=saved,
        )

        self.assertEqual(second.state.profile["destination"], "Amsterdam")
        self.assertFalse(second.state.profile_needs_refresh)


class TestFollowUpKeepsWhatWasNotMentioned(unittest.TestCase):
    """A follow-up names only what it changes; the rest still stands.

    "I want a different hotel" came back asking which city the traveller
    meant, one message after they had said Los Angeles. Re-extracting that
    message yields a profile with no destination, which then read as a
    destination change and cleared the whole trip.
    """

    PREVIOUS = normalize_profile({
        "destination": "Los Angeles",
        "destinations": ["Los Angeles"],
        "trip_days": 14,
        "needs_hotel": True,
        "max_activities_per_day": 3,
        "mobility": {"wheelchair": "powered", "step_free_required": True},
        "accessibility_needs": ["step_free_entrance", "accessible_toilet", "lift_access"],
    })

    def _updated(self, raw):
        return normalize_profile(_carry_forward(raw, self.PREVIOUS))

    def _state(self):
        state = RunState("i want a different hotel")
        state.profile = self.PREVIOUS
        state.candidates["a"] = Candidate(
            "a", "A Place", "activity", {}, verdict="supported"
        )
        state.selected_hotel_stays = [
            {"place_id": "h1", "location": "Los Angeles", "start_day": 1, "end_day": 14}
        ]
        state.plan_shape = normalize_plan_shape(None, self.PREVIOUS)
        return state

    def test_a_silent_follow_up_keeps_the_destination(self):
        updated = self._updated({"destination": None, "destinations": [], "mobility": {}})
        self.assertEqual(updated["destination"], "Los Angeles")
        self.assertEqual(updated["destinations"], ["Los Angeles"])

    def test_a_silent_follow_up_keeps_the_trip_length(self):
        updated = self._updated({"destination": None, "mobility": {}})
        self.assertEqual(updated["trip_days"], 14)

    def test_a_silent_follow_up_keeps_the_wheelchair(self):
        # Losing this would quietly relax every accessibility requirement.
        updated = self._updated({"destination": None, "mobility": {}})
        self.assertEqual(updated["mobility"]["wheelchair"], "powered")
        self.assertIn("lift_access", updated["accessibility_needs"])

    def test_a_silent_follow_up_keeps_the_paid_for_candidates(self):
        state = self._state()
        updated = self._updated({"destination": None, "destinations": [], "mobility": {}})
        _apply_profile_change(state, self.PREVIOUS, updated)
        self.assertEqual(len(state.candidates), 1)
        self.assertEqual(state.candidates["a"].verdict, "supported")
        self.assertTrue(state.selected_hotel_stays)
        self.assertIsNotNone(state.plan_shape)

    def test_a_stated_false_still_wins_over_the_old_value(self):
        # Silence is refilled; an explicit answer never is.
        updated = self._updated({"needs_hotel": False, "mobility": {}})
        self.assertFalse(updated["needs_hotel"])

    def test_a_named_new_destination_still_clears_the_trip(self):
        state = self._state()
        updated = self._updated(
            {"destination": "Berlin", "destinations": ["Berlin"], "mobility": {}}
        )
        _apply_profile_change(state, self.PREVIOUS, updated)
        self.assertEqual(updated["destination"], "Berlin")
        self.assertEqual(len(state.candidates), 0)
        self.assertIsNone(state.plan_shape)

    def test_asking_for_a_different_hotel_is_recognised(self):
        for request in (
            "i want a different hotel",
            "I want another hotel please",
            "change the hotel",
            "can we swap our accommodation",
            "I don't like the hotel",
            "book a different place to stay",
        ):
            self.assertTrue(_replacement_hotel_requested(request), request)

    def test_an_ordinary_follow_up_does_not_release_the_hotel(self):
        for request in (
            "Make the second day quieter.",
            "Add a museum on day three.",
            "Is the hotel far from the centre?",
        ):
            self.assertFalse(_replacement_hotel_requested(request), request)

    def test_releasing_the_hotel_forces_discovery_to_find_another(self):
        # Replanning alone cannot honour the request: the itinerary is rebuilt
        # from the same candidates, so the same hotel wins again.
        state = self._state()
        state.selected_hotel_id = "h1"
        state.candidates["h1"] = Candidate(
            "h1", "First Hotel", "hotel", {}, verdict="supported"
        )
        _release_hotel_selection(state)
        self.assertIsNone(state.selected_hotel_id)
        self.assertEqual(state.selected_hotel_stays, [])
        self.assertEqual(state.finder_rounds, 0)
        # The rejected hotel stays known, which is what stops the next search
        # from offering it straight back.
        self.assertIn("h1", state.candidates)

    def test_a_different_hotel_is_actually_selected(self):
        first = run_agent("Three days in Amsterdam, manual wheelchair, need a hotel.")
        saved = first.session_state()
        saved["turn_index"] = first.state.turn_index
        second = run_agent(
            "i want a different hotel", session_id="swap", saved_state=saved
        )
        self.assertIsNotNone(first.state.selected_hotel_id)
        self.assertIsNotNone(second.state.selected_hotel_id)
        self.assertNotEqual(
            first.state.selected_hotel_id, second.state.selected_hotel_id
        )
        # And the trip it belongs to is still the same trip.
        self.assertEqual(second.state.profile["destination"], "Amsterdam")
        self.assertEqual(second.state.profile["trip_days"], 3)

    def test_a_changed_trip_length_reshapes_without_discarding_candidates(self):
        state = self._state()
        updated = self._updated({"destination": None, "trip_days": 7, "mobility": {}})
        _apply_profile_change(state, self.PREVIOUS, updated)
        self.assertEqual(updated["trip_days"], 7)
        self.assertEqual(len(state.candidates), 1)
        self.assertIsNone(state.plan_shape)


if __name__ == "__main__":
    unittest.main()
