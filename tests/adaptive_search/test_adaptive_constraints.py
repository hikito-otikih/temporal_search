import unittest

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from main import app


def event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "original_query": event_id,
        "anchor_query": event_id,
    }


class AdaptiveConstraintApiTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)
        response = self.client.post(
            "/v1/search-sessions",
            json={"events": [event("e1"), event("e2"), event("e3")]},
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.session_id = response.json()["session"]["id"]

    def tearDown(self):
        self.client.close()

    def test_gap_constraints_are_typed_revisioned_and_adjacent(self):
        constraints = {
            "adjacent_gap_constraints": [
                {
                    "before_event_id": "e1",
                    "after_event_id": "e2",
                    "min_gap_seconds": 1.0,
                    "max_gap_seconds": 5.0,
                }
            ]
        }
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={"expected_revision": 0, "constraints": constraints},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["session_revision"], 1)
        self.assertEqual(response.json()["invalidated_stages"], ["tuple"])

        no_op = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={"expected_revision": 1, "constraints": constraints},
        )
        self.assertEqual(no_op.status_code, 200, no_op.text)
        self.assertEqual(no_op.json()["session_revision"], 1)
        self.assertEqual(no_op.json()["invalidated_stages"], [])

        non_adjacent = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 1,
                "constraints": {
                    "adjacent_gap_constraints": [
                        {
                            "before_event_id": "e1",
                            "after_event_id": "e3",
                        }
                    ]
                },
            },
        )
        self.assertEqual(non_adjacent.status_code, 422, non_adjacent.text)

    def test_prioritized_video_cannot_also_be_rejected(self):
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {
                    "prioritized_video_ids": ["video_a"],
                    "rejected_video_ids": ["video_a"],
                },
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_prioritized_video_ids_must_not_contain_duplicates(self):
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {"prioritized_video_ids": ["video_a", "video_a"]},
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_standalone_fixed_region_id_without_fixed_video_id_is_rejected(self):
        # A fixed_region_id with no fixed_video_id used to be silently inert
        # (_region_allowed's narrowing only fires once it knows which video
        # to scope to) - reject it outright instead of accepting a
        # constraint that pins nothing.
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {"event_constraints": {"e1": {"fixed_region_id": "some-region"}}},
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_fixed_video_id_alone_has_no_effect_and_is_rejected(self):
        # fixed_video_id alone narrows nothing in _region_allowed - reject
        # it outright instead of accepting a constraint that does nothing.
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {"event_constraints": {"e1": {"fixed_video_id": "video_a"}}},
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_fixed_frame_id_without_timestamp_has_no_effect_and_is_rejected(self):
        # fixed_frame_id's own value is never read by ranking - only its
        # nullness matters, alongside fixed_timestamp_seconds. Without a
        # timestamp, fixed_video_id + fixed_frame_id narrows nothing either.
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {
                    "event_constraints": {
                        "e1": {"fixed_video_id": "video_a", "fixed_frame_id": 42}
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_fixed_frame_and_region_without_timestamp_has_no_effect_and_is_rejected(self):
        # Regression test for a real reported bug: fixed_region_id is
        # allowed through the "fixed_video_id alone" check as long as it's
        # set, but _region_allowed's region-id-only narrowing requires
        # fixed_frame_id to be None, and its timestamp-window narrowing
        # requires fixed_timestamp_seconds to be set - with fixed_frame_id
        # set and no timestamp, neither ever fires, so this combination was
        # a 200 that silently filtered nothing.
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {
                    "event_constraints": {
                        "e1": {
                            "fixed_video_id": "video_a",
                            "fixed_region_id": "some-region",
                            "fixed_frame_id": 42,
                        }
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_fixed_region_and_timestamp_without_frame_is_ambiguous_and_rejected(self):
        # Regression test for a real reported bug: with fixed_frame_id
        # absent, _region_allowed/_effective_region_timestamp treat
        # fixed_region_id as a genuine region pin and never read
        # fixed_timestamp_seconds at all - but video_priorities.py's
        # boundary-refinement skip-path checks only fixed_video_id +
        # fixed_timestamp_seconds, with no awareness of fixed_region_id, and
        # would report that pinned timestamp as the event's refined_seconds
        # regardless of which timestamp the winning tuple actually used.
        # Two consumers disagreeing about the same constraint is not a
        # combination that should return 200.
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {
                    "event_constraints": {
                        "e1": {
                            "fixed_video_id": "video_a",
                            "fixed_region_id": "some-region",
                            "fixed_timestamp_seconds": 12.0,
                        }
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_fixed_region_id_alone_is_a_valid_plain_region_pin(self):
        # video + region_id only (no frame, no timestamp) is a genuinely
        # self-consistent shape: _region_allowed narrows to that region and
        # _effective_region_timestamp reports the region's own best
        # candidate - both agree, so this must remain accepted.
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {
                    "event_constraints": {
                        "e1": {"fixed_video_id": "video_a", "fixed_region_id": "some-region"}
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_fixed_video_frame_and_timestamp_without_region_is_accepted(self):
        # The full fix_frame()-equivalent exact-frame pin shape (region_id
        # absent) must remain accepted - fixed_frame_id set alongside a
        # timestamp is exactly what commands/fix-frame itself produces when
        # no existing region is passed through.
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {
                    "event_constraints": {
                        "e1": {
                            "fixed_video_id": "video_a",
                            "fixed_frame_id": 42,
                            "fixed_timestamp_seconds": 12.0,
                        }
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_video_scope_constraint_invalidates_downstream_stages(self):
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {"allowed_video_ids": ["video-a"]},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["invalidated_stages"],
            ["refinement", "proposal", "tuple"],
        )

    def test_unknown_event_reference_is_rejected(self):
        response = self.client.put(
            f"/v1/search-sessions/{self.session_id}/constraints",
            json={
                "expected_revision": 0,
                "constraints": {"event_constraints": {"missing": {}}},
            },
        )
        self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
