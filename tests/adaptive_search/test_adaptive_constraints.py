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

    def test_video_scope_constraint_rebuilds_frontier_contract(self):
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
            ["frontier", "refinement", "proposal", "tuple"],
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
