import unittest

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from main import app


class AdaptiveNoOpApiTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_identical_patches_do_not_increment_revision(self):
        created = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "anchor_query": "event",
                    }
                ]
            },
        )
        self.assertEqual(created.status_code, 201, created.text)
        session_id = created.json()["session"]["id"]

        event_patch = self.client.patch(
            f"/v1/search-sessions/{session_id}/events/e1",
            json={
                "expected_revision": 0,
                "patch": {"anchor_query": "event"},
            },
        )
        self.assertEqual(event_patch.status_code, 200, event_patch.text)
        self.assertEqual(event_patch.json()["session_revision"], 0)
        self.assertEqual(event_patch.json()["invalidated_stages"], [])

        parameter_patch = self.client.patch(
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={
                "expected_revision": 0,
                "patch": {"tuple_ranking": {"relative_delta": 0.15}},
            },
        )
        self.assertEqual(parameter_patch.status_code, 200, parameter_patch.text)
        self.assertEqual(parameter_patch.json()["session_revision"], 0)
        self.assertEqual(parameter_patch.json()["invalidated_stages"], [])

    def test_legacy_query_rejects_whitespace_only_text(self):
        response = self.client.post("/temporal-search", json={"query": ["   "]})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
