import unittest

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from main import app


class AdaptiveContractTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_session_endpoint_rejects_legacy_searcher_types(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "searcher_type": "legacy_temporal",
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "person enters",
                        "anchor_query": "a person entering",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_searcher_capability_maps_legacy_values_and_boundary_profiles(self):
        response = self.client.get("/v1/searchers")
        self.assertEqual(response.status_code, 200)
        searchers = {
            item["searcher_type"]: item for item in response.json()["searchers"]
        }

        self.assertEqual(
            searchers["legacy_temporal"]["legacy_request_value"],
            "TemporalSearcher",
        )
        self.assertEqual(
            searchers["legacy_ambiguous"]["legacy_request_value"],
            "AmbiguousSearcher",
        )
        self.assertEqual(
            set(searchers["adaptive_temporal"]["supported_boundary_profiles"]),
            {
                "onset",
                "offset",
                "transition",
                "state",
                "plateau_start",
                "symmetric_peak",
                "unknown",
            },
        )


if __name__ == "__main__":
    unittest.main()
