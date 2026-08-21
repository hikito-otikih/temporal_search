import unittest

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from main import app


class AdaptiveJsonContractTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_json_arrays_are_accepted_for_typed_tuple_and_set_fields(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "anchor_query": "first",
                    },
                    {
                        "event_id": "e2",
                        "anchor_query": "second",
                    },
                ],
                "constraints": {
                    "allowed_video_ids": ["video"],
                    "adjacent_gap_constraints": [
                        {
                            "before_event_id": "e1",
                            "after_event_id": "e2",
                        }
                    ],
                },
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        session = response.json()["session"]
        self.assertEqual(session["constraints"]["allowed_video_ids"], ["video"])
        self.assertEqual(
            session["constraints"]["adjacent_gap_constraints"][0]["before_event_id"],
            "e1",
        )


if __name__ == "__main__":
    unittest.main()
