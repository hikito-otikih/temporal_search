import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from adaptive_search.client import UpstreamSearchClient
from main import app


def event(event_id, anchor_query):
    return {"event_id": event_id, "anchor_query": anchor_query}


def fake_upstream_client():
    def fake_transport(url, payload, timeout):
        return {
            "query": [payload["query"]],
            "results": [
                {
                    "video_name": "video-a.mp4",
                    "frame_index": 90,
                    "timestamp": "0:03",
                    "score": 0.75,
                },
                {
                    "video_name": "video-b.mp4",
                    "frame_index": 30,
                    "timestamp": "0:01",
                    "score": 0.5,
                },
            ],
        }

    return UpstreamSearchClient(transport=fake_transport)


class AdaptivePipelineTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def create_session(self, events):
        response = self.client.post(
            "/v1/search-sessions",
            json={"events": events},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["session"]["id"]

    def test_retrieve_ingests_upstream_candidates_into_regions(self):
        session_id = self.create_session(
            [event("e1", "Khoảnh khắc đầu tiên con lân xoay vòng.")]
        )

        with patch(
            "adaptive_search.router.upstream_search_client", fake_upstream_client()
        ):
            response = self.client.post(
                f"/v1/search-sessions/{session_id}/commands/retrieve",
                json={"top_k": 5},
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["run_status"], "completed")
        self.assertGreater(body["artifact_counts"]["candidates"], 0)
        self.assertGreater(body["artifact_counts"]["regions"], 0)

        regions = self.client.get(
            f"/v1/search-sessions/{session_id}/regions"
        ).json()["items"]
        self.assertTrue(
            all(item["event_id"] == "e1" for item in regions)
        )

    def test_retrieve_uses_anchor_query_directly(self):
        # No rewrite/variant indirection exists any more - the event's own
        # anchor_query is exactly the text sent upstream.
        anchor_query = "Con lân bắt đầu xoay vòng trên cột."
        sent_queries = []

        def spy_transport(url, payload, timeout):
            sent_queries.append(payload["query"])
            return {
                "query": [payload["query"]],
                "results": [],
            }

        session_id = self.create_session([event("e1", anchor_query)])

        with patch(
            "adaptive_search.router.upstream_search_client",
            UpstreamSearchClient(transport=spy_transport),
        ):
            response = self.client.post(
                f"/v1/search-sessions/{session_id}/commands/retrieve",
                json={"top_k": 5},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(sent_queries, [anchor_query])

    def test_retrieve_restricts_to_requested_events(self):
        session_id = self.create_session(
            [event("e1", "Sự kiện thứ nhất."), event("e2", "Sự kiện thứ hai.")]
        )

        with patch(
            "adaptive_search.router.upstream_search_client", fake_upstream_client()
        ):
            response = self.client.post(
                f"/v1/search-sessions/{session_id}/commands/retrieve",
                json={"top_k": 5, "event_ids": ["e1"]},
            )

        self.assertEqual(response.status_code, 200, response.text)
        candidates = response.json()["artifact_counts"]["candidates"]
        self.assertGreater(candidates, 0)
        self.assertLess(candidates, 4)

    def test_retrieve_rejects_unknown_event(self):
        session_id = self.create_session([event("e1", "Sự kiện một.")])
        with patch(
            "adaptive_search.router.upstream_search_client", fake_upstream_client()
        ):
            response = self.client.post(
                f"/v1/search-sessions/{session_id}/commands/retrieve",
                json={"event_ids": ["nope"]},
            )
        self.assertEqual(response.status_code, 422)

    def test_retrieve_maps_upstream_errors_to_bad_gateway(self):
        def broken_transport(url, payload, timeout):
            return {"query": ["different"], "results": []}

        client = UpstreamSearchClient(transport=broken_transport)
        session_id = self.create_session([event("e1", "Sự kiện một.")])

        with patch(
            "adaptive_search.router.upstream_search_client", client
        ):
            response = self.client.post(
                f"/v1/search-sessions/{session_id}/commands/retrieve",
                json={"top_k": 5},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"]["code"], "upstream_search_error"
        )

    def test_retrieve_unknown_session_returns_404(self):
        response = self.client.post(
            "/v1/search-sessions/missing/commands/retrieve",
            json={"top_k": 5},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
