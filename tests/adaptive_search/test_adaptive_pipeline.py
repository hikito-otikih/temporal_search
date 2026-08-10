import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from adaptive_search.rewrite_bridge import build_session_plan
from adaptive_search.client import UpstreamSearchClient, UpstreamSearchError
from main import app
from rewrite import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
)


def rewritten_event(index, query, boundary="start"):
    return {
        "event_id": index,
        "original_query": query,
        "target_moment_vi": f"Khoảnh khắc con lân màu vàng đen trắng ở sự kiện {index}.",
        "anchor_query": f"Con lân màu vàng đen trắng trong màn múa lân ở sự kiện {index}.",
        "retrieval_queries_vi": [
            f"Con lân màu vàng đen trắng thực hiện hành động ở sự kiện {index}.",
            f"Khoảnh khắc lân màu vàng đen trắng ở sự kiện {index}.",
        ],
        "retrieval_queries_en": [
            f"The yellow, black, and white lion performs in event {index}.",
            f"The lion-dance moment in event {index}.",
        ],
        "subject": "con lân",
        "action": "thực hiện hành động",
        "visible_state": "hành động của con lân có thể nhìn thấy",
        "pre_state": f"Ngay trước sự kiện {index}, con lân chưa thực hiện hành động.",
        "post_state": f"Ngay sau sự kiện {index}, con lân vừa hoàn thành hành động.",
        "boundary": boundary,
        "temporal_relation": {
            "relation": "sequence_start" if index == 0 else "independent",
            "reference_event_id": None,
        },
        "required_entities": ["con lân"],
        "soft_context": ["màn múa lân"],
        "excluded_context": [],
        "inferred_information": ["Từ common_query: lân màu vàng đen trắng."],
        "ambiguities": [],
    }


def analysis(queries):
    from rewrite.schemas import RewriteResponse

    return RewriteResponse.model_validate(
        {
            "video_context": {
                "scene": "múa lân",
                "main_entities": ["một con lân màu vàng, đen và trắng"],
            },
            "events": [
                rewritten_event(index, query)
                for index, query in enumerate(queries)
            ],
        }
    )


def fake_upstream_client():
    def fake_transport(url, payload, timeout):
        return {
            "query": payload["query"],
            "top_k": payload["top_k"],
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

    def create_session_from_queries(self, queries):
        plan = build_session_plan(
            analysis(queries), common_query="Video múa lân."
        )
        rewrite_mock = AsyncMock(return_value=plan)
        with patch(
            "adaptive_search.router.rewrite_queries_and_build_plan", rewrite_mock
        ):
            response = self.client.post(
                "/v1/search-sessions/from-queries",
                json={
                    "common_query": "Video múa lân.",
                    "query": queries,
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["session"]["id"]

    def test_from_queries_creates_session_with_events_and_variants(self):
        queries = [
            "Khoảnh khắc đầu tiên con lân xoay vòng.",
            "Khoảnh khắc bốn chân chạm đất đầu tiên.",
        ]
        session_id = self.create_session_from_queries(queries)

        response = self.client.get(f"/v1/search-sessions/{session_id}")
        self.assertEqual(response.status_code, 200, response.text)
        session = response.json()["session"]

        self.assertEqual(
            [event["event_id"] for event in session["events"]],
            ["evt0", "evt1"],
        )
        self.assertEqual(session["events"][0]["boundary_type"], "onset")
        self.assertEqual(
            len(session["retrieval_variants"]["evt0"]), 4
        )
        self.assertEqual(
            session["retrieval_variants"]["evt1"][0],
            "Con lân màu vàng đen trắng thực hiện hành động ở sự kiện 1.",
        )
    def test_from_queries_rejects_unknown_request_fields(self):
        response = self.client.post(
            "/v1/search-sessions/from-queries",
            json={
                "query": ["Một sự kiện."],
                "modelname": "gpt-oss:20b",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_from_queries_maps_ollama_errors(self):
        payload = {"query": ["Một sự kiện."]}
        cases = (
            (ConfigurationError("missing"), 500, "ollama_not_configured"),
            (OllamaTimeoutError("slow"), 504, "ollama_timeout"),
            (OllamaRateLimitError("limited"), 503, "ollama_rate_limited"),
            (OllamaServiceError("down"), 502, "ollama_service_error"),
        )
        for error, expected_status, expected_code in cases:
            with self.subTest(error=type(error).__name__):
                with patch(
                    "adaptive_search.router.rewrite_queries_and_build_plan",
                    AsyncMock(side_effect=error),
                ):
                    response = self.client.post(
                        "/v1/search-sessions/from-queries", json=payload
                    )
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(
                    response.json()["detail"]["code"], expected_code
                )

    def test_retrieve_ingests_upstream_candidates_into_regions(self):
        session_id = self.create_session_from_queries(
            ["Khoảnh khắc đầu tiên con lân xoay vòng."]
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
            all(item["event_id"] == "evt0" for item in regions)
        )

    def test_retrieve_falls_back_to_anchor_query_without_stored_variants(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "Khoảnh khắc đầu tiên con lân xoay vòng.",
                        "anchor_query": "Con lân bắt đầu xoay vòng trên cột.",
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        session_id = response.json()["session"]["id"]

        with patch(
            "adaptive_search.router.upstream_search_client", fake_upstream_client()
        ):
            response = self.client.post(
                f"/v1/search-sessions/{session_id}/commands/retrieve",
                json={"top_k": 5},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(
            response.json()["artifact_counts"]["candidates"], 0
        )

    def test_retrieve_restricts_to_requested_events(self):
        session_id = self.create_session_from_queries(
            ["Sự kiện thứ nhất.", "Sự kiện thứ hai."]
        )

        with patch(
            "adaptive_search.router.upstream_search_client", fake_upstream_client()
        ):
            response = self.client.post(
                f"/v1/search-sessions/{session_id}/commands/retrieve",
                json={"top_k": 5, "event_ids": ["evt0"]},
            )

        self.assertEqual(response.status_code, 200, response.text)
        candidates = response.json()["artifact_counts"]["candidates"]
        self.assertGreater(candidates, 0)
        self.assertLess(candidates, 4)

    def test_retrieve_rejects_unknown_event(self):
        session_id = self.create_session_from_queries(["Sự kiện một."])
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
            return {"query": "different", "results": []}

        client = UpstreamSearchClient(transport=broken_transport)
        session_id = self.create_session_from_queries(["Sự kiện một."])

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
