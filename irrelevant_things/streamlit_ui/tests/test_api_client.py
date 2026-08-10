import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from httpx import MockTransport

from services.api_client import (
    ConnectionFailure,
    HttpFailure,
    RevisionConflictError,
    SessionNotFound,
    TemporalApiClient,
    ValidationFailure,
)


def make_client(handler) -> TemporalApiClient:
    transport = MockTransport(handler)
    return TemporalApiClient(transport=transport)


def json_response(payload, status_code=200):
    return httpx.Response(status_code, json=payload, request=httpx.Request("GET", "http://x"))


class ApiClientTests(unittest.TestCase):
    def test_success_capability_mapping(self):
        def handler(request):
            return json_response(
                {
                    "searchers": [
                        {
                            "searcher_type": "adaptive_temporal",
                            "available": True,
                            "session_api": True,
                            "algorithm_core_available": True,
                            "live_refinement_available": False,
                            "frame_provider": {
                                "available": False,
                                "reason": "no raw video",
                            },
                        }
                    ]
                }
            )

        client = make_client(handler)
        capabilities = client.get_capabilities()
        adaptive = capabilities.adaptive()
        self.assertIsNotNone(adaptive)
        self.assertTrue(adaptive.algorithm_core_available)
        self.assertFalse(adaptive.live_refinement_available)
        self.assertEqual(adaptive.frame_provider.reason, "no raw video")

    def test_validation_error_raises_validation_failure(self):
        def handler(request):
            return json_response(
                {
                    "detail": {
                        "code": "invalid_adaptive_input",
                        "message": "candidate session_id does not match URL",
                    }
                },
                status_code=422,
            )

        client = make_client(handler)
        with self.assertRaises(ValidationFailure) as ctx:
            client.ingest_candidates("s1", expected_revision=0, event_ids=["e1"], candidates=[])
        self.assertIn("does not match", str(ctx.exception))

    def test_revision_conflict_raises_with_revisions(self):
        def handler(request):
            return json_response(
                {
                    "detail": {
                        "code": "revision_conflict",
                        "message": "stale",
                        "expected_revision": 0,
                        "current_revision": 1,
                    }
                },
                status_code=409,
            )

        client = make_client(handler)
        with self.assertRaises(RevisionConflictError) as ctx:
            client.patch_hyperparameters("s1", expected_revision=0, patch={"ranking": {}})
        self.assertEqual(ctx.exception.expected, 0)
        self.assertEqual(ctx.exception.current, 1)

    def test_session_not_found(self):
        def handler(request):
            return json_response(
                {"detail": {"code": "session_not_found", "message": "nope"}},
                status_code=404,
            )

        client = make_client(handler)
        with self.assertRaises(SessionNotFound):
            client.get_session("missing")

    def test_generic_http_failure(self):
        def handler(request):
            return json_response({"detail": "boom"}, status_code=500)

        client = make_client(handler)
        with self.assertRaises(HttpFailure) as ctx:
            client.check_backend_health()
        self.assertEqual(ctx.exception.status_code, 500)

    def test_connection_failure_maps_to_connection_failure(self):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        client = make_client(handler)
        with self.assertRaises(ConnectionFailure):
            client.check_backend_health()

    def test_legacy_request_payload_is_exact(self):
        captured = {}

        def handler(request):
            captured["method"] = request.method
            captured["url"] = request.url
            captured["json"] = __import__("json").loads(request.content)
            return json_response({"query": ["q"], "results": []})

        client = make_client(handler)
        client.legacy_search(
            queries=["q"],
            top_k_tuple=10,
            gamma=0.1,
            searcher_type="TemporalSearcher",
        )
        payload = captured["json"]
        self.assertEqual(payload["query"], ["q"])
        self.assertEqual(payload["top_k_tuple"], 10)
        self.assertEqual(payload["gamma"], 0.1)
        self.assertEqual(payload["searcher_type"], "TemporalSearcher")
        self.assertEqual(payload["objectFilterMode"], False)

    def test_upstream_search_payload_is_sanitized(self):
        captured = {}

        def handler(request):
            captured["json"] = __import__("json").loads(request.content)
            return json_response({"query": "q", "results": []})

        client = make_client(handler)
        client.upstream_search("some event text", top_k=100)
        self.assertEqual(set(captured["json"]), {"query", "top_k"})


if __name__ == "__main__":
    unittest.main()
