import unittest

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from main import app


def event(event_id, text):
    return {
        "event_id": event_id,
        "original_query": text,
        "anchor_query": text,
        "pre_state": f"before {text}",
        "post_state": f"after {text}",
        "boundary_type": "transition",
    }


class AdaptiveApiTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def create_session(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={"events": [event("e1", "first"), event("e2", "second")]},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["session"]["id"]

    def test_capability_is_honest_when_frame_provider_is_missing(self):
        response = self.client.get("/v1/searchers")
        self.assertEqual(response.status_code, 200)
        adaptive = next(
            item
            for item in response.json()["searchers"]
            if item["searcher_type"] == "adaptive_temporal"
        )
        self.assertTrue(adaptive["algorithm_core_available"])
        self.assertFalse(adaptive["live_refinement_available"])
        self.assertIn("raw video", adaptive["frame_provider"]["reason"])

    def test_video_priorities_ranks_by_coverage_and_score(self):
        session_id = self.create_session()
        candidates = [
            {
                "id": "c1",
                "session_id": session_id,
                "event_id": "e1",
                "video_id": "video_full",
                "frame_id": 20,
                "timestamp_seconds": 2.0,
                "raw_relevance_score": 0.9,
                "query_variant": "anchor-en-0",
            },
            {
                "id": "c2",
                "session_id": session_id,
                "event_id": "e2",
                "video_id": "video_full",
                "frame_id": 60,
                "timestamp_seconds": 6.0,
                "raw_relevance_score": 0.9,
                "query_variant": "anchor-en-0",
            },
            {
                "id": "c3",
                "session_id": session_id,
                "event_id": "e1",
                "video_id": "video_partial",
                "frame_id": 20,
                "timestamp_seconds": 2.0,
                "raw_relevance_score": 0.9,
                "query_variant": "anchor-en-0",
            },
        ]
        response = self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={
                "expected_revision": 0,
                "event_ids": ["e1", "e2"],
                "candidates": candidates,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

        priorities_response = self.client.get(
            f"/v1/search-sessions/{session_id}/video-priorities"
        )
        self.assertEqual(priorities_response.status_code, 200, priorities_response.text)
        items = priorities_response.json()["items"]

        self.assertEqual([item["video_id"] for item in items], ["video_full", "video_partial"])
        self.assertGreater(items[0]["priority_score"], items[1]["priority_score"])
        self.assertEqual(items[0]["event_coverage"], 2)
        self.assertEqual(items[1]["event_coverage"], 1)

    def test_stale_hyperparameter_patch_conflicts_with_current_revision(self):
        session_id = self.create_session()
        self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={
                "expected_revision": 0,
                "event_ids": ["e1", "e2"],
                "candidates": [
                    {
                        "id": "c1", "session_id": session_id, "event_id": "e1",
                        "video_id": "video", "frame_id": 20, "timestamp_seconds": 2.0,
                        "raw_relevance_score": 0.8, "query_variant": "anchor-en-0",
                    },
                ],
            },
        )

        response = self.client.patch(
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={
                "expected_revision": 0,
                "patch": {"tuple_ranking": {"order_weight": 0.5}},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["session_revision"], 1)
        self.assertEqual(response.json()["invalidated_stages"], ["tuple"])

        stale = self.client.patch(
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={
                "expected_revision": 0,
                "patch": {"tuple_ranking": {"order_weight": 0.6}},
            },
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["detail"]["current_revision"], 1)

    def test_patching_tuple_ranking_hyperparameters_succeeds(self):
        # tuple_ranking is a separate SearchHyperparameters section from
        # "ranking" - a real, valid patch to it had no matching invalidation
        # prefix at all and was rejected with a 422 regardless of how valid
        # the patch itself was.
        session_id = self.create_session()
        self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={
                "expected_revision": 0,
                "event_ids": ["e1", "e2"],
                "candidates": [
                    {
                        "id": "c1", "session_id": session_id, "event_id": "e1",
                        "video_id": "video", "frame_id": 20, "timestamp_seconds": 2.0,
                        "raw_relevance_score": 0.8, "query_variant": "anchor-en-0",
                    },
                ],
            },
        )
        response = self.client.patch(
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={
                "expected_revision": 0,
                "patch": {"tuple_ranking": {"order_weight": 0.5}},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["invalidated_stages"], ["tuple"])

    def test_patching_one_events_anchor_query_rebuilds_regions_not_wipes_them(self):
        # anchor_query maps to stage "retrieval", whose downstream sweep
        # includes "region" - a patch used to silently delete every region
        # in the session with no rebuild afterward: 200 OK, but
        # /video-priorities and /regions were both broken until a fresh
        # commands/retrieve. Scoped to e1 only (event_ids={"e1"}), so e2's
        # candidates survive invalidation and its region must come back via
        # the same rebuild, not just e1's.
        session_id = self.create_session()
        self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={
                "expected_revision": 0,
                "event_ids": ["e1", "e2"],
                "candidates": [
                    {
                        "id": "c1", "session_id": session_id, "event_id": "e1",
                        "video_id": "video", "frame_id": 20, "timestamp_seconds": 2.0,
                        "raw_relevance_score": 0.8, "query_variant": "anchor-en-0",
                    },
                    {
                        "id": "c2", "session_id": session_id, "event_id": "e2",
                        "video_id": "video", "frame_id": 60, "timestamp_seconds": 6.0,
                        "raw_relevance_score": 0.9, "query_variant": "anchor-en-0",
                    },
                ],
            },
        )
        response = self.client.patch(
            f"/v1/search-sessions/{session_id}/events/e1",
            json={
                "expected_revision": 0,
                "patch": {"anchor_query": "a different anchor query"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["artifact_counts"]["regions"], 1)

        regions_after = self.client.get(f"/v1/search-sessions/{session_id}/regions")
        region_events = {item["event_id"] for item in regions_after.json()["items"]}
        self.assertEqual(region_events, {"e2"})

        priorities_after = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(priorities_after.status_code, 200, priorities_after.text)

    def test_legacy_request_rejects_silent_searcher_fallbacks(self):
        invalid_payloads = (
            {"query": []},
            {"query": ["one"], "searcher_type": "typo"},
            {"query": ["one"], "gamma": -0.1},
            {"query": ["one"], "unexpected": True},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/temporal-search", json=payload)
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
