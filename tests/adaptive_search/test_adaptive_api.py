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
        from adaptive_search.algorithms import prioritize_videos
        from adaptive_search.schemas import TemporalRegion

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

        regions = self.client.get(f"/v1/search-sessions/{session_id}/regions").json()["items"]
        session = self.client.get(f"/v1/search-sessions/{session_id}").json()["session"]

        def to_region(item):
            item = {k: v for k, v in item.items() if k != "selected_for_refinement"}
            item["candidate_ids"] = tuple(item["candidate_ids"])
            return TemporalRegion.model_validate(item)

        expected = prioritize_videos(
            [to_region(item) for item in regions],
            [event["event_id"] for event in session["events"]],
        )
        self.assertEqual(
            [item.video_id for item in expected],
            [item["video_id"] for item in items],
        )
        self.assertEqual(
            [item.priority_score for item in expected],
            [item["priority_score"] for item in items],
        )

    def test_session_candidate_score_and_tuple_pipeline(self):
        session_id = self.create_session()
        candidates = [
            {
                "id": "c1",
                "session_id": session_id,
                "event_id": "e1",
                "video_id": "video",
                "frame_id": 20,
                "timestamp_seconds": 2.0,
                "raw_relevance_score": 0.8,
                "query_variant": "anchor-en-0",
            },
            {
                "id": "c2",
                "session_id": session_id,
                "event_id": "e2",
                "video_id": "video",
                "frame_id": 60,
                "timestamp_seconds": 6.0,
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
        self.assertEqual(response.json()["artifact_counts"]["regions"], 2)

        regions_response = self.client.get(
            f"/v1/search-sessions/{session_id}/regions"
        )
        regions = regions_response.json()["items"]
        region_by_event = {item["event_id"]: item for item in regions}
        samples = []
        for event_id, center in (("e1", 2.0), ("e2", 6.0)):
            region = region_by_event[event_id]
            for offset in (-2, -1, 0, 1, 2):
                timestamp = center + offset
                before = offset < 0
                samples.append(
                    {
                        "session_id": session_id,
                        "event_id": event_id,
                        "video_id": "video",
                        "region_id": region["id"],
                        "frame_id": int(timestamp * 10),
                        "timestamp_seconds": timestamp,
                        "raw_anchor_score": 1.0 if offset == 0 else 0.1,
                        "raw_pre_score": 2.0 if before else -2.0,
                        "raw_post_score": -2.0 if before else 2.0,
                        "raw_motion_score": 0.0,
                    }
                )

        partial = self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/frame-scores",
            json={
                "expected_revision": 0,
                "region_ids": [item["id"] for item in regions],
                "samples": [
                    sample for sample in samples if sample["event_id"] == "e1"
                ],
            },
        )
        self.assertEqual(partial.status_code, 422, partial.text)
        self.assertEqual(
            partial.json()["detail"]["code"],
            "invalid_adaptive_input",
        )
        unchanged = self.client.get(
            f"/v1/search-sessions/{session_id}"
        ).json()
        self.assertEqual(unchanged["artifact_counts"]["frame_scores"], 0)
        self.assertTrue(
            all(
                item["refinement_status"] == "pending"
                for item in self.client.get(
                    f"/v1/search-sessions/{session_id}/regions"
                ).json()["items"]
            )
        )

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/frame-scores",
            json={
                "expected_revision": 0,
                "region_ids": [item["id"] for item in regions],
                "samples": samples,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        counts = response.json()["artifact_counts"]
        self.assertGreater(counts["proposals"], 0)
        self.assertGreater(counts["tuples"], 0)

        tuples = self.client.get(
            f"/v1/search-sessions/{session_id}/tuples"
        ).json()["items"]
        self.assertEqual(
            [item["event_id"] for item in tuples[0]["proposals"]],
            ["e1", "e2"],
        )

        response = self.client.patch(
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={
                "expected_revision": 0,
                "patch": {"ranking": {"gap_lambda": 0.02}},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["session_revision"], 1)
        self.assertEqual(response.json()["invalidated_stages"], ["tuple"])
        self.assertGreater(response.json()["artifact_counts"]["tuples"], 0)

        stale = self.client.patch(
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={
                "expected_revision": 0,
                "patch": {"ranking": {"gap_lambda": 0.03}},
            },
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["detail"]["current_revision"], 1)

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
