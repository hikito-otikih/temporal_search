import unittest

from fastapi.testclient import TestClient

from adaptive_search.dependencies import (
    adaptive_service,
    live_refinement_orchestrator,
)
from adaptive_search.providers import UnavailableFrameProvider
from main import app
from tests.adaptive_search.test_adaptive_live_refinement import (
    FakeEmbedder,
    FakeFrameProvider,
    MODEL_SPEC,
)


EVENT = {
    "event_id": "e1",
    "original_query": "person changes state",
    "anchor_query": "target state",
    "pre_state": "before state",
    "post_state": "after state",
    "boundary_type": "transition",
}


class AdaptiveLiveApiTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.original_provider = adaptive_service.frame_provider
        self.original_embedder = live_refinement_orchestrator.embedder
        self.client = TestClient(app)

    def tearDown(self):
        adaptive_service.frame_provider = self.original_provider
        live_refinement_orchestrator.embedder = self.original_embedder
        adaptive_service.repository.clear()
        self.client.close()

    def _seed_session(self, *, configured_model: bool) -> tuple[str, str]:
        payload = {"events": [EVENT]}
        if configured_model:
            payload["hyperparameters"] = {
                "refinement": {
                    "embedding_model": MODEL_SPEC.model_dump(mode="json"),
                    "max_frames_per_run": 24,
                }
            }
        created = self.client.post("/v1/search-sessions", json=payload)
        self.assertEqual(created.status_code, 201, created.text)
        session_id = created.json()["session"]["id"]
        candidate = {
            "id": "c1",
            "session_id": session_id,
            "event_id": "e1",
            "video_id": "video",
            "frame_id": 50,
            "timestamp_seconds": 2.0,
            "raw_relevance_score": 0.9,
            "query_variant": "anchor-en",
        }
        ingested = self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={
                "expected_revision": 0,
                "event_ids": ["e1"],
                "candidates": [candidate],
            },
        )
        self.assertEqual(ingested.status_code, 200, ingested.text)
        regions = self.client.get(
            f"/v1/search-sessions/{session_id}/regions"
        ).json()["items"]
        self.assertEqual(len(regions), 1)
        return session_id, regions[0]["id"]

    def test_refine_endpoint_reports_combined_runtime_unavailable(self):
        adaptive_service.frame_provider = UnavailableFrameProvider("no test video")
        live_refinement_orchestrator.embedder = None
        session_id, region_id = self._seed_session(configured_model=False)

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={
                "expected_revision": 0,
                "region_ids": [region_id],
                "max_frames": 12,
            },
        )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json()["detail"]["code"],
            "live_refinement_unavailable",
        )

    def test_refine_endpoint_persists_metrics_and_completed_region(self):
        adaptive_service.frame_provider = FakeFrameProvider()
        live_refinement_orchestrator.embedder = FakeEmbedder()
        session_id, region_id = self._seed_session(configured_model=True)

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={
                "expected_revision": 0,
                "region_ids": [region_id],
                "max_frames": 12,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["metrics"]["live_refinement"])
        self.assertGreater(body["metrics"]["frames_sampled"], 0)
        region = self.client.get(
            f"/v1/search-sessions/{session_id}/regions"
        ).json()["items"][0]
        self.assertEqual(region["refinement_status"], "completed")
        runs = self.client.get(
            f"/v1/search-sessions/{session_id}/runs"
        ).json()["items"]
        self.assertTrue(runs[-1]["metrics"]["live_refinement"])


if __name__ == "__main__":
    unittest.main()
