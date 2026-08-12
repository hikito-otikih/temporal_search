import unittest

import numpy as np
from fastapi.testclient import TestClient

from adaptive_search.dependencies import (
    adaptive_service,
    live_refinement_orchestrator,
)
from adaptive_search.providers import (
    FrameDecodeError,
    FrameProviderCapabilities,
    FrameReference,
    UnavailableFrameProvider,
)
from main import app


MODEL_SPEC = {
    "model_id": "fake/siglip",
    "revision": "commit-api-test",
    "dimension": 3,
    "preprocess": "fake-rgb-v1",
    "instruction": "",
}


class FakeFrameProvider:
    def capabilities(self):
        return FrameProviderCapabilities(
            available=True,
            supports_batch=True,
            supports_dense_sampling=True,
            supports_thumbnails=False,
        )

    def get_frames(self, video_id, pts_ms):
        return [self._frame(video_id, int(pts)) for pts in pts_ms]

    def plan_interval(self, start_pts_ms, end_pts_ms, interval_ms, *, max_frames):
        count = min(max_frames, 4)
        points = np.linspace(start_pts_ms, end_pts_ms, count, dtype=int)
        return [int(pts) for pts in points]

    def sample_interval(
        self,
        video_id,
        start_pts_ms,
        end_pts_ms,
        interval_ms,
        *,
        max_frames,
    ):
        targets = self.plan_interval(
            start_pts_ms, end_pts_ms, interval_ms, max_frames=max_frames
        )
        return self.get_frames(video_id, targets)

    @staticmethod
    def _frame(video_id, pts_ms):
        phase = (pts_ms // 1000) % 3
        return FrameReference(
            video_id=video_id,
            pts_ms=pts_ms,
            frame_index=pts_ms // 40,
            image=np.eye(3, dtype=np.float32)[phase],
        )


class NoPixelFrameProvider(FakeFrameProvider):
    @staticmethod
    def _frame(video_id, pts_ms):
        return FrameReference(
            video_id=video_id,
            pts_ms=pts_ms,
            frame_index=pts_ms // 40,
            image=None,
        )


class FailingFrameProvider(FakeFrameProvider):
    def get_frames(self, video_id, pts_ms):
        raise FrameDecodeError("corrupt video stream")


class FakeEmbedder:
    model_id = MODEL_SPEC["model_id"]
    revision = MODEL_SPEC["revision"]
    output_dimension = MODEL_SPEC["dimension"]
    preprocess_version = MODEL_SPEC["preprocess"]
    instruction = MODEL_SPEC["instruction"]

    def encode_texts(self, texts):
        assert len(texts) == 3
        return np.eye(3, dtype=np.float32)

    def encode_images(self, images):
        return np.asarray(images, dtype=np.float32)


class FailingEmbedder(FakeEmbedder):
    def encode_texts(self, texts):
        raise RuntimeError("simulated CUDA out of memory")


class AdaptiveRefineApiTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.original_provider = adaptive_service.frame_provider
        self.original_embedder = live_refinement_orchestrator.embedder
        adaptive_service.frame_provider = FakeFrameProvider()
        live_refinement_orchestrator.embedder = FakeEmbedder()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        adaptive_service.repository.clear()
        adaptive_service.frame_provider = self.original_provider
        live_refinement_orchestrator.embedder = self.original_embedder

    def create_seeded_session(self, *, interval_seconds=0.5):
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "person changes state",
                        "anchor_query": "target state",
                        "pre_state": "before state",
                        "post_state": "after state",
                        "boundary_type": "transition",
                    }
                ],
                "hyperparameters": {
                    "refinement": {
                        "embedding_model": MODEL_SPEC,
                        "max_frames_per_run": 12,
                        "medium_interval_seconds": interval_seconds,
                        "dense_interval_seconds": min(0.1, interval_seconds),
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        session_id = response.json()["session"]["id"]
        response = self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={
                "expected_revision": 0,
                "event_ids": ["e1"],
                "candidates": [
                    {
                        "id": "c1",
                        "session_id": session_id,
                        "event_id": "e1",
                        "video_id": "video-a",
                        "frame_id": 100,
                        "timestamp_seconds": 4.0,
                        "raw_relevance_score": 0.9,
                        "query_variant": "anchor-en-0",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        regions = self.client.get(
            f"/v1/search-sessions/{session_id}/regions"
        ).json()["items"]
        self.assertEqual(len(regions), 1)
        return session_id, regions[0]["id"]

    def test_refine_command_persists_metrics_and_completes_region(self):
        session_id, region_id = self.create_seeded_session()

        capability = self.client.get("/v1/searchers").json()["searchers"][2]
        self.assertTrue(capability["live_refinement_available"])
        self.assertTrue(capability["frame_provider"]["available"])
        self.assertTrue(capability["live_refinement"]["model_available"])

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={
                "expected_revision": 0,
                "region_ids": [region_id],
                "max_frames": 6,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["metrics"]["live_refinement"])
        self.assertFalse(response.json()["metrics"]["motion_scores_available"])
        self.assertEqual(response.json()["metrics"]["embedding_batch_size"], 64)
        self.assertLessEqual(response.json()["metrics"]["frames_sampled"], 6)
        self.assertGreater(response.json()["artifact_counts"]["frame_scores"], 0)

        regions = self.client.get(
            f"/v1/search-sessions/{session_id}/regions"
        ).json()["items"]
        self.assertEqual(regions[0]["refinement_status"], "completed")
        runs = self.client.get(
            f"/v1/search-sessions/{session_id}/runs"
        ).json()["items"]
        self.assertTrue(runs[-1]["metrics"]["live_refinement"])
        self.assertFalse(runs[-1]["metrics"]["motion_scores_available"])
        self.assertIn("frames_sampled", runs[-1]["metrics"])

    def test_minimal_session_inherits_exact_configured_runtime_spec(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "target",
                        "anchor_query": "target",
                        "boundary_type": "state",
                    }
                ],
                "hyperparameters": {
                    "refinement": {"max_frames_per_run": 12}
                },
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        stored_spec = body["session"]["hyperparameters"]["refinement"][
            "embedding_model"
        ]
        self.assertEqual(stored_spec, MODEL_SPEC)
        self.assertTrue(body["live_refinement"]["available"])

        adaptive = self.client.get("/v1/searchers").json()["searchers"][2]
        self.assertEqual(adaptive["runtime_model_spec"], MODEL_SPEC)

    def test_explicit_model_spec_is_preserved_and_mismatch_is_reported(self):
        explicit_spec = {**MODEL_SPEC, "revision": "different-pinned-revision"}
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "target",
                        "anchor_query": "target",
                        "boundary_type": "state",
                    }
                ],
                "hyperparameters": {
                    "refinement": {"embedding_model": explicit_spec}
                },
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(
            body["session"]["hyperparameters"]["refinement"]["embedding_model"],
            explicit_spec,
        )
        self.assertFalse(body["live_refinement"]["available"])
        self.assertIn("does not match", body["live_refinement"]["reason"])

    def test_unimplemented_reranker_flag_is_rejected(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "target",
                        "anchor_query": "target",
                        "boundary_type": "state",
                    }
                ],
                "hyperparameters": {
                    "refinement": {"use_reranker": True}
                },
            },
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("not supported", response.text)

    def test_server_side_frame_and_embedding_batch_caps_are_strict(self):
        for refinement in (
            {"max_frames_per_run": 4097},
            {"embedding_batch_size": 513},
        ):
            with self.subTest(refinement=refinement):
                response = self.client.post(
                    "/v1/search-sessions",
                    json={
                        "events": [
                            {
                                "event_id": "e1",
                                "original_query": "target",
                                "anchor_query": "target",
                                "boundary_type": "state",
                            }
                        ],
                        "hyperparameters": {"refinement": refinement},
                    },
                )
                self.assertEqual(response.status_code, 422, response.text)

    def test_default_unavailable_runtime_returns_503(self):
        session_id, region_id = self.create_seeded_session()
        adaptive_service.frame_provider = UnavailableFrameProvider("decoder missing")
        live_refinement_orchestrator.embedder = None

        capability = self.client.get("/v1/searchers").json()["searchers"][2]
        self.assertFalse(capability["live_refinement_available"])
        self.assertFalse(capability["frame_provider"]["available"])
        self.assertFalse(capability["live_refinement"]["model_available"])

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={"expected_revision": 0, "region_ids": [region_id]},
        )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json()["detail"]["code"],
            "live_refinement_unavailable",
        )

    def test_provider_contract_error_returns_422(self):
        session_id, region_id = self.create_seeded_session()
        adaptive_service.frame_provider = NoPixelFrameProvider()

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={"expected_revision": 0, "region_ids": [region_id]},
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(
            response.json()["detail"]["code"],
            "live_refinement_data_error",
        )

    def test_provider_runtime_error_returns_422(self):
        session_id, region_id = self.create_seeded_session()
        adaptive_service.frame_provider = FailingFrameProvider()

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={"expected_revision": 0, "region_ids": [region_id]},
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(
            response.json()["detail"]["code"],
            "live_refinement_data_error",
        )

    def test_embedding_runtime_failure_returns_503(self):
        session_id, region_id = self.create_seeded_session()
        live_refinement_orchestrator.embedder = FailingEmbedder()

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={"expected_revision": 0, "region_ids": [region_id]},
        )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json()["detail"]["code"],
            "live_refinement_unavailable",
        )

    def test_sub_millisecond_sampling_configuration_returns_422(self):
        session_id, region_id = self.create_seeded_session(
            interval_seconds=0.0001
        )

        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={"expected_revision": 0, "region_ids": [region_id]},
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(
            response.json()["detail"]["code"],
            "live_refinement_configuration_error",
        )

    def test_refine_request_is_strict(self):
        session_id, _ = self.create_seeded_session()
        response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/refine",
            json={"expected_revision": 0, "unexpected": True},
        )
        self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
