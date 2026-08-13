"""HTTP-level tests for GET /video-priorities?apply_boundary_refinement=."""

import unittest

import numpy as np
from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service, boundary_refinement_runtime
from adaptive_search.embedding import l2_normalize
from adaptive_search.providers import FrameProviderCapabilities, FrameReference
from main import app


class FakeFrameProvider:
    def __init__(self, known_video_ids=("video_full",)):
        self._known = set(known_video_ids)
        self.calls = []

    def capabilities(self):
        return FrameProviderCapabilities(
            available=True, supports_batch=True, supports_dense_sampling=True, supports_thumbnails=False
        )

    def get_frames(self, video_id, pts_ms):
        self.calls.append((video_id, list(pts_ms)))
        return [self._frame(video_id, pts) for pts in pts_ms]

    @staticmethod
    def _frame(video_id, pts_ms):
        phase = (pts_ms // 1000) % 3
        image = np.eye(3, dtype=np.float32)[phase]
        return FrameReference(video_id=video_id, pts_ms=pts_ms, frame_index=pts_ms // 40, image=image)

    @property
    def catalog(self):
        return _FakeCatalog(self._known)


class _FakeMetadata:
    def __init__(self, fps=30.0, duration_ms=600_000):
        self.fps = fps
        self.duration_ms = duration_ms


class _FakeCatalog:
    def __init__(self, known_video_ids):
        self._known = known_video_ids

    def metadata(self, video_id):
        return _FakeMetadata() if video_id in self._known else None


class FakeEmbedder:
    model_id = "fake/siglip"
    revision = "commit-abc123"
    output_dimension = 3
    preprocess_version = "fake-rgb-v1"
    instruction = ""

    def encode_texts(self, texts):
        return np.eye(3, dtype=np.float32)

    def encode_images(self, images):
        return l2_normalize(np.asarray(images, dtype=np.float32))


def event(event_id, text):
    return {"event_id": event_id, "original_query": text, "anchor_query": text}


class VideoPrioritiesBoundaryRefinementTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)
        self._original_provider = boundary_refinement_runtime.frame_provider
        self._original_embedder = boundary_refinement_runtime.embedder

    def tearDown(self):
        self.client.close()
        boundary_refinement_runtime.frame_provider = self._original_provider
        boundary_refinement_runtime.embedder = self._original_embedder

    def _create_session_with_one_candidate(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={"events": [event("e1", "cook pours oil")]},
        )
        session_id = response.json()["session"]["id"]
        candidates = [
            {
                "id": "c1",
                "session_id": session_id,
                "event_id": "e1",
                "video_id": "video_full",
                "frame_id": 300,
                "timestamp_seconds": 10.0,
                "raw_relevance_score": 0.9,
                "query_variant": "anchor-en-0",
            }
        ]
        response = self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={"expected_revision": 0, "event_ids": ["e1"], "candidates": candidates},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return session_id

    def test_default_is_on_and_reports_capability_unavailable_without_a_configured_runtime(self):
        session_id = self._create_session_with_one_candidate()
        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["boundary_refinement_capability"]["requested"])
        self.assertFalse(body["boundary_refinement_capability"]["available"])
        self.assertEqual(
            body["items"][0]["boundary_refinement"]["status"], "skipped_runtime_unavailable"
        )
        self.assertIsNone(body["items"][0]["boundary_refinement"]["events"])

    def test_flag_off_skips_refinement_entirely(self):
        session_id = self._create_session_with_one_candidate()
        response = self.client.get(
            f"/v1/search-sessions/{session_id}/video-priorities",
            params={"apply_boundary_refinement": "false"},
        )
        body = response.json()
        self.assertFalse(body["boundary_refinement_capability"]["requested"])
        self.assertEqual(body["items"][0]["boundary_refinement"]["status"], "not_requested")

    def test_applied_when_runtime_is_configured(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=["video_full"])
        boundary_refinement_runtime.embedder = FakeEmbedder()
        session_id = self._create_session_with_one_candidate()

        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["boundary_refinement_capability"]["available"])
        item = body["items"][0]
        self.assertEqual(item["boundary_refinement"]["status"], "applied")
        events = item["boundary_refinement"]["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], "e1")
        self.assertEqual(events[0]["anchor_seconds"], 10.0)
        self.assertIsInstance(events[0]["refined_seconds"], float)

    def test_per_video_metadata_miss_is_skipped_without_failing_the_request(self):
        # video_full has no known metadata in this fake catalog -> that one
        # item degrades gracefully; the request as a whole still succeeds.
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=[])
        boundary_refinement_runtime.embedder = FakeEmbedder()
        session_id = self._create_session_with_one_candidate()

        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]
        self.assertEqual(item["boundary_refinement"]["status"], "skipped_no_metadata")


if __name__ == "__main__":
    unittest.main()
