"""HTTP-level tests for POST /temporal-search?apply_boundary_refinement=.

legacy_search has no session/region concept of its own - each returned
ClusteredCandidate already is the one frame TemporalSearcher/AmbiguousSearcher
picked for its query, so refinement here reuses adaptive_search's runtime and
refine_event_boundary() directly (no boundary_seeds involved - see
service.py's _refine_tuple_candidates docstring).
"""

import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from adaptive_search.dependencies import boundary_refinement_runtime
from adaptive_search.embedding import l2_normalize
from adaptive_search.providers import FrameProviderCapabilities, FrameReference
from legacy_search.schemas import Candidate, QueryResponse
from main import app


class FakeFrameProvider:
    def __init__(self, known_video_ids=("video_full",)):
        self._known = set(known_video_ids)

    def capabilities(self):
        return FrameProviderCapabilities(
            available=True, supports_batch=True, supports_dense_sampling=True, supports_thumbnails=False
        )

    def get_frames(self, video_id, pts_ms):
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


def _query_response(query, video_name):
    return QueryResponse(
        query=query,
        results=[
            Candidate(
                frame_index=10,
                timestamp="00:10",
                score=0.9,
                video_name=video_name,
                video_title="title",
                author="author",
                watch_url="https://example.invalid/watch",
            ),
            Candidate(
                frame_index=20,
                timestamp="00:20",
                score=0.5,
                video_name=video_name,
                video_title="title",
                author="author",
                watch_url="https://example.invalid/watch",
            ),
        ],
    )


class LegacyBoundaryRefinementTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self._original_provider = boundary_refinement_runtime.frame_provider
        self._original_embedder = boundary_refinement_runtime.embedder

    def tearDown(self):
        self.client.close()
        boundary_refinement_runtime.frame_provider = self._original_provider
        boundary_refinement_runtime.embedder = self._original_embedder

    def _search(self, video_name="video_full.mp4", **payload_overrides):
        payload = {"query": ["cook pours oil"], "top_k_tuple": 10, "top_k_each_query": 10}
        payload.update(payload_overrides)
        with patch(
            "legacy_search.service.search_queries",
            return_value=[_query_response("cook pours oil", video_name)],
        ):
            return self.client.post("/temporal-search", json=payload)

    def test_flag_off_marks_every_candidate_not_requested(self):
        response = self._search(apply_boundary_refinement=False)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertFalse(body["boundary_refinement_capability"]["requested"])
        self.assertFalse(body["boundary_refinement_capability"]["available"])
        for result in body["results"]:
            for candidate in result["tuple"]:
                self.assertEqual(candidate["boundary_refinement_status"], "not_requested")
                self.assertIsNone(candidate["refined_timestamp_seconds"])

    def test_flag_defaults_to_off_when_omitted(self):
        response = self._search()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["boundary_refinement_capability"]["requested"])

    def test_flag_on_without_configured_runtime_skips_gracefully(self):
        response = self._search(apply_boundary_refinement=True)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["boundary_refinement_capability"]["requested"])
        self.assertFalse(body["boundary_refinement_capability"]["available"])
        for result in body["results"]:
            for candidate in result["tuple"]:
                self.assertEqual(candidate["boundary_refinement_status"], "skipped_runtime_unavailable")

    def test_flag_on_with_configured_runtime_applies_refinement(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=["video_full"])
        boundary_refinement_runtime.embedder = FakeEmbedder()

        response = self._search(video_name="video_full.mp4", apply_boundary_refinement=True)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["boundary_refinement_capability"]["available"])
        candidates = body["results"][0]["tuple"]
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertEqual(candidate["boundary_refinement_status"], "applied")
            self.assertIsInstance(candidate["refined_timestamp_seconds"], float)

    def test_video_name_that_cannot_be_normalized_is_skipped_without_failing_the_request(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=["video_full"])
        boundary_refinement_runtime.embedder = FakeEmbedder()

        response = self._search(video_name="", apply_boundary_refinement=True)
        self.assertEqual(response.status_code, 200, response.text)
        candidates = response.json()["results"][0]["tuple"]
        for candidate in candidates:
            self.assertEqual(candidate["boundary_refinement_status"], "skipped_video_not_in_catalog")

    def test_metadata_miss_is_skipped_without_failing_the_request(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=[])
        boundary_refinement_runtime.embedder = FakeEmbedder()

        response = self._search(video_name="video_full.mp4", apply_boundary_refinement=True)
        self.assertEqual(response.status_code, 200, response.text)
        candidates = response.json()["results"][0]["tuple"]
        for candidate in candidates:
            self.assertEqual(candidate["boundary_refinement_status"], "skipped_no_metadata")


if __name__ == "__main__":
    unittest.main()
