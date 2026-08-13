"""HTTP-level tests for GET /v1/media/{video_id}/frame-preview."""

import base64
import io
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from adaptive_search.dependencies import boundary_refinement_runtime
from adaptive_search.providers import FrameProviderCapabilities, FrameReference, UnavailableFrameProvider
from main import app


class FakeFrameProvider:
    def __init__(self, known_video_ids=("video1",), fps=30.0, duration_ms=600_000):
        self._known = set(known_video_ids)
        self._fps = fps
        self._duration_ms = duration_ms
        self.calls = []

    def capabilities(self):
        return FrameProviderCapabilities(
            available=True, supports_batch=True, supports_dense_sampling=True, supports_thumbnails=False
        )

    def get_frames(self, video_id, pts_ms):
        self.calls.append(list(pts_ms))
        return [self._frame(video_id, pts) for pts in pts_ms]

    @staticmethod
    def _frame(video_id, pts_ms):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :] = [pts_ms % 256, 0, 0]
        return FrameReference(video_id=video_id, pts_ms=pts_ms, frame_index=pts_ms // 40, image=image)

    @property
    def catalog(self):
        return _FakeCatalog(self._known, self._fps, self._duration_ms)


class _FakeMetadata:
    def __init__(self, fps, duration_ms):
        self.fps = fps
        self.duration_ms = duration_ms


class _FakeCatalog:
    def __init__(self, known_video_ids, fps, duration_ms):
        self._known = known_video_ids
        self._fps = fps
        self._duration_ms = duration_ms

    def metadata(self, video_id):
        return _FakeMetadata(self._fps, self._duration_ms) if video_id in self._known else None


class NoFpsCatalogProvider(FakeFrameProvider):
    @property
    def catalog(self):
        return _FakeCatalog(self._known, None, None)


class FramePreviewTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self._original_provider = boundary_refinement_runtime.frame_provider
        self._original_embedder = boundary_refinement_runtime.embedder

    def tearDown(self):
        self.client.close()
        boundary_refinement_runtime.frame_provider = self._original_provider
        boundary_refinement_runtime.embedder = self._original_embedder

    def test_returns_expected_frame_count_and_span(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(fps=30.0)
        response = self.client.get(
            "/v1/media/video1/frame-preview",
            params={"anchor_seconds": 10.0, "radius_frames": 15, "stride": 1},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["video_id"], "video1")
        self.assertEqual(body["fps"], 30.0)
        # radius=15 -> 31 frames (offsets -15..+15 inclusive), no clamping
        # near duration since anchor is far from the edges.
        self.assertEqual(len(body["frames"]), 31)
        offsets = [frame["offset"] for frame in body["frames"]]
        self.assertEqual(offsets, list(range(-15, 16)))
        # +/-15 native frames at 30fps spans 1.0s total.
        seconds = [frame["timestamp_seconds"] for frame in body["frames"]]
        self.assertAlmostEqual(max(seconds) - min(seconds), 1.0, delta=0.05)

    def test_frames_are_decodable_jpeg_images(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(fps=30.0)
        response = self.client.get(
            "/v1/media/video1/frame-preview",
            params={"anchor_seconds": 5.0, "radius_frames": 2},
        )
        body = response.json()
        first = body["frames"][0]
        decoded = base64.b64decode(first["image_base64"])
        image = Image.open(io.BytesIO(decoded))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(image.size, (4, 4))

    def test_stride_widens_span_without_changing_frame_count(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(fps=30.0)
        response = self.client.get(
            "/v1/media/video1/frame-preview",
            params={"anchor_seconds": 10.0, "radius_frames": 15, "stride": 3},
        )
        body = response.json()
        self.assertEqual(len(body["frames"]), 31)
        seconds = [frame["timestamp_seconds"] for frame in body["frames"]]
        self.assertAlmostEqual(max(seconds) - min(seconds), 3.0, delta=0.05)

    def test_unconfigured_provider_returns_clean_error_not_500(self):
        boundary_refinement_runtime.frame_provider = UnavailableFrameProvider()
        response = self.client.get(
            "/v1/media/video1/frame-preview", params={"anchor_seconds": 5.0}
        )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["detail"]["code"], "live_refinement_unavailable")

    def test_unknown_video_id_returns_404_not_500(self):
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=["other_video"])
        response = self.client.get(
            "/v1/media/video1/frame-preview", params={"anchor_seconds": 5.0}
        )
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["detail"]["code"], "video_not_found")

    def test_video_without_fps_metadata_returns_404_not_500(self):
        boundary_refinement_runtime.frame_provider = NoFpsCatalogProvider()
        response = self.client.get(
            "/v1/media/video1/frame-preview", params={"anchor_seconds": 5.0}
        )
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["detail"]["code"], "video_not_found")


if __name__ == "__main__":
    unittest.main()
