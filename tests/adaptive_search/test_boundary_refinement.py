import unittest
from unittest.mock import patch

import numpy as np

from adaptive_search.embedding import l2_normalize
from adaptive_search import boundary_refinement as boundary_refinement_module
from adaptive_search.boundary_refinement import refine_event_boundary
from adaptive_search.providers import FrameProviderCapabilities, FrameReference
from adaptive_search.schemas import BoundaryHyperparameters, ModelRuntimeSpec


MODEL_SPEC = ModelRuntimeSpec(
    model_id="fake/siglip",
    revision="commit-abc123",
    dimension=3,
    preprocess="fake-rgb-v1",
    instruction="",
)


class FakeFrameProvider:
    """Ported from the now-removed test_adaptive_live_refinement.py fixture."""

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return FrameProviderCapabilities(
            available=True,
            supports_batch=True,
            supports_dense_sampling=True,
            supports_thumbnails=False,
        )

    def get_frames(self, video_id, pts_ms):
        self.calls.append(list(pts_ms))
        return [self._frame(video_id, pts) for pts in pts_ms]

    @staticmethod
    def _frame(video_id, pts_ms):
        phase = (pts_ms // 1000) % 3
        image = np.eye(3, dtype=np.float32)[phase]
        return FrameReference(
            video_id=video_id, pts_ms=pts_ms, frame_index=pts_ms // 40, image=image
        )


class EmptyFrameProvider:
    def capabilities(self):
        return FrameProviderCapabilities(
            available=True, supports_batch=True, supports_dense_sampling=True, supports_thumbnails=False
        )

    def get_frames(self, video_id, pts_ms):
        return []


class FakeEmbedder:
    model_id = MODEL_SPEC.model_id
    revision = MODEL_SPEC.revision
    output_dimension = MODEL_SPEC.dimension
    preprocess_version = MODEL_SPEC.preprocess
    instruction = MODEL_SPEC.instruction

    def encode_texts(self, texts):
        self.texts = list(texts)
        return np.eye(3, dtype=np.float32)

    def encode_images(self, images):
        return l2_normalize(np.asarray(images, dtype=np.float32))


class RefineEventBoundaryTests(unittest.TestCase):
    def test_transition_profile_stays_within_the_sweep_window(self) -> None:
        provider = FakeFrameProvider()
        outcome = refine_event_boundary(
            provider=provider,
            embedder=FakeEmbedder(),
            video_id="video-a",
            event_id="e1",
            anchor_query="cook pours oil",
            pre_state="oil has not been poured yet",
            post_state="oil is being poured",
            boundary_type="onset",
            anchor_seconds=10.0,
            fps=30.0,
            radius_frames=10,
        )
        self.assertFalse(outcome.used_fallback)
        self.assertGreater(outcome.sampled_frame_count, 0)
        # +/-10 native frames at 30fps is +/-0.333s.
        self.assertAlmostEqual(outcome.refined_seconds, 10.0, delta=0.5)

    def test_state_profile_with_no_pre_post_state_does_not_crash_or_degenerate(self) -> None:
        # pre_state=None/post_state=None + boundary_type="unknown" must dispatch
        # to generate_profiled_proposals()'s "state" profile, which never reads
        # raw_pre_score/raw_post_score at all - this is the fix for the
        # pairwise_softmax(x, x) == (0.5, 0.5) degeneracy that a naive
        # identical-text fallback into the transition profile would hit.
        provider = FakeFrameProvider()
        outcome = refine_event_boundary(
            provider=provider,
            embedder=FakeEmbedder(),
            video_id="video-a",
            event_id="e1",
            anchor_query="cut onion",
            pre_state=None,
            post_state=None,
            boundary_type="unknown",
            anchor_seconds=5.0,
            fps=30.0,
            radius_frames=10,
        )
        self.assertFalse(outcome.used_fallback)
        self.assertIsNotNone(outcome.top_proposal)
        self.assertEqual(outcome.top_proposal.raw_boundary_score, 0.0)

    def test_no_frames_returned_falls_back_to_the_original_anchor(self) -> None:
        outcome = refine_event_boundary(
            provider=EmptyFrameProvider(),
            embedder=FakeEmbedder(),
            video_id="video-a",
            event_id="e1",
            anchor_query="cut onion",
            pre_state="before",
            post_state="after",
            boundary_type="onset",
            anchor_seconds=5.0,
            fps=30.0,
        )
        self.assertTrue(outcome.used_fallback)
        self.assertEqual(outcome.refined_seconds, 5.0)
        self.assertEqual(outcome.sampled_frame_count, 0)
        self.assertIsNone(outcome.top_proposal)

    def test_stride_widens_native_frame_coverage_without_changing_sample_count(self) -> None:
        # frame count = 2*radius_frames+1 regardless of stride - only the time
        # span covered changes.
        dense_provider = FakeFrameProvider()
        refine_event_boundary(
            provider=dense_provider,
            embedder=FakeEmbedder(),
            video_id="video-a",
            event_id="e1",
            anchor_query="cut onion",
            pre_state="before",
            post_state="after",
            boundary_type="onset",
            anchor_seconds=10.0,
            fps=30.0,
            radius_frames=10,
            stride=1,
        )
        sparse_provider = FakeFrameProvider()
        refine_event_boundary(
            provider=sparse_provider,
            embedder=FakeEmbedder(),
            video_id="video-a",
            event_id="e1",
            anchor_query="cut onion",
            pre_state="before",
            post_state="after",
            boundary_type="onset",
            anchor_seconds=10.0,
            fps=30.0,
            radius_frames=10,
            stride=5,
        )
        dense_targets = dense_provider.calls[0]
        sparse_targets = sparse_provider.calls[0]
        self.assertEqual(len(dense_targets), len(sparse_targets))
        self.assertLess(max(dense_targets) - min(dense_targets), max(sparse_targets) - min(sparse_targets))

    def test_default_window_options_seconds_is_still_natively_derived(self) -> None:
        with patch(
            "adaptive_search.boundary_refinement.generate_profiled_proposals",
            wraps=boundary_refinement_module.generate_profiled_proposals,
        ) as spy:
            refine_event_boundary(
                provider=FakeFrameProvider(),
                embedder=FakeEmbedder(),
                video_id="video-a",
                event_id="e1",
                anchor_query="cut onion",
                pre_state="before",
                post_state="after",
                boundary_type="onset",
                anchor_seconds=10.0,
                fps=30.0,
                radius_frames=10,
                stride=1,
                parameters=BoundaryHyperparameters(),  # untouched schema default
            )
        used_parameters = spy.call_args.args[2]
        self.assertNotEqual(used_parameters.window_options_seconds, (0.5, 1.0, 2.0, 3.0))

    def test_explicitly_customized_window_options_seconds_is_honored(self) -> None:
        # Regression test for a real reported bug: window_options_seconds is
        # a real, publicly-patchable BoundaryHyperparameters field (the
        # youcook2 benchmark's own boundary_refinement.py constructs it
        # directly), but live refinement always silently overwrote it with
        # native fps/stride-derived values regardless of what a caller
        # configured. A value that differs from the schema default must now
        # survive unchanged.
        custom_windows = (0.7, 1.3, 5.0)
        with patch(
            "adaptive_search.boundary_refinement.generate_profiled_proposals",
            wraps=boundary_refinement_module.generate_profiled_proposals,
        ) as spy:
            refine_event_boundary(
                provider=FakeFrameProvider(),
                embedder=FakeEmbedder(),
                video_id="video-a",
                event_id="e1",
                anchor_query="cut onion",
                pre_state="before",
                post_state="after",
                boundary_type="onset",
                anchor_seconds=10.0,
                fps=30.0,
                radius_frames=10,
                stride=1,
                parameters=BoundaryHyperparameters(window_options_seconds=custom_windows),
            )
        used_parameters = spy.call_args.args[2]
        self.assertEqual(used_parameters.window_options_seconds, custom_windows)


if __name__ == "__main__":
    unittest.main()
