import unittest

import numpy as np

from adaptive_search.embedding import l2_normalize
from adaptive_search.schemas import (
    EventDefinition,
    ModelRuntimeSpec,
    SearchHyperparameters,
    SparseCandidate,
)
from adaptive_search.providers import (
    FrameProviderCapabilities,
    FrameReference,
    UnavailableFrameProvider,
)
from adaptive_search.refinement import (
    LiveRefinementOrchestrator,
    LiveRefinementUnavailableError,
)
from adaptive_search.service import AdaptiveSearchService


MODEL_SPEC = ModelRuntimeSpec(
    model_id="fake/siglip",
    revision="commit-abc123",
    dimension=3,
    preprocess="fake-rgb-v1",
    instruction="",
)


class FakeFrameProvider:
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
        return [self._frame(video_id, pts) for pts in pts_ms]

    def sample_interval(
        self,
        video_id,
        start_pts_ms,
        end_pts_ms,
        interval_ms,
        *,
        max_frames,
    ):
        self.calls.append(
            (video_id, start_pts_ms, end_pts_ms, interval_ms, max_frames)
        )
        points = _nominal_points(
            start_pts_ms, end_pts_ms, interval_ms, max_frames
        )
        return [self._frame(video_id, pts) for pts in points]

    @staticmethod
    def _frame(video_id, pts_ms):
        phase = (pts_ms // 1000) % 3
        image = np.eye(3, dtype=np.float32)[phase]
        return FrameReference(
            video_id=video_id,
            pts_ms=pts_ms,
            frame_index=pts_ms // 40,
            image=image,
        )


class FakeEmbedder:
    model_id = MODEL_SPEC.model_id
    revision = MODEL_SPEC.revision
    output_dimension = MODEL_SPEC.dimension
    preprocess_version = MODEL_SPEC.preprocess
    instruction = MODEL_SPEC.instruction

    def __init__(self):
        self.image_count = 0

    def encode_texts(self, texts):
        self.texts = list(texts)
        return np.eye(3, dtype=np.float32)

    def encode_images(self, images):
        self.image_count += len(images)
        return l2_normalize(np.asarray(images, dtype=np.float32))


def event(event_id="e1"):
    return EventDefinition(
        event_id=event_id,
        original_query="person changes state",
        anchor_query="target state",
        pre_state="before state",
        post_state="after state",
        boundary_type="transition",
    )


def configured_parameters(max_frames=24):
    payload = SearchHyperparameters().model_dump()
    payload["refinement"]["embedding_model"] = MODEL_SPEC.model_dump()
    payload["refinement"]["max_frames_per_run"] = max_frames
    return SearchHyperparameters.model_validate(payload)


def seed_one_region(service, *, max_frames=24):
    bundle = service.create_session(
        events=[event()],
        hyperparameters=configured_parameters(max_frames),
    )
    candidate = SparseCandidate(
        id="c1",
        session_id=bundle.session.id,
        event_id="e1",
        video_id="video-a",
        frame_id=75,
        timestamp_seconds=3.0,
        raw_relevance_score=0.9,
        query_variant="anchor-en-0",
    )
    bundle, _ = service.replace_candidates(
        bundle.session.id,
        expected_revision=0,
        event_ids=["e1"],
        candidates=[candidate],
    )
    return bundle


class AdaptiveLiveRefinementTests(unittest.TestCase):
    def test_medium_dense_pipeline_preserves_pts_and_collects_metrics(self):
        provider = FakeFrameProvider()
        embedder = FakeEmbedder()
        service = AdaptiveSearchService(frame_provider=provider)
        bundle = seed_one_region(service, max_frames=12)
        region = bundle.artifacts.regions[0]

        result = LiveRefinementOrchestrator(service, embedder).refine_session(
            bundle.session.id,
            expected_revision=0,
            region_ids=[region.id],
            max_frames=12,
        )

        scores = result.bundle.artifacts.frame_scores
        self.assertGreater(len(scores), 1)
        self.assertLessEqual(len(scores), 12)
        self.assertEqual(
            [score.timestamp_seconds for score in scores],
            sorted(score.timestamp_seconds for score in scores),
        )
        expected_pts = {
            frame_pts
            for _, start, end, interval, limit in provider.calls
            for frame_pts in _nominal_points(start, end, interval, limit)
        }
        self.assertTrue(
            all(
                int(round(score.timestamp_seconds * 1000)) in expected_pts
                for score in scores
            )
        )
        self.assertTrue(result.metrics["live_refinement"])
        self.assertEqual(result.metrics["frames_sampled"], len(scores))
        self.assertEqual(
            result.metrics["refinement_model_revision"], MODEL_SPEC.revision
        )
        self.assertEqual(embedder.image_count, len(scores))
        self.assertIn("total_elapsed_ms", result.metrics)
        self.assertEqual(len(provider.calls), 2)

    def test_global_budget_is_shared_across_selected_regions(self):
        provider = FakeFrameProvider()
        service = AdaptiveSearchService(frame_provider=provider)
        bundle = service.create_session(
            events=[event("e1"), event("e2")],
            hyperparameters=configured_parameters(max_frames=10),
        )
        candidates = [
            SparseCandidate(
                id=f"c{index}",
                session_id=bundle.session.id,
                event_id=event_id,
                video_id="video-a",
                frame_id=index * 100,
                timestamp_seconds=float(index * 4),
                raw_relevance_score=0.9,
                query_variant="anchor-en-0",
            )
            for index, event_id in enumerate(("e1", "e2"), start=1)
        ]
        bundle, _ = service.replace_candidates(
            bundle.session.id,
            expected_revision=0,
            event_ids=["e1", "e2"],
            candidates=candidates,
        )

        result = LiveRefinementOrchestrator(
            service, FakeEmbedder()
        ).refine_session(
            bundle.session.id,
            expected_revision=0,
            region_ids=[region.id for region in bundle.artifacts.regions],
            max_frames=10,
        )

        by_region = {region.id: 0 for region in bundle.artifacts.regions}
        for sample in result.bundle.artifacts.frame_scores:
            by_region[sample.region_id] += 1
        self.assertTrue(all(count > 0 for count in by_region.values()))
        self.assertLessEqual(sum(by_region.values()), 10)
        self.assertEqual(result.metrics["regions_refined"], 2)

    def test_unavailable_provider_fails_before_model_inference(self):
        service = AdaptiveSearchService(
            frame_provider=UnavailableFrameProvider("decoder missing")
        )
        bundle = seed_one_region(service)
        embedder = FakeEmbedder()

        with self.assertRaisesRegex(
            LiveRefinementUnavailableError, "decoder missing"
        ):
            LiveRefinementOrchestrator(service, embedder).refine_session(
                bundle.session.id, expected_revision=0
            )
        self.assertEqual(embedder.image_count, 0)

    def test_unpinned_model_is_capability_gated_before_sampling(self):
        provider = FakeFrameProvider()
        service = AdaptiveSearchService(frame_provider=provider)
        bundle = service.create_session(events=[event()])
        candidate = SparseCandidate(
            id="c1",
            session_id=bundle.session.id,
            event_id="e1",
            video_id="video-a",
            frame_id=10,
            timestamp_seconds=1.0,
            raw_relevance_score=0.9,
            query_variant="anchor-en-0",
        )
        service.replace_candidates(
            bundle.session.id,
            expected_revision=0,
            event_ids=["e1"],
            candidates=[candidate],
        )

        with self.assertRaisesRegex(
            LiveRefinementUnavailableError,
            "session model revision must be pinned",
        ):
            LiveRefinementOrchestrator(service, FakeEmbedder()).refine_session(
                bundle.session.id, expected_revision=0
            )
        self.assertEqual(provider.calls, [])


def _nominal_points(start, end, interval, limit):
    if limit == 1:
        return [(start + end) // 2]
    natural = list(range(start, end + 1, interval))
    if not natural or natural[-1] != end:
        natural.append(end)
    if len(natural) <= limit:
        return natural
    indexes = np.linspace(0, len(natural) - 1, limit)
    return [natural[int(round(index))] for index in indexes]


if __name__ == "__main__":
    unittest.main()
