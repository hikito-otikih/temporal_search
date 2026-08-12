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
    LiveRefinementDataError,
    LiveRefinementOrchestrator,
    LiveRefinementUnavailableError,
    _extract_region_frames,
    _region_end_ms,
    _region_start_ms,
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

    def plan_interval(self, start_pts_ms, end_pts_ms, interval_ms, *, max_frames):
        self.calls.append((start_pts_ms, end_pts_ms, interval_ms, max_frames))
        return _nominal_points(start_pts_ms, end_pts_ms, interval_ms, max_frames)

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


def seed_one_region(service, *, max_frames=24, timestamp_seconds=3.0):
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
        timestamp_seconds=timestamp_seconds,
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


class BoundaryOvershootFrameProvider(FakeFrameProvider):
    """Mimics nearest-frame decoding snapping a few ms past a boundary.

    Real video frames rarely land exactly on a region's start/end timestamp
    (they sit on the container's own frame grid), so the decoder's "nearest"
    frame is often a handful of ms outside the requested window.
    """

    def __init__(self):
        super().__init__()
        self._starts: set[int] = set()
        self._ends: set[int] = set()

    def plan_interval(self, start_pts_ms, end_pts_ms, interval_ms, *, max_frames):
        self._starts.add(start_pts_ms)
        self._ends.add(end_pts_ms)
        return super().plan_interval(
            start_pts_ms, end_pts_ms, interval_ms, max_frames=max_frames
        )

    def get_frames(self, video_id, pts_ms):
        shifted = [
            pts - 5 if pts in self._starts
            else pts + 5 if pts in self._ends
            else pts
            for pts in pts_ms
        ]
        return [self._frame(video_id, pts) for pts in shifted]


class AlwaysEmptyFrameProvider(FakeFrameProvider):
    """Simulates a region whose video/timestamps yield no decodable frame."""

    def get_frames(self, video_id, pts_ms):
        return []


class PartiallyFailingFrameProvider(FakeFrameProvider):
    def __init__(self, failing_video_id):
        super().__init__()
        self.failing_video_id = failing_video_id

    def get_frames(self, video_id, pts_ms):
        if video_id == self.failing_video_id:
            return []
        return super().get_frames(video_id, pts_ms)


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
            for start, end, interval, limit in provider.calls
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

    def test_extract_region_frames_clamps_a_frame_that_overshoots_the_boundary_slightly(self):
        service = AdaptiveSearchService(frame_provider=FakeFrameProvider())
        bundle = seed_one_region(service, max_frames=24, timestamp_seconds=60.0)
        region = bundle.artifacts.regions[0]
        start_ms, end_ms = _region_start_ms(region), _region_end_ms(region)
        targets = [start_ms, end_ms]
        pool = [
            FakeFrameProvider._frame(region.video_id, start_ms - 5),
            FakeFrameProvider._frame(region.video_id, end_ms + 5),
        ]

        frames, stats = _extract_region_frames(pool, region, targets)

        self.assertEqual(len(frames), 2)
        self.assertEqual(stats.out_of_region_frames, 0)
        self.assertTrue(all(start_ms <= frame.pts_ms <= end_ms for frame in frames))

    def test_extract_region_frames_still_drops_frames_far_outside_the_boundary(self):
        service = AdaptiveSearchService(frame_provider=FakeFrameProvider())
        bundle = seed_one_region(service, max_frames=24, timestamp_seconds=60.0)
        region = bundle.artifacts.regions[0]
        start_ms = _region_start_ms(region)
        targets = [start_ms]
        pool = [FakeFrameProvider._frame(region.video_id, start_ms - 5000)]

        frames, stats = _extract_region_frames(pool, region, targets)

        self.assertEqual(frames, [])
        self.assertEqual(stats.out_of_region_frames, 1)

    def test_a_regions_bad_frame_data_does_not_abort_the_rest_of_the_frontier(self):
        provider = PartiallyFailingFrameProvider(failing_video_id="video-b")
        service = AdaptiveSearchService(frame_provider=provider)
        bundle = service.create_session(
            events=[event("e1"), event("e2")],
            hyperparameters=configured_parameters(max_frames=10),
        )
        candidates = [
            SparseCandidate(
                id="c1",
                session_id=bundle.session.id,
                event_id="e1",
                video_id="video-a",
                frame_id=100,
                timestamp_seconds=4.0,
                raw_relevance_score=0.9,
                query_variant="anchor-en-0",
            ),
            SparseCandidate(
                id="c2",
                session_id=bundle.session.id,
                event_id="e2",
                video_id="video-b",
                frame_id=200,
                timestamp_seconds=8.0,
                raw_relevance_score=0.9,
                query_variant="anchor-en-0",
            ),
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

        good_region = next(
            r for r in bundle.artifacts.regions if r.video_id == "video-a"
        )
        bad_region = next(
            r for r in bundle.artifacts.regions if r.video_id == "video-b"
        )
        refined_region_ids = {
            sample.region_id for sample in result.bundle.artifacts.frame_scores
        }
        self.assertIn(good_region.id, refined_region_ids)
        self.assertNotIn(bad_region.id, refined_region_ids)
        self.assertEqual(result.metrics["regions_refined"], 1)
        self.assertEqual(result.metrics["regions_failed"], 1)

    def test_refine_raises_when_every_selected_region_fails(self):
        provider = AlwaysEmptyFrameProvider()
        service = AdaptiveSearchService(frame_provider=provider)
        bundle = seed_one_region(service, max_frames=24)
        region = bundle.artifacts.regions[0]

        with self.assertRaisesRegex(
            LiveRefinementDataError, "failed for all 1 selected regions"
        ):
            LiveRefinementOrchestrator(service, FakeEmbedder()).refine_session(
                bundle.session.id,
                expected_revision=0,
                region_ids=[region.id],
                max_frames=24,
            )


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
