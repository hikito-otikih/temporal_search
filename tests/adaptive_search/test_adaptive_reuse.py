import unittest

from adaptive_search.schemas import EventDefinition, FrameScoreSample, SparseCandidate
from adaptive_search.service import AdaptiveSearchService


class AdaptiveReuseTests(unittest.TestCase):
    def test_boundary_only_changes_rebuild_from_reusable_frame_scores(self):
        service = AdaptiveSearchService()
        bundle = service.create_session(
            events=[
                EventDefinition(
                    event_id="e1",
                    original_query="door opens",
                    anchor_query="an opening door",
                    pre_state="the door is closed",
                    post_state="the door is open",
                    boundary_type="transition",
                )
            ]
        )
        session_id = bundle.session.id
        bundle, _ = service.replace_candidates(
            session_id,
            expected_revision=0,
            event_ids=["e1"],
            candidates=[
                SparseCandidate(
                    id="candidate",
                    session_id=session_id,
                    event_id="e1",
                    video_id="video",
                    frame_id=20,
                    timestamp_seconds=2.0,
                    raw_relevance_score=1.0,
                    query_variant="variant-1",
                )
            ],
        )
        region = bundle.artifacts.regions[0]
        samples = [
            FrameScoreSample(
                session_id=session_id,
                event_id="e1",
                video_id="video",
                region_id=region.id,
                frame_id=index * 10,
                timestamp_seconds=float(index),
                raw_anchor_score=2.0 if index == 2 else 0.0,
                raw_pre_score=2.0 if index < 2 else -2.0,
                raw_post_score=-2.0 if index < 2 else 2.0,
            )
            for index in range(5)
        ]
        bundle, _ = service.replace_frame_scores(
            session_id,
            expected_revision=0,
            region_ids=[region.id],
            samples=samples,
        )
        frame_score_count = len(bundle.artifacts.frame_scores)
        self.assertGreater(len(bundle.artifacts.proposals), 0)

        bundle, invalidated = service.patch_event(
            session_id,
            "e1",
            expected_revision=0,
            patch={"boundary_type": "state"},
        )
        self.assertEqual(invalidated, ["proposal", "tuple"])
        self.assertEqual(len(bundle.artifacts.frame_scores), frame_score_count)
        self.assertGreater(len(bundle.artifacts.proposals), 0)

        bundle, invalidated, _ = service.patch_hyperparameters(
            session_id,
            expected_revision=1,
            patch={"boundary": {"nms_radius_seconds": 0.25}},
        )
        self.assertEqual(invalidated, ["proposal", "tuple"])
        self.assertEqual(len(bundle.artifacts.frame_scores), frame_score_count)
        self.assertGreater(len(bundle.artifacts.proposals), 0)

    def test_transition_profile_requires_observable_pre_and_post_states(self):
        with self.assertRaisesRegex(ValueError, "require both pre_state"):
            EventDefinition(
                event_id="e1",
                original_query="door opens",
                anchor_query="an opening door",
                boundary_type="transition",
            )


if __name__ == "__main__":
    unittest.main()
