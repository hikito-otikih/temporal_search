import unittest

from adaptive_search.exceptions import AdaptiveInputError
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

    def test_frame_score_acceptance_window_capped_at_neighboring_event(self):
        # e1's candidate at t=100, e2's candidate at t=104 - a real,
        # observed pattern (region_tuple_ranking_results.md: 25% of
        # top-ranked videos have two different events within seconds of
        # each other). Default margin=3.0 would naively accept up to
        # t=103 for e1's region, reaching into territory closer to e2's
        # own candidate - the guard caps it at the midpoint (t=102)
        # instead.
        service = AdaptiveSearchService()
        bundle = service.create_session(
            events=[
                EventDefinition(
                    event_id="e1", original_query="chop onions", anchor_query="chopping onions",
                    pre_state="onion whole", post_state="onion chopped", boundary_type="transition",
                ),
                EventDefinition(
                    event_id="e2", original_query="fry onions", anchor_query="frying onions",
                    pre_state="onion raw", post_state="onion cooked", boundary_type="transition",
                ),
            ]
        )
        session_id = bundle.session.id
        bundle, _ = service.replace_candidates(
            session_id, expected_revision=0, event_ids=["e1", "e2"],
            candidates=[
                SparseCandidate(
                    id="c1", session_id=session_id, event_id="e1", video_id="video",
                    frame_id=1000, timestamp_seconds=100.0, raw_relevance_score=1.0,
                    query_variant="variant-1",
                ),
                SparseCandidate(
                    id="c2", session_id=session_id, event_id="e2", video_id="video",
                    frame_id=1040, timestamp_seconds=104.0, raw_relevance_score=1.0,
                    query_variant="variant-1",
                ),
            ],
        )
        e1_region = next(r for r in bundle.artifacts.regions if r.event_id == "e1")

        accepted_sample = FrameScoreSample(
            session_id=session_id, event_id="e1", video_id="video", region_id=e1_region.id,
            frame_id=1015, timestamp_seconds=101.5,
            raw_anchor_score=1.0, raw_pre_score=1.0, raw_post_score=-1.0,
        )
        bundle, _ = service.replace_frame_scores(
            session_id, expected_revision=0, region_ids=[e1_region.id], samples=[accepted_sample],
        )
        self.assertEqual(len(bundle.artifacts.frame_scores), 1)

        rejected_sample = FrameScoreSample(
            session_id=session_id, event_id="e1", video_id="video", region_id=e1_region.id,
            frame_id=1025, timestamp_seconds=102.5,
            raw_anchor_score=1.0, raw_pre_score=1.0, raw_post_score=-1.0,
        )
        with self.assertRaisesRegex(AdaptiveInputError, "outside its region"):
            service.replace_frame_scores(
                session_id, expected_revision=0, region_ids=[e1_region.id], samples=[rejected_sample],
            )

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
