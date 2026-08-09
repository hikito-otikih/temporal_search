import unittest

from adaptive_search.algorithms import (
    assemble_ordered_tuples,
    calibrate_frame_scores,
    cluster_temporal_regions,
    generate_boundary_proposals,
    prioritize_videos,
    select_refinement_frontier,
    temporal_nms,
)
from adaptive_search.models import (
    BoundaryHyperparameters,
    ClusteringHyperparameters,
    EventConstraint,
    EventDefinition,
    EventProposal,
    FrameScoreSample,
    RankingHyperparameters,
    RefinementHyperparameters,
    SearchConstraints,
    SparseCandidate,
    TemporalRegion,
)


def candidate(candidate_id, event_id, video_id, timestamp, score=0.8):
    return SparseCandidate(
        id=candidate_id,
        session_id="session",
        event_id=event_id,
        video_id=video_id,
        frame_id=int(timestamp * 10),
        timestamp_seconds=float(timestamp),
        raw_relevance_score=float(score),
        query_variant="anchor-en-0",
    )


def frame_sample(event_id, region_id, timestamp, *, transition=2.0):
    before = timestamp < transition
    return FrameScoreSample(
        session_id="session",
        event_id=event_id,
        video_id="video",
        region_id=region_id,
        frame_id=int(timestamp * 10),
        timestamp_seconds=float(timestamp),
        raw_anchor_score=1.0 if timestamp == transition else 0.1,
        raw_pre_score=2.0 if before else -2.0,
        raw_post_score=-2.0 if before else 2.0,
        raw_motion_score=0.0,
    )


def proposal(proposal_id, event_id, timestamp, score, *, frame_id=None):
    return EventProposal(
        id=proposal_id,
        session_id="session",
        event_id=event_id,
        video_id="video",
        region_id=f"region-{event_id}",
        timestamp_seconds=float(timestamp),
        frame_id=frame_id if frame_id is not None else int(timestamp * 10),
        raw_semantic_score=float(score),
        normalized_semantic_score=float(score),
        raw_boundary_score=float(score),
        normalized_boundary_score=float(score),
        pre_consistency_score=float(score),
        post_persistence_score=float(score),
        final_event_score=float(score),
    )


class AdaptiveAlgorithmTests(unittest.TestCase):
    def test_region_clustering_uses_seconds_and_merges_expanded_overlap(self):
        regions = cluster_temporal_regions(
            [
                candidate("c1", "e1", "video", 1.0),
                candidate("c2", "e1", "video", 5.0),
                candidate("c3", "e1", "video", 12.0),
            ],
            ClusteringHyperparameters(gap_seconds=3.0, margin_seconds=2.0),
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].start_seconds, 0.0)
        self.assertEqual(regions[0].end_seconds, 7.0)
        self.assertEqual(regions[0].candidate_ids, ("c1", "c2"))
        self.assertEqual(regions[1].start_seconds, 10.0)

    def test_calibration_preserves_raw_scores_and_pairs_pre_post(self):
        samples = [frame_sample("e1", "r1", value) for value in range(5)]
        calibrated = calibrate_frame_scores(samples)

        self.assertEqual(
            [item.raw_anchor_score for item in calibrated],
            [item.raw_anchor_score for item in samples],
        )
        for item in calibrated:
            self.assertAlmostEqual(
                item.normalized_pre_score + item.normalized_post_score,
                1.0,
            )

    def test_boundary_proposal_peaks_at_pre_to_post_transition(self):
        parameters = BoundaryHyperparameters(
            window_options_seconds=(1.0, 2.0),
            min_samples_per_side=1,
            motion_contrast_weight=0.0,
            window_length_regularization=0.0,
            window_asymmetry_regularization=0.0,
            nms_radius_seconds=0.0,
            max_proposals_per_region=10,
        )
        proposals = generate_boundary_proposals(
            [frame_sample("e1", "r1", value) for value in range(5)],
            parameters,
        )

        self.assertTrue(proposals)
        best = max(proposals, key=lambda item: item.final_event_score)
        self.assertEqual(best.timestamp_seconds, 2.0)
        self.assertGreater(best.pre_consistency_score, 0.9)
        self.assertGreater(best.post_persistence_score, 0.9)

    def test_boundary_requires_enough_samples_on_both_sides(self):
        parameters = BoundaryHyperparameters(
            window_options_seconds=(1.0,),
            min_samples_per_side=2,
        )
        samples = [frame_sample("e1", "r1", value) for value in (0, 1, 2)]
        self.assertEqual(generate_boundary_proposals(samples, parameters), [])

    def test_temporal_nms_keeps_strongest_local_peak(self):
        kept = temporal_nms(
            [
                proposal("weak", "e1", 1.0, 0.4),
                proposal("strong", "e1", 1.2, 0.9),
                proposal("far", "e1", 3.0, 0.5),
            ],
            radius_seconds=0.5,
        )
        self.assertEqual({item.id for item in kept}, {"strong", "far"})

    def test_frontier_keeps_independent_evidence_for_each_event(self):
        regions = [
            TemporalRegion(
                id="r1",
                session_id="session",
                event_id="e1",
                video_id="video-a",
                start_seconds=0.0,
                end_seconds=2.0,
                candidate_ids=("c1",),
                raw_coarse_score=0.9,
                normalized_coarse_score=0.9,
            ),
            TemporalRegion(
                id="r2",
                session_id="session",
                event_id="e2",
                video_id="video-b",
                start_seconds=3.0,
                end_seconds=5.0,
                candidate_ids=("c2",),
                raw_coarse_score=0.8,
                normalized_coarse_score=0.8,
            ),
        ]
        parameters = RefinementHyperparameters(
            max_initial_videos=1,
            max_regions_per_event_per_video=1,
            max_total_regions=2,
            exploration_region_ratio=0.0,
        )
        priorities = prioritize_videos(regions, ["e1", "e2"], parameters)
        frontier = select_refinement_frontier(
            regions,
            priorities,
            ["e1", "e2"],
            parameters,
        )

        self.assertEqual({item.event_id for item in frontier}, {"e1", "e2"})

    def test_tuple_assembly_enforces_order_hard_constraint_and_gap_penalty(self):
        events = [
            EventDefinition(
                event_id="e1", original_query="one", anchor_query="one"
            ),
            EventDefinition(
                event_id="e2", original_query="two", anchor_query="two"
            ),
        ]
        proposals = [
            proposal("e1-good", "e1", 1.0, 0.8, frame_id=10),
            proposal("e1-late", "e1", 5.0, 0.95, frame_id=50),
            proposal("e2-good", "e2", 3.0, 0.9, frame_id=30),
        ]
        constraints = SearchConstraints(
            event_constraints={
                "e1": EventConstraint(fixed_frame_id=10),
            }
        )
        ranking = RankingHyperparameters(
            top_k=10,
            default_gap_tau_seconds=1.0,
            gap_lambda=0.1,
            fixed_constraint_bonus=0.0,
        )

        results = assemble_ordered_tuples(
            events,
            proposals,
            constraints,
            ranking,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            [item.id for item in results[0].proposals],
            ["e1-good", "e2-good"],
        )
        self.assertEqual(results[0].adjacent_gaps_seconds, (2.0,))
        self.assertAlmostEqual(results[0].raw_gap_penalty, 0.1)


if __name__ == "__main__":
    unittest.main()
