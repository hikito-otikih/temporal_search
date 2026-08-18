from __future__ import annotations

import unittest

from adaptive_search.schemas import SparseCandidate
from benchmarks.youcook2.legacy_clustering import (
    ClusteringHyperparameters,
    cluster_temporal_regions,
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


class LegacyClusteringTests(unittest.TestCase):
    def test_region_clustering_uses_seconds_and_merges_expanded_overlap(self) -> None:
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

    def test_region_chain_is_split_and_not_remerged_past_max_span(self) -> None:
        candidates = [
            candidate(f"c{frame}", "e1", "video", frame, 1.0)
            for frame in range(21)
        ]
        regions = cluster_temporal_regions(
            candidates,
            ClusteringHyperparameters(
                gap_seconds=2.0,
                margin_seconds=1.0,
                max_region_seconds=10.0,
            ),
        )

        self.assertGreater(len(regions), 1)
        self.assertTrue(
            all(
                region.end_seconds - region.start_seconds <= 10.0
                for region in regions
            )
        )

    def test_candidate_calibration_is_independent_per_event(self) -> None:
        regions = cluster_temporal_regions(
            [
                candidate("a", "e1", "video", 1, 100.0),
                candidate("b", "e1", "video", 2, 101.0),
                candidate("c", "e2", "video", 1, 0.0),
                candidate("d", "e2", "video", 2, 1.0),
            ],
            ClusteringHyperparameters(
                gap_seconds=3.0,
                margin_seconds=0.0,
                max_region_seconds=10.0,
            ),
        )
        scores = {region.event_id: region.normalized_coarse_score for region in regions}
        self.assertAlmostEqual(scores["e1"], scores["e2"])


if __name__ == "__main__":
    unittest.main()
