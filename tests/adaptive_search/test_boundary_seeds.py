import unittest

from adaptive_search.boundary_seeds import select_event_seeds
from adaptive_search.schemas import SparseCandidate, TemporalRegion


def _region(region_id, *, event_id="e1", video_id="video-a", candidate_ids, raw_coarse_score):
    return TemporalRegion(
        id=region_id,
        session_id="s1",
        event_id=event_id,
        video_id=video_id,
        start_seconds=0.0,
        end_seconds=6.0,
        candidate_ids=tuple(candidate_ids),
        raw_coarse_score=raw_coarse_score,
    )


def _candidate(candidate_id, *, event_id="e1", video_id="video-a", timestamp_seconds, raw_relevance_score):
    return SparseCandidate(
        id=candidate_id,
        session_id="s1",
        event_id=event_id,
        video_id=video_id,
        frame_id=int(timestamp_seconds * 30),
        timestamp_seconds=timestamp_seconds,
        raw_relevance_score=raw_relevance_score,
        query_variant="e1:v0",
    )


class SelectEventSeedsTests(unittest.TestCase):
    def test_no_matching_regions_returns_empty(self) -> None:
        seeds = select_event_seeds(
            regions=[], candidates_by_id={}, event_id="e1", video_id="video-a"
        )
        self.assertEqual(seeds, [])

    def test_single_region_returns_its_candidates_best_score_first(self) -> None:
        candidates = [
            _candidate("c1", timestamp_seconds=10.0, raw_relevance_score=0.9),
            _candidate("c2", timestamp_seconds=11.0, raw_relevance_score=0.95),
        ]
        regions = [_region("r1", candidate_ids=["c1", "c2"], raw_coarse_score=0.95)]
        seeds = select_event_seeds(
            regions=regions,
            candidates_by_id={c.id: c for c in candidates},
            event_id="e1",
            video_id="video-a",
        )
        self.assertEqual(seeds, [11.0, 10.0])

    def test_low_scoring_region_excluded_by_score_threshold(self) -> None:
        candidates = [
            _candidate("c1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            _candidate("c2", timestamp_seconds=50.0, raw_relevance_score=0.5),  # < 90% of 1.0
        ]
        regions = [
            _region("r1", candidate_ids=["c1"], raw_coarse_score=1.0),
            _region("r2", candidate_ids=["c2"], raw_coarse_score=0.5),
        ]
        seeds = select_event_seeds(
            regions=regions,
            candidates_by_id={c.id: c for c in candidates},
            event_id="e1",
            video_id="video-a",
        )
        self.assertEqual(seeds, [10.0])

    def test_region_count_capped_independent_of_score_threshold(self) -> None:
        # Three regions all within 10% of each other's score - MAX_REGIONS_PER_EVENT=2
        # excludes the third even though it would pass the threshold alone.
        candidates = [
            _candidate("c1", timestamp_seconds=10.0, raw_relevance_score=1.00),
            _candidate("c2", timestamp_seconds=50.0, raw_relevance_score=0.98),
            _candidate("c3", timestamp_seconds=90.0, raw_relevance_score=0.96),
        ]
        regions = [
            _region("r1", candidate_ids=["c1"], raw_coarse_score=1.00),
            _region("r2", candidate_ids=["c2"], raw_coarse_score=0.98),
            _region("r3", candidate_ids=["c3"], raw_coarse_score=0.96),
        ]
        seeds = select_event_seeds(
            regions=regions,
            candidates_by_id={c.id: c for c in candidates},
            event_id="e1",
            video_id="video-a",
        )
        self.assertEqual(seeds, [10.0, 50.0])

    def test_seed_count_capped_at_max_seeds_per_event(self) -> None:
        candidates = [
            _candidate(f"c{i}", timestamp_seconds=float(i), raw_relevance_score=1.0 - i * 0.001)
            for i in range(10)
        ]
        regions = [_region("r1", candidate_ids=[c.id for c in candidates], raw_coarse_score=1.0)]
        seeds = select_event_seeds(
            regions=regions,
            candidates_by_id={c.id: c for c in candidates},
            event_id="e1",
            video_id="video-a",
        )
        self.assertEqual(len(seeds), 6)  # MAX_SEEDS_PER_EVENT
        self.assertEqual(seeds, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_wrong_event_or_video_id_does_not_match(self) -> None:
        regions = [_region("r1", candidate_ids=["c1"], raw_coarse_score=1.0)]
        candidates_by_id = {"c1": _candidate("c1", timestamp_seconds=10.0, raw_relevance_score=1.0)}
        self.assertEqual(
            select_event_seeds(
                regions=regions, candidates_by_id=candidates_by_id, event_id="e2", video_id="video-a"
            ),
            [],
        )
        self.assertEqual(
            select_event_seeds(
                regions=regions, candidates_by_id=candidates_by_id, event_id="e1", video_id="video-b"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
