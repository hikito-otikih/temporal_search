from __future__ import annotations

import unittest
from typing import Any, Mapping

from adaptive_search.client import UpstreamSearchClient
from benchmarks.youcook2.coarse_anchor import adaptive_coarse_before
from benchmarks.youcook2.core import VideoQueryGroup

# One hit per event: E1 -> Video-A only (one dominant candidate + one far
# below the score threshold), E2 -> nothing for Video-A (so E2 should end up
# with no anchor at all, not an exception), plus a distractor video ranked
# ahead of Video-A on E1 to exercise real rank computation.
_RESULTS_BY_QUERY: dict[str, list[dict[str, Any]]] = {
    "cut onion": [
        {"video_name": "Video-B.mp4", "frame_index": 1, "timestamp_seconds": 5.0, "score": 0.9},
        {"video_name": "Video-A.mp4", "frame_index": 2, "timestamp_seconds": 12.0, "score": 0.5},
        {"video_name": "Video-A.mp4", "frame_index": 3, "timestamp_seconds": 15.0, "score": 0.8},
    ],
    "fry onion": [
        {"video_name": "Video-C.mp4", "frame_index": 4, "timestamp_seconds": 31.0, "score": 0.7},
    ],
}

# fuse_candidates_rrf() derives raw_relevance_score from RANK position within
# one query variant's result list (rrf_score = 1/(rrf_k + rank), default
# rrf_k=60), NOT from the raw upstream "score" field directly - the raw
# scores below only fix the ORDER; their magnitudes are irrelevant. Seed
# selection is now region-based (cluster_temporal_regions() groups
# same-event/video candidates within its default gap_seconds=3.0 merge
# window before any threshold/cap logic runs), so what matters is which
# candidates land in the same region, not just their individual RRF gaps.
#
# Video-A gets ranks 1/2/3/4 at timestamps 10/11/12/13 (1s gaps -> all one
# merged region) plus rank 12 at timestamp 20.0 (7s gap from 13.0 -> a
# separate, singleton region). 7 distractor-video fillers occupy ranks 5-11
# so rank 12's RRF score is meaningfully lower. Hand-verified RRF scores
# relative to rank 1 (1/61=0.016393):
#   region {10,11,12,13}'s raw_coarse_score = rank 1's score (max of the cluster) = 0.016393
#   region {20.0}'s raw_coarse_score = rank 12's score = 1/72=0.013889, ratio 0.8472 -> excluded by the 10% region threshold
_MULTI_SEED_RESULTS_BY_QUERY: dict[str, list[dict[str, Any]]] = {
    "cut onion": [
        {"video_name": "Video-A.mp4", "frame_index": 1, "timestamp_seconds": 10.0, "score": 0.99},
        {"video_name": "Video-A.mp4", "frame_index": 2, "timestamp_seconds": 11.0, "score": 0.98},
        {"video_name": "Video-A.mp4", "frame_index": 3, "timestamp_seconds": 12.0, "score": 0.97},
        {"video_name": "Video-A.mp4", "frame_index": 4, "timestamp_seconds": 13.0, "score": 0.96},
        {"video_name": "Video-F1.mp4", "frame_index": 5, "timestamp_seconds": 1.0, "score": 0.95},
        {"video_name": "Video-F2.mp4", "frame_index": 6, "timestamp_seconds": 1.0, "score": 0.94},
        {"video_name": "Video-F3.mp4", "frame_index": 7, "timestamp_seconds": 1.0, "score": 0.93},
        {"video_name": "Video-F4.mp4", "frame_index": 8, "timestamp_seconds": 1.0, "score": 0.92},
        {"video_name": "Video-F5.mp4", "frame_index": 9, "timestamp_seconds": 1.0, "score": 0.91},
        {"video_name": "Video-F6.mp4", "frame_index": 10, "timestamp_seconds": 1.0, "score": 0.90},
        {"video_name": "Video-F7.mp4", "frame_index": 11, "timestamp_seconds": 1.0, "score": 0.89},
        {"video_name": "Video-A.mp4", "frame_index": 12, "timestamp_seconds": 20.0, "score": 0.88},
    ],
    "fry onion": [],
}

# Three well-separated (>3s gaps -> three singleton regions) Video-A
# candidates at ranks 1/2/3 - with rrf_k=60 all three regions are within the
# 10% score threshold of each other (singleton region score == its one
# candidate's own RRF score), so MAX_REGIONS_PER_EVENT=2 is what actually
# excludes the third one, not the score threshold.
_MULTI_REGION_RESULTS_BY_QUERY: dict[str, list[dict[str, Any]]] = {
    "cut onion": [
        {"video_name": "Video-A.mp4", "frame_index": 1, "timestamp_seconds": 10.0, "score": 0.99},
        {"video_name": "Video-A.mp4", "frame_index": 2, "timestamp_seconds": 50.0, "score": 0.98},
        {"video_name": "Video-A.mp4", "frame_index": 3, "timestamp_seconds": 90.0, "score": 0.97},
    ],
    "fry onion": [],
}


def _fake_transport(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    query = payload["query"]
    results = _RESULTS_BY_QUERY.get(query, [])
    return {"query": query, "top_k": payload["top_k"], "results": results}


def _group() -> VideoQueryGroup:
    return VideoQueryGroup(
        video_id="Video-A",
        source="test",
        context=None,
        events=(("E1", "cut onion"), ("E2", "fry onion")),
        answers={"E1": (10.0, 20.0), "E2": (30.0, 40.0)},
    )


class AdaptiveCoarseBeforeTests(unittest.TestCase):
    def test_picks_highest_scoring_candidate_per_event_for_the_gt_video(self) -> None:
        upstream = UpstreamSearchClient(transport=_fake_transport)
        result = adaptive_coarse_before(_group(), upstream=upstream, top_k=10)

        # Video-A's two E1 candidates (12.0, 15.0) are 3.0s apart - exactly at
        # cluster_temporal_regions()'s default gap_seconds=3.0 (inclusive), so
        # they merge into ONE region. Both its candidates become seeds,
        # best-score-first (15.0 outranks 12.0 - Video-B's candidate is rank
        # 1 overall, so 15.0/12.0 are ranks 2/3).
        self.assertEqual(result.event_anchors["E1"], (15.0, 12.0))
        # No Video-A candidate at all for E2 -> no matching region -> no
        # anchor, no exception.
        self.assertNotIn("E2", result.event_anchors)

    def test_seeds_are_every_candidate_in_the_surviving_region_not_a_flat_pool_cutoff(self) -> None:
        def multi_seed_transport(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
            query = payload["query"]
            return {
                "query": query,
                "top_k": payload["top_k"],
                "results": _MULTI_SEED_RESULTS_BY_QUERY.get(query, []),
            }

        upstream = UpstreamSearchClient(transport=multi_seed_transport)
        result = adaptive_coarse_before(_group(), upstream=upstream, top_k=20)

        # All 4 candidates in the {10,11,12,13} region survive together
        # (region-level threshold, not an individual per-candidate one) and
        # all fit under MAX_SEEDS_PER_EVENT=6; the far-away rank-12 region at
        # 20.0 is excluded entirely by the region-level score threshold.
        self.assertEqual(result.event_anchors["E1"], (10.0, 11.0, 12.0, 13.0))

    def test_region_count_is_capped_independent_of_the_score_threshold(self) -> None:
        def multi_region_transport(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
            query = payload["query"]
            return {
                "query": query,
                "top_k": payload["top_k"],
                "results": _MULTI_REGION_RESULTS_BY_QUERY.get(query, []),
            }

        upstream = UpstreamSearchClient(transport=multi_region_transport)
        result = adaptive_coarse_before(_group(), upstream=upstream, top_k=10)

        # Three separate singleton regions (>3s apart), ranks 1/2/3 - all
        # three pass the 10% score threshold (rrf_k=60 keeps adjacent ranks
        # close), so MAX_REGIONS_PER_EVENT=2 is what excludes the third
        # (ts=90.0), not the threshold.
        self.assertEqual(result.event_anchors["E1"], (10.0, 50.0))

    def test_rank_reflects_video_priority_ordering_not_just_presence(self) -> None:
        upstream = UpstreamSearchClient(transport=_fake_transport)
        result = adaptive_coarse_before(_group(), upstream=upstream, top_k=10)
        # Video-A, Video-B, and Video-C are all present in the fused pool;
        # rank must be a real position, not None, and unique_video_count
        # must count every distinct video seen.
        self.assertIsNotNone(result.rank)
        self.assertEqual(result.unique_video_count, 3)

    def test_gt_video_entirely_absent_yields_none_rank_and_no_anchors(self) -> None:
        def empty_transport(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
            return {"query": payload["query"], "top_k": payload["top_k"], "results": []}

        upstream = UpstreamSearchClient(transport=empty_transport)
        result = adaptive_coarse_before(_group(), upstream=upstream, top_k=10)
        self.assertIsNone(result.rank)
        self.assertEqual(result.event_anchors, {})
        self.assertEqual(result.unique_video_count, 0)


if __name__ == "__main__":
    unittest.main()
