from __future__ import annotations

import unittest

from benchmarks.youcook2.boundary_metrics import (
    DEFAULT_KS,
    aggregate_boundary_metrics,
    final_query_score,
    frame_hits,
    recall_at_k_new,
)
from benchmarks.youcook2.core import VideoQueryGroup


def _group() -> VideoQueryGroup:
    return VideoQueryGroup(
        video_id="Video-A",
        source="test",
        context=None,
        events=(("E1", "cut onion"), ("E2", "fry onion")),
        answers={"E1": (10.0, 20.0), "E2": (30.0, 40.0)},
    )


class FrameHitsTests(unittest.TestCase):
    def test_counts_only_anchors_inside_their_own_interval(self) -> None:
        group = _group()
        self.assertEqual(frame_hits(group, {"E1": 15.0, "E2": 35.0}), 2)
        self.assertEqual(frame_hits(group, {"E1": 15.0, "E2": 999.0}), 1)
        self.assertEqual(frame_hits(group, {"E1": 999.0, "E2": 999.0}), 0)

    def test_missing_event_anchor_contributes_zero_not_an_exception(self) -> None:
        group = _group()
        self.assertEqual(frame_hits(group, {}), 0)
        self.assertEqual(frame_hits(group, {"E1": 15.0}), 1)


class RecallAtKNewTests(unittest.TestCase):
    def test_zero_when_rank_missing_or_beyond_k_regardless_of_hits(self) -> None:
        self.assertEqual(recall_at_k_new(rank=None, hits=2, k=100), 0)
        self.assertEqual(recall_at_k_new(rank=101, hits=2, k=100), 0)

    def test_equals_hits_when_rank_within_k(self) -> None:
        self.assertEqual(recall_at_k_new(rank=1, hits=2, k=1), 2)
        self.assertEqual(recall_at_k_new(rank=50, hits=0, k=100), 0)

    def test_closed_form_matches_direct_mean(self) -> None:
        # final_query_score(query) = hits * |{k : rank <= k}| / |Ks|
        rank, hits = 8, 3
        ks = DEFAULT_KS
        direct = sum(recall_at_k_new(rank, hits, k) for k in ks) / len(ks)
        cleared = sum(1 for k in ks if rank <= k)
        closed_form = hits * cleared / len(ks)
        self.assertAlmostEqual(final_query_score(rank, hits, ks), direct)
        self.assertAlmostEqual(final_query_score(rank, hits, ks), closed_form)

    def test_final_query_score_zero_when_video_never_found(self) -> None:
        self.assertEqual(final_query_score(rank=None, hits=4), 0.0)


class AggregateBoundaryMetricsTests(unittest.TestCase):
    def test_empty_rows_return_zeroed_summary(self) -> None:
        summary = aggregate_boundary_metrics([])
        self.assertEqual(summary["query_count"], 0)
        self.assertEqual(summary["final_query_score_mean"], 0.0)
        for k in DEFAULT_KS:
            self.assertEqual(summary[f"recall_at_{k}_new_mean"], 0.0)

    def test_aggregates_across_multiple_queries(self) -> None:
        rows = [
            {"rank": 1, "hits": 2},  # clears every k -> contributes 2 at every k
            {"rank": None, "hits": 4},  # never found -> contributes 0 everywhere
            {"rank": 10, "hits": 1},  # clears k=20,50,100 but not k=1,5
        ]
        summary = aggregate_boundary_metrics(rows)
        self.assertEqual(summary["query_count"], 3)
        self.assertAlmostEqual(summary["recall_at_1_new_mean"], (2 + 0 + 0) / 3)
        self.assertAlmostEqual(summary["recall_at_5_new_mean"], (2 + 0 + 0) / 3)
        self.assertAlmostEqual(summary["recall_at_20_new_mean"], (2 + 0 + 1) / 3)
        self.assertAlmostEqual(summary["recall_at_100_new_mean"], (2 + 0 + 1) / 3)


if __name__ == "__main__":
    unittest.main()
