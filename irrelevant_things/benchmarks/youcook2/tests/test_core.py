from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.youcook2.core import (
    DatasetFormatError,
    QueryRecord,
    aggregate_video_hits,
    canonical_video_id,
    compute_metrics,
    load_official_annotations,
    load_query_directory,
    rank_of,
)


class QueryParsingTests(unittest.TestCase):
    def test_local_query_file_parses_events_without_answer_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            query_dir = Path(temporary)
            (query_dir / "Video-A.txt").write_text(
                "Hướng dẫn nấu ăn, tìm các sự kiện sau:\n"
                "E1: cắt hành tây\n"
                "E2: chiên hành tây\n"
                "**Answer\n"
                'video_path: "C:\\\\data\\\\Video-A.mp4"\n'
                "E1: 0:41 - 0:53\n"
                "E2: 1:26 - 1:33\n",
                encoding="utf-8",
            )
            records = load_query_directory(query_dir)

        self.assertEqual([record.query_id for record in records], ["Video-A:E1", "Video-A:E2"])
        self.assertEqual(records[0].query_text, "cắt hành tây")
        self.assertNotIn("video_path", records[0].query_text)
        self.assertEqual(records[0].ground_truth_video, "Video-A")
        self.assertEqual((records[0].start_seconds, records[0].end_seconds), (41.0, 53.0))

    def test_official_annotations_support_subset_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations.json"
            path.write_text(
                json.dumps(
                    {
                        "database": {
                            "TrainVideo": {
                                "subset": "training",
                                "annotations": [{"id": 1, "sentence": "mix flour", "segment": [2, 5]}],
                            },
                            "ValVideo": {
                                "subset": "validation",
                                "annotations": [{"id": 7, "sentence": "slice onion", "segment": [6, 9]}],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            records = load_official_annotations(path, subsets=["validation"])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].query_id, "ValVideo:7")
        self.assertEqual(records[0].query_text, "slice onion")

    def test_query_record_rejects_invalid_temporal_ranges(self) -> None:
        common = {
            "query_id": "q1",
            "query_text": "slice onion",
            "ground_truth_video": "video",
            "source": "fixture",
        }
        for values in (
            {"start_seconds": -1.0},
            {"start_seconds": float("nan")},
            {"start_seconds": 5.0, "end_seconds": 4.0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(DatasetFormatError):
                    QueryRecord(**common, **values)


class AggregationAndMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [
            {"video_name": "A.mp4", "score": 0.9},
            {"video_name": "A.mp4", "score": 0.4},
            {"video_name": "B.mp4", "score": 0.8},
        ]

    def test_canonical_id_preserves_case(self) -> None:
        self.assertEqual(canonical_video_id(r"C:\videos\Ab_C-1.MP4"), "Ab_C-1")

    def test_max_and_top_m_mean_produce_unique_rankings(self) -> None:
        max_ranking = aggregate_video_hits(self.hits, method="max")
        mean_ranking = aggregate_video_hits(self.hits, method="top_m_mean", top_m=2)
        self.assertEqual([item.video_id for item in max_ranking], ["A", "B"])
        self.assertEqual([item.video_id for item in mean_ranking], ["B", "A"])
        self.assertEqual(rank_of(max_ranking, "A.mp4"), 1)

    def test_logsumexp_rewards_repeated_evidence(self) -> None:
        ranking = aggregate_video_hits(self.hits, method="logsumexp", temperature=0.2)
        self.assertEqual(ranking[0].video_id, "A")

    def test_recall_mrr_and_optimistic_imputed_median(self) -> None:
        rows = [
            {"status": "ok", "rank": 1, "unique_video_count": 3, "latency_ms": 10},
            {"status": "ok", "rank": 3, "unique_video_count": 4, "latency_ms": 20},
            {"status": "ok", "rank": None, "unique_video_count": 4, "latency_ms": 30},
            {"status": "error", "rank": None, "unique_video_count": 0, "latency_ms": 1},
        ]
        metrics = compute_metrics(rows, [1, 5])
        self.assertAlmostEqual(metrics["recall_at_1"], 1 / 4)
        self.assertAlmostEqual(metrics["recall_at_5"], 2 / 4)
        self.assertAlmostEqual(
            metrics["recall_at_1_successful_requests"], 1 / 3
        )
        self.assertAlmostEqual(
            metrics["recall_at_5_successful_requests"], 2 / 3
        )
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 3) / 4)
        self.assertAlmostEqual(
            metrics["mrr_successful_requests"], (1 + 1 / 3) / 3
        )
        self.assertEqual(metrics["median_rank"], 3)
        self.assertEqual(metrics["error_count"], 1)
        self.assertEqual(metrics["ground_truth_found_count"], 2)
        self.assertEqual(metrics["ground_truth_miss_count"], 1)
        self.assertEqual(metrics["unique_video_count_min"], 3)
        self.assertEqual(metrics["unique_video_count_median"], 4)
        self.assertAlmostEqual(metrics["unique_video_count_mean"], 11 / 3)
        self.assertEqual(metrics["unique_video_count_max"], 4)


if __name__ == "__main__":
    unittest.main()
