import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.benchmark_client import (
    BenchmarkRun,
    ManifestError,
    QueryItem,
    aggregate_video_scores,
    build_retrieval_payload,
    deduplicate_ranked_videos,
    mean_reciprocal_rank,
    median_rank,
    parse_manifest,
    rank_of_ground_truth,
    rank_videos,
    recall_at_k,
    summarize_metrics,
)


class ManifestLeakageGuardTests(unittest.TestCase):
    def test_video_path_parsed_only_in_evaluator(self):
        manifest = {
            "queries": [
                {
                    "query": "a person opens a door",
                    "video_path": "/datasets/youcook2/L21_V001.mp4",
                },
                "a person sits down",
            ]
        }
        items = parse_manifest(manifest)
        self.assertEqual(items[0].ground_truth_video_id, "L21_V001")
        self.assertIsNone(items[1].ground_truth_video_id)

    def test_retrieval_payload_never_contains_ground_truth(self):
        payload = build_retrieval_payload("a person opens a door", top_k=100)
        self.assertEqual(set(payload), {"query", "top_k"})
        serialized = __import__("json").dumps(payload)
        self.assertNotIn("L21_V001", serialized)
        self.assertNotIn("video_path", serialized)

    def test_manifest_error_on_empty_queries(self):
        with self.assertRaises(ManifestError):
            parse_manifest({"queries": [{"query": ""}]})

    def test_manifest_accepts_bare_list(self):
        items = parse_manifest(["one", "two"])
        self.assertEqual([item.query for item in items], ["one", "two"])


class DedupAggregationTests(unittest.TestCase):
    def test_frame_hits_deduplicated_by_video(self):
        hits = [
            {"video_name": "V1", "score": 0.9},
            {"video_name": "V1", "score": 0.2},
            {"video_name": "V2", "score": 0.7},
        ]
        ranked = deduplicate_ranked_videos(hits)
        self.assertEqual(ranked, ["V1", "V2"])

    def test_max_aggregation(self):
        hits = [
            {"video_name": "V1", "score": 0.5},
            {"video_name": "V1", "score": 0.9},
            {"video_name": "V2", "score": 0.8},
        ]
        scores = aggregate_video_scores(hits, method="max")
        self.assertAlmostEqual(scores["V1"], 0.9)
        self.assertAlmostEqual(scores["V2"], 0.8)

    def test_rank_videos_by_aggregation(self):
        hits = [
            {"video_name": "V1", "score": 0.1},
            {"video_name": "V1", "score": 0.1},
            {"video_name": "V2", "score": 0.5},
        ]
        self.assertEqual(rank_videos(hits, method="max"), ["V2", "V1"])

    def test_unknown_aggregation_raises(self):
        with self.assertRaises(ValueError):
            aggregate_video_scores([], method="bogus")


class MetricTests(unittest.TestCase):
    def test_recall_at_k_with_known_ranks(self):
        # Denominator is queries that have ground truth; GT-less queries are
        # excluded and reported separately via missing_ground_truth.
        ranks = [1, 3, 5, None]
        self.assertAlmostEqual(recall_at_k(ranks, 5), 3 / 3)
        self.assertAlmostEqual(recall_at_k(ranks, 1), 1 / 3)
        self.assertEqual(recall_at_k([None, None], 10), 0.0)

    def test_mrr(self):
        self.assertAlmostEqual(mean_reciprocal_rank([1, 2, None]), (1.0 + 0.5) / 2)

    def test_median_rank(self):
        self.assertEqual(median_rank([1, 2, 3]), 2.0)
        self.assertEqual(median_rank([1, 2, 3, 4]), 2.5)
        self.assertIsNone(median_rank([None]))

    def test_summary_counts_missing(self):
        summary = summarize_metrics([1, None, 7], k_list=[1, 5])
        self.assertEqual(summary["queries_with_ground_truth"], 2)
        self.assertEqual(summary["missing_ground_truth"], 1)
        self.assertAlmostEqual(summary["recall"][5], 0.5)  # only rank 1 hits @5
        self.assertAlmostEqual(summary["recall"][1], 0.5)


class BenchmarkRunTests(unittest.TestCase):
    def test_checkpoint_resume(self):
        run = BenchmarkRun(queries=[QueryItem(query="q0", ground_truth_video_id="V1", index=0),
                                   QueryItem(query="q1", ground_truth_video_id="V2", index=1)])
        run.record(0, ["V1", "V2"], {"payload": {}, "frame_hits": 1})
        pending = run.resume()
        self.assertEqual([item.index for item in pending], [1])

    def test_record_rank_hit_and_missing(self):
        run = BenchmarkRun(queries=[QueryItem(query="q", ground_truth_video_id="V1", index=0)])
        run.record(0, ["V1", "V2"], {"payload": {}, "frame_hits": 2})
        self.assertEqual(run.completed[0]["rank"], 1)
        run.record(0, ["V2", "V3"], {"payload": {}, "frame_hits": 2})
        self.assertIsNone(run.completed[0]["rank"])
        self.assertFalse(run.completed[0]["hit_at"][5])

    def test_run_query_uses_sanitized_payload(self):
        captured = {}

        def search(query, top_k):
            captured["payload"] = build_retrieval_payload(query, top_k)
            return [{"video_name": "V1", "score": 0.9}]

        run = BenchmarkRun(queries=[QueryItem(query="q", ground_truth_video_id="V1", index=0)])
        run.run_query(run.queries[0], search=search)
        self.assertEqual(captured["payload"], {"query": "q", "top_k": 100})
        self.assertEqual(run.completed[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
