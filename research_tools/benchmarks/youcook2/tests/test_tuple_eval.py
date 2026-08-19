from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.youcook2.core import DatasetFormatError, load_query_directory_grouped
from benchmarks.youcook2.tuple_runner import TupleRunConfig, run_tuple_benchmark


def _write_query_file(directory: Path, name: str, video_path: str) -> None:
    (directory / name).write_text(
        "Hướng dẫn nấu ăn, tìm các sự kiện sau:\n"
        "E1: cắt hành tây\n"
        "E2: chiên hành tây\n"
        "**Answer\n"
        f'video_path: "{video_path}"\n'
        "E1: 0:10 - 0:20\n"
        "E2: 0:30 - 0:40\n",
        encoding="utf-8",
    )


class GroupedParsingTests(unittest.TestCase):
    def test_load_query_directory_grouped_preserves_event_order_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            query_dir = Path(temporary)
            _write_query_file(query_dir, "Video-A.txt", "C:\\data\\Video-A.mp4")
            groups = load_query_directory_grouped(query_dir)

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group.video_id, "Video-A")
        self.assertEqual(group.events, (("E1", "cắt hành tây"), ("E2", "chiên hành tây")))
        self.assertEqual(group.answers, {"E1": (10.0, 20.0), "E2": (30.0, 40.0)})

    def test_duplicate_video_id_across_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            query_dir = Path(temporary)
            _write_query_file(query_dir, "one.txt", "same-video.mp4")
            _write_query_file(query_dir, "two.txt", "same-video.mp4")
            with self.assertRaises(DatasetFormatError):
                load_query_directory_grouped(query_dir)


class FakeBackendHandler(BaseHTTPRequestHandler):
    legacy_results: list[dict[str, object]] = []
    legacy_search_truncated: bool = False
    video_priorities: list[dict[str, object]] = []
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def log_message(self, *_args: object) -> None:
        return

    def _json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, method: str, payload: dict[str, object] | None) -> None:
        type(self).requests.append((method, self.path, payload))

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._record("GET", None)
        path = urllib.parse.urlsplit(self.path).path
        if path.endswith("/video-priorities"):
            self._json({"items": self.video_priorities, "total": len(self.video_priorities), "offset": 0, "limit": 100})
        else:
            self._json({"detail": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        self._record("POST", payload)
        if self.path == "/temporal-search":
            self._json({
                "query": payload["query"],
                "results": self.legacy_results,
                "search_truncated": self.legacy_search_truncated,
            })
        elif self.path == "/v1/search-sessions":
            self._json(
                {
                    "session": {"id": "sess-1"},
                    "artifact_counts": {},
                    "live_refinement": {},
                },
                201,
            )
        elif self.path.endswith("/commands/retrieve"):
            self._json({"session_revision": 1, "run_id": "r1", "run_status": "completed", "metrics": {}, "artifact_counts": {}})
        else:
            self._json({"detail": "not found"}, 404)


class TupleRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackendHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        FakeBackendHandler.requests = []
        FakeBackendHandler.legacy_search_truncated = False

    def _config(self, pipeline: str, **overrides) -> TupleRunConfig:
        return TupleRunConfig(
            backend_base_url=self.base_url,
            pipeline=pipeline,
            top_k_tuple=10,
            top_k_each_query=10,
            gamma=0.05,
            adaptive_top_k=20,
            recall_ks=(1, 5),
            timeout_seconds=2.0,
            retries=0,
            retry_backoff_seconds=0.0,
            **overrides,
        )

    def _group(self):
        with tempfile.TemporaryDirectory() as temporary:
            query_dir = Path(temporary)
            _write_query_file(query_dir, "video.txt", "ground-truth.mp4")
            return load_query_directory_grouped(query_dir)[0]

    def test_legacy_pipeline_ranks_ground_truth_video(self) -> None:
        group = self._group()
        FakeBackendHandler.legacy_results = [
            {
                "score": 0.9,
                "video_name": "distractor.mp4",
                "tuple": [
                    {"frame_index": 1, "timestamp": "0:10", "score": 0.5, "query_id": 0},
                    {"frame_index": 2, "timestamp": "0:30", "score": 0.5, "query_id": 1},
                ],
            },
            {
                "score": 0.8,
                "video_name": "ground-truth.mp4",
                "tuple": [
                    {"frame_index": 3, "timestamp": "0:15", "score": 0.4, "query_id": 0},
                    {"frame_index": 4, "timestamp": "0:35", "score": 0.4, "query_id": 1},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            metrics, rows = run_tuple_benchmark(
                [group],
                output_dir,
                self._config("legacy_temporal"),
                {"type": "fixture"},
                "digest",
                progress_every=0,
            )
        self.assertEqual(rows[0]["rank"], 2)
        self.assertEqual(rows[0]["unique_video_count"], 2)
        self.assertEqual(rows[0]["event_timestamp_accuracy"], {"E1": True, "E2": True})
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        payload = next(item for method, path, item in FakeBackendHandler.requests if path == "/temporal-search")
        self.assertEqual(payload["searcher_type"], "TemporalSearcher")
        self.assertNotIn("ground-truth", json.dumps(payload))  # GT id never sent to backend

    def test_legacy_search_truncated_is_surfaced_not_silently_dropped(self) -> None:
        # Regression test for a real reported bug: the backend's
        # search_truncated flag (true when a video's backtracking search hit
        # its node budget before exploring exhaustively) was read from the
        # response but then discarded - a benchmark run summary had no way
        # to tell truncated results from exhaustive ones, so it could report
        # incomplete results as if the search had been complete.
        group = self._group()
        FakeBackendHandler.legacy_results = [
            {
                "score": 0.8,
                "video_name": "ground-truth.mp4",
                "tuple": [
                    {"frame_index": 3, "timestamp": "0:15", "score": 0.4, "query_id": 0},
                    {"frame_index": 4, "timestamp": "0:35", "score": 0.4, "query_id": 1},
                ],
            },
        ]
        FakeBackendHandler.legacy_search_truncated = True
        with tempfile.TemporaryDirectory() as output_dir:
            _metrics, rows = run_tuple_benchmark(
                [group],
                output_dir,
                self._config("legacy_temporal"),
                {"type": "fixture"},
                "digest",
                progress_every=0,
            )
        self.assertTrue(rows[0]["search_truncated"])

    def test_adaptive_coarse_uses_video_priorities_endpoint(self) -> None:
        group = self._group()
        FakeBackendHandler.video_priorities = [
            {"video_id": "distractor", "priority_score": 0.9, "event_coverage": 2, "normalized_coverage": 1.0, "mean_best_event_score": 0.9, "min_best_event_score": 0.9},
            {"video_id": "ground-truth", "priority_score": 0.5, "event_coverage": 1, "normalized_coverage": 0.5, "mean_best_event_score": 0.5, "min_best_event_score": 0.5},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            metrics, rows = run_tuple_benchmark(
                [group],
                output_dir,
                self._config("adaptive_coarse"),
                {"type": "fixture"},
                "digest",
                progress_every=0,
            )
        self.assertEqual(rows[0]["rank"], 2)
        self.assertEqual(rows[0]["status"], "ok")
        paths = [path for _, path, _ in FakeBackendHandler.requests]
        self.assertTrue(any(path.endswith("/commands/retrieve") for path in paths))
        self.assertTrue(any(path.endswith("/video-priorities") for path in paths))
        self.assertFalse(any(path.endswith("/commands/refine") for path in paths))

    def test_adaptive_ranking_top_k_caps_video_priorities_page_for_coarse(self) -> None:
        group = self._group()
        with tempfile.TemporaryDirectory() as output_dir:
            run_tuple_benchmark(
                [group],
                output_dir,
                self._config("adaptive_coarse", adaptive_ranking_top_k=250),
                {"type": "fixture"},
                "digest",
                progress_every=0,
            )
        path = next(
            path
            for method, path, _ in FakeBackendHandler.requests
            if method == "GET" and "/video-priorities" in path
        )
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
        self.assertEqual(query["limit"], ["250"])

    def test_adaptive_coarse_sends_retrieval_and_refinement_overrides(self) -> None:
        group = self._group()
        with tempfile.TemporaryDirectory() as output_dir:
            run_tuple_benchmark(
                [group],
                output_dir,
                self._config(
                    "adaptive_coarse",
                    adaptive_top_n_fused=500,
                    adaptive_video_mean_weight=1.0,
                    adaptive_video_coverage_weight=0.0,
                    adaptive_video_min_weight=0.0,
                ),
                {"type": "fixture"},
                "digest",
                progress_every=0,
            )
        payload = next(
            item for method, path, item in FakeBackendHandler.requests if path == "/v1/search-sessions"
        )
        self.assertEqual(payload["hyperparameters"]["retrieval"], {"top_n_fused": 500})
        self.assertEqual(
            payload["hyperparameters"]["refinement"],
            {
                "video_mean_weight": 1.0,
                "video_coverage_weight": 0.0,
                "video_min_weight": 0.0,
            },
        )
        self.assertNotIn("ranking", payload["hyperparameters"])

    def test_adaptive_coarse_does_not_send_ranking_hyperparameters(self) -> None:
        group = self._group()
        with tempfile.TemporaryDirectory() as output_dir:
            run_tuple_benchmark(
                [group],
                output_dir,
                self._config("adaptive_coarse", adaptive_ranking_top_k=100),
                {"type": "fixture"},
                "digest",
                progress_every=0,
            )
        payload = next(
            item for method, path, item in FakeBackendHandler.requests if path == "/v1/search-sessions"
        )
        self.assertNotIn("hyperparameters", payload)

    def test_resume_skips_completed_groups(self) -> None:
        group = self._group()
        FakeBackendHandler.video_priorities = [
            {"video_id": "ground-truth", "priority_score": 0.9, "event_coverage": 2, "normalized_coverage": 1.0, "mean_best_event_score": 0.9, "min_best_event_score": 0.9},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            run_tuple_benchmark(
                [group], output_dir, self._config("adaptive_coarse"), {"type": "fixture"}, "digest", progress_every=0
            )
            request_count = len(FakeBackendHandler.requests)
            metrics, rows = run_tuple_benchmark(
                [group],
                output_dir,
                self._config("adaptive_coarse"),
                {"type": "fixture"},
                "digest",
                resume=True,
                progress_every=0,
            )
            self.assertEqual(len(FakeBackendHandler.requests), request_count)
            self.assertEqual(rows[0]["status"], "ok")
            for artifact in ("query_results.jsonl", "query_results.csv", "metrics.json", "run_manifest.json"):
                self.assertTrue((Path(output_dir) / artifact).is_file())


if __name__ == "__main__":
    unittest.main()
