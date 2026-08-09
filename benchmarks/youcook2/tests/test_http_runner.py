from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.youcook2.client import SearchClient
from benchmarks.youcook2.core import QueryRecord
from benchmarks.youcook2.runner import RunConfig, run_benchmark


class FakeSearchHandler(BaseHTTPRequestHandler):
    search_payloads: list[dict[str, object]] = []

    def log_message(self, *_args: object) -> None:
        return

    def _json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path == "/health":
            self._json({"status": "ok", "model": "fake-siglip", "n_vectors": 3})
        else:
            self._json({"detail": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).search_payloads.append(payload)
        if self.path != "/search":
            self._json({"detail": "not found"}, 404)
            return
        self._json(
            {
                "query": payload["query"],
                "english_query": "translated query",
                "top_k": payload["top_k"],
                "results": [
                    {"video_name": "distractor.mp4", "frame_index": 1, "timestamp": "0:01", "score": 0.9},
                    {"video_name": "ground-truth.mp4", "frame_index": 2, "timestamp": "0:02", "score": 0.8},
                    {"video_name": "ground-truth.mp4", "frame_index": 3, "timestamp": "0:03", "score": 0.7},
                ],
            }
        )


class HttpAndRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FakeSearchHandler.search_payloads = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSearchHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        FakeSearchHandler.search_payloads.clear()

    def test_client_uses_exact_sanitized_payload(self) -> None:
        response = SearchClient(self.base_url, retries=0).search("cắt hành tây", top_k=20)
        self.assertEqual(len(response.hits), 3)
        self.assertEqual(response.english_query, "translated query")
        self.assertEqual(FakeSearchHandler.search_payloads, [{"query": "cắt hành tây", "top_k": 20}])

    def test_runner_writes_artifacts_and_resume_skips_success(self) -> None:
        record = QueryRecord(
            query_id="ground-truth:E1",
            query_text="slice an onion",
            ground_truth_video="ground-truth",
            source="fixture.txt",
            event_id="E1",
        )
        config = RunConfig(
            base_url=self.base_url,
            frame_top_k=20,
            recall_ks=(1, 2, 5),
            aggregation="max",
            top_m=3,
            temperature=1.0,
            timeout_seconds=2.0,
            retries=0,
            retry_backoff_seconds=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            metrics, rows = run_benchmark(
                [record], temporary, config, {"type": "fixture"}, "digest", progress_every=0
            )
            self.assertEqual(rows[0]["rank"], 2)
            self.assertEqual(metrics["recall_at_1"], 0.0)
            self.assertEqual(metrics["recall_at_2"], 1.0)
            self.assertTrue((Path(temporary) / "query_results.jsonl").is_file())
            self.assertTrue((Path(temporary) / "query_results.csv").is_file())
            self.assertTrue((Path(temporary) / "metrics.json").is_file())
            self.assertTrue((Path(temporary) / "run_manifest.json").is_file())

            search_calls = len(FakeSearchHandler.search_payloads)
            run_benchmark(
                [record],
                temporary,
                config,
                {"type": "fixture"},
                "digest",
                resume=True,
                progress_every=0,
            )
            self.assertEqual(len(FakeSearchHandler.search_payloads), search_calls)

        self.assertEqual(
            FakeSearchHandler.search_payloads[0],
            {"query": "slice an onion", "top_k": 20},
        )
        self.assertNotIn("ground_truth_video", FakeSearchHandler.search_payloads[0])


if __name__ == "__main__":
    unittest.main()
