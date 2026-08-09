from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from benchmarks.youcook2.client import SearchApiError, SearchClient, SearchResponse
from benchmarks.youcook2.core import QueryRecord
from benchmarks.youcook2.runner import RunConfig, run_benchmark


def _record(query_text: str = "slice an onion") -> QueryRecord:
    return QueryRecord(
        query_id="ground-truth:E1",
        query_text=query_text,
        ground_truth_video="ground-truth",
        source="fixture.txt",
        event_id="E1",
    )


def _config(*, aggregation: str = "max") -> RunConfig:
    return RunConfig(
        base_url="http://benchmark.invalid",
        frame_top_k=20,
        recall_ks=(1, 5),
        aggregation=aggregation,
        top_m=3,
        temperature=1.0,
        timeout_seconds=2.0,
        retries=0,
        retry_backoff_seconds=0.0,
    )


class _StubClient:
    def __init__(
        self,
        health: Mapping[str, Any] | None = None,
    ) -> None:
        self.health_payload = dict(
            health
            or {
                "status": "ok",
                "model": "fake-siglip",
                "model_revision": "commit-a",
                "index_hash": "index-a",
                "n_vectors": 3,
            }
        )
        self.health_calls = 0
        self.search_calls: list[tuple[str, int]] = []

    def health(self) -> Mapping[str, Any]:
        self.health_calls += 1
        return dict(self.health_payload)

    def search(self, query: str, top_k: int) -> SearchResponse:
        self.search_calls.append((query, top_k))
        return SearchResponse(
            hits=[
                {"video_name": "ground-truth.mp4", "score": 0.9},
                {"video_name": "distractor.mp4", "score": 0.8},
            ],
            response_query=query,
            english_query=query,
            response_top_k=top_k,
        )


class RunnerIntegrityTests(unittest.TestCase):
    def test_force_resume_is_rejected_instead_of_mixing_old_successes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = _StubClient()
            run_benchmark(
                [_record()],
                temporary,
                _config(),
                {"type": "fixture"},
                "source-a",
                client=first,
                progress_every=0,
            )

            second = _StubClient()
            with self.assertRaisesRegex(ValueError, "force-resume is disabled"):
                run_benchmark(
                    [_record("changed query")],
                    temporary,
                    _config(aggregation="top_m_mean"),
                    {"type": "fixture"},
                    "source-b",
                    resume=True,
                    force_resume=True,
                    client=second,
                    progress_every=0,
                )
            self.assertEqual(second.health_calls, 0)
            self.assertEqual(second.search_calls, [])

    def test_resume_rejects_backend_model_or_index_drift_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = _StubClient()
            run_benchmark(
                [_record()],
                temporary,
                _config(),
                {"type": "fixture"},
                "source-a",
                client=first,
                progress_every=0,
            )
            manifest_path = Path(temporary) / "run_manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")

            changed = _StubClient(
                {
                    "status": "ok",
                    "model": "fake-siglip",
                    "model_revision": "commit-a",
                    "index_hash": "index-b",
                    "n_vectors": 3,
                }
            )
            with self.assertRaisesRegex(ValueError, "backend identity differs"):
                run_benchmark(
                    [_record()],
                    temporary,
                    _config(),
                    {"type": "fixture"},
                    "source-a",
                    resume=True,
                    client=changed,
                    progress_every=0,
                )
            self.assertEqual(changed.search_calls, [])
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"),
                original_manifest,
            )

    def test_resume_rejects_source_or_configuration_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = _StubClient()
            run_benchmark(
                [_record()],
                temporary,
                _config(),
                {"type": "fixture"},
                "source-a",
                client=first,
                progress_every=0,
            )
            second = _StubClient()
            with self.assertRaisesRegex(
                ValueError,
                "configuration/source/backend identity differs",
            ):
                run_benchmark(
                    [_record("changed query")],
                    temporary,
                    _config(aggregation="top_m_mean"),
                    {"type": "fixture"},
                    "source-b",
                    resume=True,
                    client=second,
                    progress_every=0,
                )
            self.assertEqual(second.search_calls, [])

    def test_resume_repairs_partial_jsonl_tail_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            client = _StubClient()
            run_benchmark(
                [_record()],
                temporary,
                _config(),
                {"type": "fixture"},
                "source-a",
                client=client,
                progress_every=0,
            )
            checkpoint = Path(temporary) / "query_results.jsonl"
            prior = json.loads(checkpoint.read_text(encoding="utf-8"))
            prior["status"] = "error"
            prior["error"] = "synthetic interrupted attempt"
            checkpoint.write_text(
                json.dumps(prior, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with checkpoint.open("ab") as handle:
                handle.write(b'{"query_id":"partial"')

            calls_before_retry = len(client.search_calls)
            run_benchmark(
                [_record()],
                temporary,
                _config(),
                {"type": "fixture"},
                "source-a",
                resume=True,
                client=client,
                progress_every=0,
            )
            self.assertEqual(len(client.search_calls), calls_before_retry + 1)
            payload = checkpoint.read_bytes()
            self.assertTrue(payload.endswith(b"\n"))
            rows = [
                json.loads(line)
                for line in payload.decode("utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["query_id"] == _record().query_id for row in rows))
            self.assertEqual(rows[-1]["status"], "ok")

            # A second resume proves the repaired checkpoint stays parseable.
            run_benchmark(
                [_record()],
                temporary,
                _config(),
                {"type": "fixture"},
                "source-a",
                resume=True,
                client=client,
                progress_every=0,
            )

    def test_resume_requires_checkpoint_and_manifest_as_a_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for present_name in ("query_results.jsonl", "run_manifest.json"):
                with self.subTest(present_name=present_name):
                    for artifact in root.iterdir():
                        artifact.unlink()
                    (root / present_name).write_text(
                        "{}\n" if present_name.endswith(".jsonl") else "{}",
                        encoding="utf-8",
                    )
                    client = _StubClient()
                    with self.assertRaisesRegex(ValueError, "only one"):
                        run_benchmark(
                            [_record()],
                            root,
                            _config(),
                            {"type": "fixture"},
                            "source-a",
                            resume=True,
                            client=client,
                            progress_every=0,
                        )
                    self.assertEqual(client.health_calls, 0)
            for artifact in root.iterdir():
                artifact.unlink()
            empty_client = _StubClient()
            with self.assertRaisesRegex(ValueError, "requires both"):
                run_benchmark(
                    [_record()],
                    root,
                    _config(),
                    {"type": "fixture"},
                    "source-a",
                    resume=True,
                    client=empty_client,
                    progress_every=0,
                )
            self.assertEqual(empty_client.health_calls, 0)

    def test_search_client_rejects_top_k_contract_violations(self) -> None:
        client = SearchClient("http://127.0.0.1:8000", retries=0)
        base = {
            "query": "query",
            "top_k": 1,
            "results": [{"video_name": "video.mp4", "score": 1.0}],
        }
        with patch.object(client, "_request_json", return_value={**base, "top_k": 2}):
            with self.assertRaisesRegex(SearchApiError, "echoed top_k"):
                client.search("query", 1)
        with patch.object(client, "_request_json", return_value={**base, "top_k": 1.5}):
            with self.assertRaisesRegex(SearchApiError, "invalid top_k"):
                client.search("query", 1)
        with patch.object(
            client,
            "_request_json",
            return_value={
                **base,
                "results": [
                    {"video_name": "a.mp4", "score": 1.0},
                    {"video_name": "b.mp4", "score": 0.9},
                ],
            },
        ):
            with self.assertRaisesRegex(SearchApiError, "returned 2 results"):
                client.search("query", 1)


if __name__ == "__main__":
    unittest.main()
