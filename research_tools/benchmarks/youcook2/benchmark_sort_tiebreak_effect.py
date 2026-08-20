"""One-shot benchmark: does the video-priorities sort tie-break change
(priority_score -> (priority_score, raw_score, event_coverage, video_id),
see router/video_priorities.py) move recall@K/MRR versus the previous sort
key (priority_score, video_id only)?

Runs "baseline" (ground-truth event text straight into the adaptive
session, no rewrite - same condition run_rewrite_pipeline_benchmark.py
calls "baseline") against the live backend, which must already be running
the new sort code (this script does not restart it). For each query, one
GET .../video-priorities call at the requested top_k/top_n_per_variant/
ranking-limit returns the FULL item list including priority_score,
raw_score, and event_coverage; from that single list this script derives
two ranks of the ground-truth video:
  - "new_sort": rank in the order the server actually returned (the live,
    new tie-break)
  - "old_sort": rank after re-sorting the SAME items locally by the old key
    (-priority_score, video_id)
Reconstructing old_sort locally - rather than re-running the whole pipeline
against a pre-change server - isolates the sort key as the only variable:
no second network round-trip, no risk of retrieval nondeterminism
confounding the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .core import canonical_video_id, compute_metrics, load_query_directory_grouped
from .tuple_client import TemporalSearchBackendClient
from .tuple_runner import TupleRunConfig, _adaptive_hyperparameters, _build_event_payload


def _rank_with_key(items: list[dict[str, Any]], target: str, key) -> int | None:
    ordered = sorted(items, key=key)
    for position, item in enumerate(ordered, 1):
        if canonical_video_id(str(item["video_id"])) == target:
            return position
    return None


def run(
    *, query_dir: Path, cache_path: Path | None, base_url: str, recall_ks: tuple[int, ...],
    limit: int | None, top_k: int, top_n_per_variant: int, ranking_top_k: int,
) -> None:
    groups = load_query_directory_grouped(query_dir)
    if cache_path is not None:
        cached_ids = {
            json.loads(line)["video_id"]
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and "error" not in json.loads(line)
        }
        groups = [g for g in groups if g.video_id in cached_ids]
    if limit is not None:
        groups = groups[:limit]
    print(
        f"{len(groups)} videos; top_k={top_k} top_n_per_variant={top_n_per_variant} "
        f"ranking_top_k={ranking_top_k}"
    )

    config = TupleRunConfig(
        backend_base_url=base_url, pipeline="adaptive_coarse", top_k_tuple=100,
        top_k_each_query=100, gamma=0.05, adaptive_top_k=top_k, recall_ks=recall_ks,
        timeout_seconds=120.0, retries=2, retry_backoff_seconds=0.5,
        adaptive_ranking_top_k=ranking_top_k, adaptive_top_n_per_variant=top_n_per_variant,
    )
    backend = TemporalSearchBackendClient(
        base_url=base_url, timeout_seconds=config.timeout_seconds, retries=config.retries,
    )

    rows_new: list[dict[str, Any]] = []
    rows_old: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, group in enumerate(groups, 1):
        try:
            session_id = backend.create_adaptive_session(
                _build_event_payload(group), common_query=group.context,
                hyperparameters=_adaptive_hyperparameters(config),
            )
            backend.retrieve(session_id, top_k=config.adaptive_top_k)
            priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)

            rank_new = None
            for position, item in enumerate(priorities, 1):
                if canonical_video_id(str(item["video_id"])) == group.video_id:
                    rank_new = position
                    break
            rank_old = _rank_with_key(
                priorities, group.video_id, key=lambda it: (-it["priority_score"], it["video_id"]),
            )
            rows_new.append({"status": "ok", "rank": rank_new, "unique_video_count": len(priorities)})
            rows_old.append({"status": "ok", "rank": rank_old, "unique_video_count": len(priorities)})
            if rank_new != rank_old:
                moved.append(
                    {"video_id": group.video_id, "rank_old_sort": rank_old, "rank_new_sort": rank_new}
                )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            rows_new.append({"status": "error", "rank": None, "error": error})
            rows_old.append({"status": "error", "rank": None, "error": error})
        if index % 10 == 0 or index == len(groups):
            print(f"  {index}/{len(groups)} videos evaluated ({time.monotonic()-started:.0f}s elapsed)")

    metrics_new = compute_metrics(rows_new, recall_ks)
    metrics_old = compute_metrics(rows_old, recall_ks)
    print("\n=== new_sort (priority_score, raw_score, event_coverage, video_id) ===")
    print(json.dumps(metrics_new, indent=2))
    print("\n=== old_sort (priority_score, video_id) ===")
    print(json.dumps(metrics_old, indent=2))
    print(f"\n{len(moved)}/{len(groups)} queries had a different ground-truth rank under the two sort keys:")
    for row in moved:
        print(f"  {row}")

    print("\n=== SIDE BY SIDE ===")
    for k in sorted(recall_ks):
        key = f"recall_at_{k}"
        print(f"  recall@{k:<3d} old_sort={metrics_old[key]!s:8}  new_sort={metrics_new[key]!s:8}")
    print(f"  mrr       old_sort={metrics_old['mrr']!s:8}  new_sort={metrics_new['mrr']!s:8}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/query")
    parser.add_argument("--cache", default="runs/rewrite_output_cache.jsonl", help="restrict to this cache's video cohort; empty string disables")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--recall-k", default="1,5,10,20,50,100")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-n-per-variant", type=int, default=1000)
    parser.add_argument("--ranking-top-k", type=int, default=1000)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run(
        query_dir=Path(args.query_dir),
        cache_path=Path(args.cache) if args.cache else None,
        base_url=args.base_url,
        recall_ks=tuple(int(k) for k in args.recall_k.split(",")),
        limit=args.limit, top_k=args.top_k, top_n_per_variant=args.top_n_per_variant,
        ranking_top_k=args.ranking_top_k,
    )


if __name__ == "__main__":
    main()
