"""Aggregate region_tuple_experiment.py's rows.jsonl into a before/after
comparison table, using the repo's existing recall@k_new / final_query_score
definitions (boundary_metrics.py) plus a rank-only recall@k view.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from .boundary_metrics import DEFAULT_KS, final_query_score, recall_at_k_new


def _rank_recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def summarize(rows: list[dict[str, Any]], label: str, rank_key: str, hits_key: str) -> dict[str, Any]:
    ranks = [row[rank_key] for row in rows]
    hits = [row[hits_key] for row in rows]
    summary: dict[str, Any] = {"label": label, "query_count": len(rows)}
    for k in DEFAULT_KS:
        summary[f"recall_at_{k}"] = _rank_recall_at_k(ranks, k)
    summary["mrr"] = statistics.fmean(1.0 / rank if rank is not None else 0.0 for rank in ranks)
    found = [rank for rank in ranks if rank is not None]
    summary["ground_truth_found_count"] = len(found)
    summary["median_rank_found"] = statistics.median(found) if found else None
    summary["mean_hits"] = statistics.fmean(hits)
    summary["final_query_score_mean"] = statistics.fmean(
        final_query_score(rank, hit, DEFAULT_KS) for rank, hit in zip(ranks, hits)
    )
    return summary


def build_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = [summarize(rows, "baseline (prioritize_videos)", "rank_baseline", "hits_baseline")]
    variant_labels = list(rows[0]["variants"].keys())
    for label in variant_labels:
        variant_rows = [
            {"rank": row["variants"][label]["rank"], "hits": row["variants"][label]["hits"]}
            for row in rows
        ]
        summaries.append(summarize(variant_rows, label, "rank", "hits"))
    return summaries


def format_table(summaries: list[dict[str, Any]]) -> str:
    headers = ["config", "recall@1", "recall@5", "recall@20", "recall@50", "recall@100",
               "mrr", "median_rank_found", "mean_hits/4", "final_query_score"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for s in summaries:
        lines.append(" | ".join([
            s["label"],
            f'{s["recall_at_1"]:.3f}', f'{s["recall_at_5"]:.3f}', f'{s["recall_at_20"]:.3f}',
            f'{s["recall_at_50"]:.3f}', f'{s["recall_at_100"]:.3f}',
            f'{s["mrr"]:.4f}',
            f'{s["median_rank_found"]:.0f}' if s["median_rank_found"] is not None else "n/a",
            f'{s["mean_hits"]:.3f}',
            f'{s["final_query_score_mean"]:.4f}',
        ]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    rows = [json.loads(line) for line in Path(args.rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries = build_report(rows)
    print(format_table(summaries))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
