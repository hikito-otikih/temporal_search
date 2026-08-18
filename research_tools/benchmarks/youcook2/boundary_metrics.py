"""recall@k_new: true_video(k) * (number of event frames that land in ground truth).

User-specified formula:
    recall@k_new = true_video * (number of frames that lie in ground truth)
    true_video = 1 if the video_id matches ground truth video (within top-k)
    final query score = mean of recall@k for k in {1, 5, 20, 50, 100}

Note on the closed form: refinement only relocates a timestamp *within* an
already-identified video - it never changes video rank. So `rank` (and
therefore which k-thresholds a query "clears") is identical before and after
refinement; only `hits` (frame_hits) can differ between conditions. That
means recall_at_k_new(rank, hits, k) = hits if rank <= k else 0 - hits is
computed once per query, not per k.
"""

from __future__ import annotations

import statistics
from typing import Mapping, Sequence

from .core import VideoQueryGroup, event_timestamp_accuracy

DEFAULT_KS: tuple[int, ...] = (1, 5, 20, 50, 100)


def frame_hits(group: VideoQueryGroup, event_anchors: Mapping[str, float]) -> int:
    """Count of events whose anchor timestamp falls inside its GT interval."""

    accuracy = event_timestamp_accuracy(group, event_anchors)
    return sum(1 for hit in accuracy.values() if hit)


def recall_at_k_new(rank: int | None, hits: int, k: int) -> int:
    return hits if (rank is not None and rank <= k) else 0


def final_query_score(rank: int | None, hits: int, ks: Sequence[int] = DEFAULT_KS) -> float:
    return statistics.fmean(recall_at_k_new(rank, hits, k) for k in ks)


def aggregate_boundary_metrics(
    rows: Sequence[Mapping[str, object]], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float | int]:
    """Summarize a list of `{"rank": int|None, "hits": int}` rows.

    One call per `(pipeline, condition)` pair - the natural unit for the
    before/after comparison table.
    """

    if not rows:
        return {
            **{f"recall_at_{k}_new_mean": 0.0 for k in ks},
            "final_query_score_mean": 0.0,
            "query_count": 0,
        }
    summary: dict[str, float | int] = {
        f"recall_at_{k}_new_mean": statistics.fmean(
            recall_at_k_new(row["rank"], row["hits"], k) for row in rows  # type: ignore[arg-type]
        )
        for k in ks
    }
    summary["final_query_score_mean"] = statistics.fmean(
        final_query_score(row["rank"], row["hits"], ks) for row in rows  # type: ignore[arg-type]
    )
    summary["query_count"] = len(rows)
    return summary
