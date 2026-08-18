"""YouCook2 Video Recall@K evaluation support.

Leakage guard
-------------
Ground truth arrives only as ``video_path`` in the manifest.  The evaluator
derives ``ground_truth_video_id`` locally and never serializes it into a
retrieval payload.  ``build_retrieval_payload`` is the only place that builds
search requests, and its output is unit tested to contain exactly the event
text and top-K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

VideoAggregation = Callable[[list[float]], float]


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryItem:
    query: str
    ground_truth_video_id: str | None = None
    video_path: str | None = None
    index: int = 0


class ManifestError(ValueError):
    pass


def parse_manifest(data: Any) -> list[QueryItem]:
    """Parse a YouCook2 query manifest into ``QueryItem`` values.

    Accepts either ``{"queries": [...]}`` or a bare list.  Each item may be a
    string (no ground truth) or an object with ``query`` text and an optional
    ``video_path`` used only by the evaluator.
    """
    if isinstance(data, dict):
        items = data.get("queries")
        if items is None:
            raise ManifestError("manifest object must contain a 'queries' array")
    elif isinstance(data, list):
        items = data
    else:
        raise ManifestError("manifest must be a JSON array or an object")

    parsed: list[QueryItem] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            text = item.strip()
            parsed.append(QueryItem(query=text, index=index))
            continue
        if not isinstance(item, dict):
            raise ManifestError(f"manifest item {index} must be a string or object")
        text = str(item.get("query") or item.get("text") or "").strip()
        if not text:
            raise ManifestError(f"manifest item {index} has an empty query")
        video_path = item.get("video_path") or item.get("video_name")
        parsed.append(
            QueryItem(
                query=text,
                video_path=str(video_path) if video_path else None,
                ground_truth_video_id=_video_id_from_path(video_path),
                index=index,
            )
        )
    return parsed


def _video_id_from_path(video_path: Any) -> str | None:
    if not video_path:
        return None
    name = str(video_path).replace("\\", "/").split("/")[-1]
    for ext in (".mp4", ".avi", ".mkv", ".webm"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    name = name.strip()
    return name or None


def build_retrieval_payload(query: str, top_k: int) -> dict[str, Any]:
    """The ONLY sanctioned retrieval payload: event text and top-K only."""
    return {"query": query, "top_k": top_k}


# ---------------------------------------------------------------------------
# Frame aggregation and dedup
# ---------------------------------------------------------------------------

def deduplicate_ranked_videos(frame_results: Sequence[dict[str, Any]]) -> list[str]:
    """Rank unique videos from upstream frame hits, strongest frame first.

    Frame hits are grouped by ``video_name`` and collapsed so the corpus-level
    ranking never contains duplicate videos.
    """
    by_video: dict[str, float] = {}
    for hit in frame_results:
        video_id = hit.get("video_name") or hit.get("video_id")
        if not video_id:
            continue
        score = float(hit.get("score", 0.0))
        by_video[video_id] = max(by_video.get(video_id, float("-inf")), score)
    return sorted(by_video, key=lambda item: (-by_video[item], item))


def aggregate_video_scores(
    frame_results: Sequence[dict[str, Any]], method: str = "max", top_m: int = 3
) -> dict[str, float]:
    """Aggregate frame scores per video using the selected method."""
    per_video: dict[str, list[float]] = {}
    for hit in frame_results:
        video_id = hit.get("video_name") or hit.get("video_id")
        if not video_id:
            continue
        per_video.setdefault(video_id, []).append(float(hit.get("score", 0.0)))

    aggregator = _aggregator_for(method)
    ranked: dict[str, float] = {}
    for video_id, scores in per_video.items():
        ranked[video_id] = aggregator(scores, top_m)
    return ranked


def _aggregator_for(method: str):
    def max_score(scores: list[float], top_m: int) -> float:
        return max(scores)

    def mean_top_m(scores: list[float], top_m: int) -> float:
        ordered = sorted(scores, reverse=True)[: max(1, min(top_m, len(scores)))]
        return sum(ordered) / len(ordered)

    def logsumexp(scores: list[float], top_m: int) -> float:
        m = max(scores)
        return m + math.log(sum(math.exp(s - m) for s in scores))

    def rrf(scores: list[float], top_m: int) -> float:
        ranked = sorted(scores, reverse=True)
        return sum(1.0 / (60.0 + rank) for rank, _ in enumerate(ranked, start=1))

    methods: dict[str, Any] = {
        "max": max_score,
        "mean_top_m": mean_top_m,
        "logsumexp": logsumexp,
        "rrf": rrf,
    }
    if method not in methods:
        raise ValueError(f"unknown aggregation method: {method}")
    return methods[method]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rank_of_ground_truth(ranked_videos: Sequence[str], ground_truth: str) -> int | None:
    try:
        return ranked_videos.index(ground_truth) + 1
    except ValueError:
        return None


def recall_at_k(ranks: Iterable[int | None], k: int) -> float:
    values = [rank for rank in ranks if rank is not None and rank <= k]
    total = sum(1 for rank in ranks if rank is not None)
    if total == 0:
        return 0.0
    return len(values) / total


def mean_reciprocal_rank(ranks: Iterable[int | None]) -> float:
    total = 0
    count = 0
    for rank in ranks:
        if rank is not None:
            total += 1.0 / rank
            count += 1
    if count == 0:
        return 0.0
    return total / count


def median_rank(ranks: Iterable[int | None]) -> float | None:
    found = sorted(rank for rank in ranks if rank is not None)
    if not found:
        return None
    middle = len(found) // 2
    if len(found) % 2 == 1:
        return float(found[middle])
    return (found[middle - 1] + found[middle]) / 2.0


def summarize_metrics(ranks: Sequence[int | None], k_list: Sequence[int]) -> dict[str, Any]:
    with_gt = [rank for rank in ranks if rank is not None]
    return {
        "queries_total": len(ranks),
        "queries_with_ground_truth": len(with_gt),
        "missing_index": len(with_gt) - sum(1 for rank in with_gt if rank is not None),
        "missing_ground_truth": len(ranks) - len(with_gt),
        "recall": {int(k): recall_at_k(ranks, int(k)) for k in k_list},
        "mrr": mean_reciprocal_rank(ranks),
        "median_rank": median_rank(ranks),
    }


# ---------------------------------------------------------------------------
# Batched run with checkpoint resume
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRun:
    queries: list[QueryItem]
    method: str = "max"
    top_m: int = 3
    frame_top_n: int = 100
    k_list: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])

    completed: dict[int, dict[str, Any]] = field(default_factory=dict)

    def resume(self) -> list[QueryItem]:
        return [item for item in self.queries if item.index not in self.completed]

    def record(self, query_index: int, ranked_videos: Sequence[str], raw: dict[str, Any]) -> None:
        item = next(q for q in self.queries if q.index == query_index)
        rank = (
            rank_of_ground_truth(ranked_videos, item.ground_truth_video_id)
            if item.ground_truth_video_id
            else None
        )
        self.completed[query_index] = {
            "query_index": query_index,
            "query": item.query,
            "ground_truth_video_id": item.ground_truth_video_id,
            "rank": rank,
            "hit_at": {int(k): bool(rank is not None and rank <= k) for k in self.k_list},
            "ranked_videos": list(ranked_videos),
            "raw": raw,
        }

    def per_query_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.queries:
            record = self.completed.get(item.index)
            if record is None:
                rows.append(
                    {
                        "query_index": item.index,
                        "query": item.query,
                        "ground_truth_video_id": item.ground_truth_video_id,
                        "rank": None,
                        "status": "pending",
                    }
                )
                continue
            rows.append(
                {
                    "query_index": item.index,
                    "query": item.query,
                    "ground_truth_video_id": item.ground_truth_video_id,
                    "rank": record["rank"],
                    "hit@1": record["hit_at"].get(1, False),
                    "hit@5": record["hit_at"].get(5, False),
                    "hit@10": record["hit_at"].get(10, False),
                    "hit@20": record["hit_at"].get(20, False),
                    "status": "completed",
                }
            )
        return rows

    def summary(self) -> dict[str, Any]:
        ranks = [record["rank"] for record in self.completed.values()]
        summary = summarize_metrics(ranks, self.k_list)
        summary["method"] = self.method
        summary["frame_top_n"] = self.frame_top_n
        return summary

    def run_query(
        self,
        item: QueryItem,
        *,
        search: Callable[[str, int], list[dict[str, Any]]],
    ) -> None:
        payload = build_retrieval_payload(item.query, self.frame_top_n)
        frame_results = search(payload["query"], payload["top_k"])
        ranked = rank_videos(frame_results, self.method, self.top_m)
        self.record(item.index, ranked, {"payload": payload, "frame_hits": len(frame_results)})


def rank_videos(
    frame_results: Sequence[dict[str, Any]],
    method: str = "max",
    top_m: int = 3,
) -> list[str]:
    """Return unique videos ranked by the selected frame-score aggregation."""
    scores = aggregate_video_scores(frame_results, method, top_m)
    return sorted(scores, key=lambda item: (-scores[item], item))
