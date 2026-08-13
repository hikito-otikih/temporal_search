"""Experimental Branch-A alternative: multi-region pooling + order-aware tuple
scoring, benchmarked against the existing mean-only `prioritize_videos()`.

Motivation (session discussion, not yet validated): `prioritize_videos()`
takes an independent per-event max over `TemporalRegion.normalized_coarse_score`,
discarding every candidate region but the single best one per event - so
`top_n_fused=1000` and clustering's own region pool are barely exploited, and
there is no reward for a video whose regions land in query order versus one
where they collide or invert. This module tests whether pooling more regions
per event and scoring same-video combinations jointly (mean region score +
an order-agreement term) changes video rank or ground-truth timestamp
accuracy relative to today's production ranking.

Deliberately lives in the benchmark tree, not `src/adaptive_search/`, until
proven better here - same convention as `coarse_anchor.py`/`boundary_refinement.py`.
Reuses production `cluster_temporal_regions()`/`prioritize_videos()` for the
regions themselves (only the video-scoring step on top of those regions is new),
and the existing `frame_hits()`/`event_timestamp_accuracy()` ground-truth harness.

Sanity invariant (also unit-tested): at `order_weight=0.0`, the best tuple's
mean region score for a video is mathematically identical to
`prioritize_videos()`'s `mean_best_event_score` - a candidate pool always
contains that event's single best region (delta-threshold with delta>=0 can
only add options, never remove the top one), so the unconstrained best
combination is just "each event's own best region," which is exactly what
`prioritize_videos()` already computes independently. Rank should therefore
match production byte-for-byte at order_weight=0, modulo tie-breaks - this is
the primary correctness check that the new code isn't silently different from
the old code for the trivial case.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from adaptive_search.schemas import SparseCandidate, TemporalRegion


@dataclass(frozen=True)
class RegionTupleParams:
    """Sweepable hyperparameters for the multi-region/tuple ranking variant."""

    absolute_threshold: float = 0.0
    """alpha: a region must score >= this on normalized_coarse_score (0..1)."""

    relative_delta: float = 0.15
    """delta: a region must score >= (event's best region score - delta)."""

    max_regions_per_event: int = 20
    """N: cap on pooled regions per (event, video), best-score-first."""

    order_weight: float = 0.1
    """Weight of the order-agreement term added to the tuple's mean region
    score. 0.0 disables order-awareness entirely (see module docstring's
    sanity invariant)."""

    pooling: str = "max"
    """How a video's kept tuples combine into one video score: "max" (best
    tuple wins, directly comparable to today's argmax-based ranking) or
    "mean" (rewards videos with many good tuples, not just one)."""

    max_combinations_per_video: int = 20000
    """Hard cap on tuples explored per video during backtracking - a safety
    valve against the N^event_count combinatorial blowup the proposal itself
    flagged as a risk. Search order is best-region-first per event, so this
    cap truncates from the least-promising end first."""

    max_tuples_per_video: int = 20
    """How many top-scoring tuples per video to retain (only matters for
    pooling="mean" and for exposing more than the top-1 to a caller)."""

    confidence_gate: str = "none"
    """How `order_weight` is scaled by the tuple's own `region_mean_score`
    before being applied, addressing the finding that the order bonus can
    mislead when the underlying region scores are themselves weak/noisy:
    "none" (today's fixed order_weight), "linear" (order_weight *
    region_mean_score - continuous, no extra hyperparameter), or
    "threshold" (order_weight if region_mean_score >= confidence_gate_threshold
    else 0.0 - a hard cutoff)."""

    confidence_gate_threshold: float = 0.5
    """Only used when confidence_gate="threshold"."""

    def __post_init__(self) -> None:
        if self.absolute_threshold < 0.0 or self.absolute_threshold > 1.0:
            raise ValueError("absolute_threshold must be within [0, 1]")
        if self.relative_delta < 0.0:
            raise ValueError("relative_delta must be non-negative")
        if self.max_regions_per_event < 1:
            raise ValueError("max_regions_per_event must be >= 1")
        if self.order_weight < 0.0:
            raise ValueError("order_weight must be non-negative")
        if self.pooling not in ("max", "mean"):
            raise ValueError('pooling must be "max" or "mean"')
        if self.max_combinations_per_video < 1:
            raise ValueError("max_combinations_per_video must be >= 1")
        if self.max_tuples_per_video < 1:
            raise ValueError("max_tuples_per_video must be >= 1")
        if self.confidence_gate not in ("none", "linear", "threshold"):
            raise ValueError('confidence_gate must be "none", "linear", or "threshold"')
        if self.confidence_gate_threshold < 0.0 or self.confidence_gate_threshold > 1.0:
            raise ValueError("confidence_gate_threshold must be within [0, 1]")


def _effective_order_weight(params: "RegionTupleParams", region_mean_score: float) -> float:
    if params.confidence_gate == "none":
        return params.order_weight
    if params.confidence_gate == "linear":
        return params.order_weight * region_mean_score
    return params.order_weight if region_mean_score >= params.confidence_gate_threshold else 0.0


@dataclass(frozen=True)
class RegionTupleResult:
    video_id: str
    score: float
    region_mean_score: float
    order_score: float
    region_ids: tuple[str | None, ...]
    timestamps: tuple[float | None, ...]


def pool_event_regions(
    regions: Sequence[TemporalRegion],
    *,
    event_id: str,
    video_id: str,
    params: RegionTupleParams,
) -> list[TemporalRegion]:
    """Regions surviving both the absolute and relative-delta thresholds,
    best-score-first, capped at `max_regions_per_event`."""

    matching = [
        region
        for region in regions
        if region.event_id == event_id and region.video_id == video_id
    ]
    if not matching:
        return []
    ranked = sorted(matching, key=lambda region: -(region.normalized_coarse_score or 0.0))
    best_score = ranked[0].normalized_coarse_score or 0.0
    survivors = [
        region
        for region in ranked
        if (region.normalized_coarse_score or 0.0) >= params.absolute_threshold
        and (region.normalized_coarse_score or 0.0) >= best_score - params.relative_delta
    ]
    return survivors[: params.max_regions_per_event]


def _region_timestamp(
    region: TemporalRegion, candidates_by_id: Mapping[str, SparseCandidate]
) -> float:
    """Representative timestamp: the region's own best-scoring candidate -
    the same quantity `select_event_seeds()` would return as seeds[0] for
    this region alone."""

    members = [
        candidates_by_id[candidate_id]
        for candidate_id in region.candidate_ids
        if candidate_id in candidates_by_id
    ]
    if not members:
        return (region.start_seconds + region.end_seconds) / 2.0
    return max(members, key=lambda candidate: candidate.raw_relevance_score).timestamp_seconds


def _order_score(
    timestamps: Sequence[float | None],
    constraints: Sequence[tuple[int, int]] | None = None,
) -> float:
    """(# satisfied constraints - # violated constraints) / (# constraints
    actually checked), in [-1, 1].

    `constraints` is a list of (predecessor_index, successor_index) pairs
    into `timestamps`, each meaning "predecessor is expected to have the
    smaller timestamp" - the DIRECTION comes from `temporal_relation`
    (`after`/`reference_event_id`), not from list position. This is
    deliberately NOT "adjacent pairs in query-list order": an event's
    `reference_event_id` can point anywhere, and "after" vs "before" can
    make an earlier-listed event the expected *successor* of a later-listed
    one. Getting this from list position instead (an earlier version of this
    module did) would silently score exactly backwards whenever a query's
    array order doesn't match its claimed chronology.

    None defaults to the adjacent-list-position chain
    `[(0,1), (1,2), ..., (n-2,n-1)]` - today's blanket behavior, used when no
    `temporal_relation` classification is available for a video.

    An event with no covering region (`None` timestamp) drops any
    constraint touching it - an uncovered event imposes no ordering
    constraint on its neighbors, it just isn't part of the sequence being
    checked. Ties (equal timestamps) count as violations, not credit -
    events are expected to be visually distinct instants."""

    if constraints is None:
        constraints = [(i, i + 1) for i in range(max(0, len(timestamps) - 1))]
    pairs = [
        (timestamps[predecessor], timestamps[successor])
        for predecessor, successor in constraints
        if timestamps[predecessor] is not None and timestamps[successor] is not None
    ]
    if not pairs:
        return 0.0
    correct = sum(1 for a, b in pairs if b > a)
    return (2 * correct - len(pairs)) / len(pairs)


def assemble_region_tuples_for_video(
    video_id: str,
    event_ids: Sequence[str],
    regions: Sequence[TemporalRegion],
    candidates_by_id: Mapping[str, SparseCandidate],
    params: RegionTupleParams,
    order_constraints: Sequence[tuple[int, int]] | None = None,
) -> list[RegionTupleResult]:
    """All (bounded) same-video region combinations, one region per event,
    scored by mean region score plus an order-agreement bonus/penalty.

    An event with zero surviving regions in this video is not fatal to the
    whole video (matching `prioritize_videos()`'s zero-fill semantics for an
    uncovered event, not this module's own separate all-or-nothing choice):
    it contributes a fixed 0.0 to the mean and is skipped when checking
    order, rather than eliminating the video from tuple ranking entirely.
    A video with zero covered events in total still yields no tuples, same
    as `prioritize_videos()` never ranking it at all.

    `order_constraints`: see `_order_score` - (predecessor_index,
    successor_index) pairs from `temporal_relation`, not list position.
    None means the adjacent-list-position chain (today's blanket behavior).
    """

    pools: list[list[TemporalRegion | None]] = []
    for event_id in event_ids:
        pool = pool_event_regions(regions, event_id=event_id, video_id=video_id, params=params)
        pools.append(list(pool) if pool else [None])
    if all(pool == [None] for pool in pools):
        return []

    results: list[RegionTupleResult] = []
    selected: list[TemporalRegion | None] = []
    combinations_seen = 0

    def backtrack(index: int) -> None:
        nonlocal combinations_seen
        if combinations_seen >= params.max_combinations_per_video:
            return
        if index == len(event_ids):
            combinations_seen += 1
            timestamps = tuple(
                _region_timestamp(region, candidates_by_id) if region is not None else None
                for region in selected
            )
            scores = [region.normalized_coarse_score or 0.0 if region is not None else 0.0 for region in selected]
            region_mean = fmean(scores)
            order = _order_score(timestamps, order_constraints)
            final = region_mean + _effective_order_weight(params, region_mean) * order
            results.append(
                RegionTupleResult(
                    video_id=video_id,
                    score=final,
                    region_mean_score=region_mean,
                    order_score=order,
                    region_ids=tuple(region.id if region is not None else None for region in selected),
                    timestamps=timestamps,
                )
            )
            return
        for region in pools[index]:
            if combinations_seen >= params.max_combinations_per_video:
                return
            selected.append(region)
            backtrack(index + 1)
            selected.pop()

    backtrack(0)
    results.sort(key=lambda item: (-item.score, item.region_ids))
    return results[: params.max_tuples_per_video]


def rank_videos_by_region_tuples(
    regions: Sequence[TemporalRegion],
    candidates_by_id: Mapping[str, SparseCandidate],
    event_ids: Sequence[str],
    params: RegionTupleParams,
    order_constraints: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[tuple[str, float]], dict[str, list[RegionTupleResult]]]:
    """Videos ranked by pooled tuple score, plus each video's kept tuples
    (so a caller can also read off the winning tuple's per-event anchors)."""

    video_ids = sorted({region.video_id for region in regions})
    video_scores: list[tuple[str, float]] = []
    tuples_by_video: dict[str, list[RegionTupleResult]] = {}
    for video_id in video_ids:
        tuples = assemble_region_tuples_for_video(
            video_id, event_ids, regions, candidates_by_id, params, order_constraints
        )
        if not tuples:
            continue
        tuples_by_video[video_id] = tuples
        score = tuples[0].score if params.pooling == "max" else fmean(t.score for t in tuples)
        video_scores.append((video_id, score))
    video_scores.sort(key=lambda item: (-item[1], item[0]))
    return video_scores, tuples_by_video
