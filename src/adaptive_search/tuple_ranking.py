"""Multi-region pooling + order-aware tuple video ranking.

An opt-in alternative to `algorithms.py::prioritize_videos()`'s independent
per-event max: instead of picking each event's best-scoring region in
isolation, pool up to `max_regions_per_event` candidate regions per event,
assemble same-video combinations across events (bounded backtracking - see
`assemble_region_tuples_for_video`), and score each combination by its mean
region score plus a confidence-gated bonus/penalty for whether its per-event
timestamps land in the expected order.

Ported from `irrelevant_things/benchmarks/youcook2/region_tuple_ranking.py`,
validated there against real YouCook2 queries (n=30,
`region_tuple_ranking_results.md`): +15% final_query_score over
`prioritize_videos()` at this module's default hyperparameters, with one
correctness fix carried over from that report's own Round 3: order checking
uses explicit (predecessor_index, successor_index) constraints, not adjacent
query-list position - list position is only used as the *default* constraint
set (`order_constraints=None`) when no richer relation data is available,
which is the only case exercised in production today (`EventDefinition` does
not yet carry `temporal_relation`/`reference_event_id` - see that report's
"Known scope gaps").
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from .schemas import SparseCandidate, TemporalRegion, TupleRankingHyperparameters


@dataclass(frozen=True)
class TupleRankingResult:
    video_id: str
    score: float
    region_mean_score: float
    order_score: float
    region_ids: tuple[str | None, ...]
    timestamps: tuple[float | None, ...]


def _effective_order_weight(
    params: TupleRankingHyperparameters, region_mean_score: float
) -> float:
    if params.confidence_gate == "none":
        return params.order_weight
    if params.confidence_gate == "linear":
        return params.order_weight * region_mean_score
    return params.order_weight if region_mean_score >= params.confidence_gate_threshold else 0.0


def pool_event_regions(
    regions: Sequence[TemporalRegion],
    *,
    event_id: str,
    video_id: str,
    params: TupleRankingHyperparameters,
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
    the same quantity `boundary_seeds.select_event_seeds()` would return as
    seeds[0] for this region alone."""

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
    smaller timestamp" - the direction comes from an event's actual claimed
    relation to another event, not from list position (see module
    docstring). None defaults to the adjacent-list-position chain
    `[(0,1), (1,2), ..., (n-2,n-1)]`.

    An event with no covering region (`None` timestamp) drops any
    constraint touching it - an uncovered event imposes no ordering
    constraint on its neighbors. Ties (equal timestamps) count as
    violations, not credit - events are expected to be visually distinct
    instants.
    """

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
    params: TupleRankingHyperparameters,
    order_constraints: Sequence[tuple[int, int]] | None = None,
) -> list[TupleRankingResult]:
    """All (bounded) same-video region combinations, one region per event,
    scored by mean region score plus a confidence-gated order-agreement bonus.

    An event with zero surviving regions in this video is not fatal to the
    whole video (matching `prioritize_videos()`'s zero-fill semantics for an
    uncovered event): it contributes a fixed 0.0 to the mean and is skipped
    when checking order, rather than eliminating the video from tuple
    ranking entirely. A video with zero covered events in total still yields
    no tuples, same as `prioritize_videos()` never ranking it at all.
    """

    pools: list[list[TemporalRegion | None]] = []
    for event_id in event_ids:
        pool = pool_event_regions(regions, event_id=event_id, video_id=video_id, params=params)
        pools.append(list(pool) if pool else [None])
    if all(pool == [None] for pool in pools):
        return []

    results: list[TupleRankingResult] = []
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
                TupleRankingResult(
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
    params: TupleRankingHyperparameters,
    order_constraints: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[tuple[str, float]], dict[str, list[TupleRankingResult]]]:
    """Videos ranked by pooled tuple score (raw, not yet normalized to
    [0,1] - callers writing this into a `UnitScore` field should
    `algorithms.robust_sigmoid()` the scores first, same as
    `assemble_ordered_tuples()` already does for its own raw scores), plus
    each video's kept tuples (so a caller can also read off the winning
    tuple's per-event anchors for e.g. seeding boundary refinement)."""

    video_ids = sorted({region.video_id for region in regions})
    video_scores: list[tuple[str, float]] = []
    tuples_by_video: dict[str, list[TupleRankingResult]] = {}
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
