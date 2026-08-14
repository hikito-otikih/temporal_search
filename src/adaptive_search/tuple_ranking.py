"""Multi-region pooling + order-aware tuple video ranking.

An opt-in alternative to `algorithms.py::prioritize_videos()`'s independent
per-event max: instead of picking each event's best-scoring region in
isolation, pool up to `max_regions_per_event` candidate regions per event,
assemble same-video combinations across events (bounded backtracking - see
`assemble_region_tuples_for_video`), and score each combination by its mean
region score plus a confidence-gated bonus/penalty for whether its per-event
timestamps land in the expected order.

Ported from `irrelevant_things/benchmarks/youcook2/region_tuple_ranking.py`,
validated there against real YouCook2 queries (n=30/n=60,
`region_tuple_ranking_results.md`): +15% final_query_score over
`prioritize_videos()` at this module's default hyperparameters, with two
correctness fixes carried over from that report: order checking uses
explicit (predecessor_index, successor_index) constraints, not adjacent
query-list position (list position is only the *default* constraint set,
`order_constraints=None`, when no relation data is available for a video);
and those constraints are built from `EventDefinition.temporal_relation`/
`.reference_event_id` with transitive closure over the full relation graph,
not just direct single-hop edges (`build_order_constraints`) - see that
report's Methodology section for why both were necessary, not just
theoretically nicer.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from .algorithms import _candidate_normalized_scores_by_event
from .schemas import EventDefinition, SparseCandidate, TemporalRegion, TupleRankingHyperparameters


def atomic_regions(candidates: Sequence[SparseCandidate]) -> list[TemporalRegion]:
    """One `TemporalRegion` per candidate - no clustering (Stage 3) at all.

    Uses the exact same population-relative `robust_sigmoid` normalization
    `cluster_temporal_regions` applies, via the same shared helper, so a
    region's `normalized_coarse_score` here is directly comparable to (not
    recalibrated relative to) a clustered region's. Production default for
    the tuple-ranking path as of the n=60 comparison in
    `region_tuple_ranking_results.md`: atomic regions + `temporal_relation` +
    `confidence_gate="threshold"@0.5` was the best `final_query_score` found
    across all rounds, with a per-video regression profile (44/60 identical,
    15/60 off by a tie-break-level rank shift with hits held exactly
    constant, 1/60 better) judged acceptable relative to the aggregate gain.
    """

    normalized_by_id = _candidate_normalized_scores_by_event(candidates)
    return [
        TemporalRegion(
            id=f"atomic_{candidate.id}",
            session_id=candidate.session_id,
            event_id=candidate.event_id,
            video_id=candidate.video_id,
            start_seconds=candidate.timestamp_seconds,
            end_seconds=candidate.timestamp_seconds,
            candidate_ids=(candidate.id,),
            raw_coarse_score=candidate.raw_relevance_score,
            normalized_coarse_score=normalized_by_id[candidate.id],
        )
        for candidate in candidates
    ]


@dataclass(frozen=True)
class TupleRankingResult:
    video_id: str
    score: float
    region_mean_score: float
    margin_score: float
    order_score: float
    region_ids: tuple[str | None, ...]
    timestamps: tuple[float | None, ...]


def _effective_order_weight(
    params: TupleRankingHyperparameters, confidence_value: float
) -> float:
    """`confidence_value` is whichever signal `params.confidence_gate`
    selects - the tuple's mean region score for `"linear"`/`"threshold"`,
    or its mean per-event margin for `"margin"` (see `_region_margin`); the
    gating *shape* (none/linear-scale/hard-cutoff) is the same regardless of
    which signal feeds it, so callers just pick the right float to pass."""

    if params.confidence_gate == "none":
        return params.order_weight
    if params.confidence_gate == "linear":
        return params.order_weight * confidence_value
    return params.order_weight if confidence_value >= params.confidence_gate_threshold else 0.0


def _region_margin(region: TemporalRegion, pool: Sequence[TemporalRegion]) -> float:
    """`score(region) - score(pool's best OTHER member)` - how much better
    this *specific* choice is than the pool's best alternative, not just how
    good the event's top option was in the abstract.

    If `region` is already the pool's top scorer, this is the classic
    top-vs-runner-up gap. If a tuple deliberately reaches past the top
    region (to satisfy the order term), this goes *negative* - correctly
    signaling lower confidence the further down the pool the choice reaches,
    which a pool-level (not choice-level) margin could not express. A pool
    with only one member has no alternative to compare against; the
    region's own score is used (nothing to be less confident than)."""

    scores = [member.normalized_coarse_score or 0.0 for member in pool]
    if len(scores) < 2:
        return scores[0] if scores else 0.0
    own = region.normalized_coarse_score or 0.0
    best = scores[0]
    if region is pool[0]:
        return best - scores[1]
    return own - best


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
            margins = [
                _region_margin(region, pools[i]) if region is not None else 0.0
                for i, region in enumerate(selected)
            ]
            margin_mean = fmean(margins)
            order = _order_score(timestamps, order_constraints)
            confidence_value = margin_mean if params.confidence_gate == "margin" else region_mean
            final = region_mean + _effective_order_weight(params, confidence_value) * order
            results.append(
                TupleRankingResult(
                    video_id=video_id,
                    score=final,
                    region_mean_score=region_mean,
                    margin_score=margin_mean,
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


def build_order_constraints(
    events: Sequence[EventDefinition],
) -> list[tuple[int, int]] | None:
    """Directed (predecessor_index, successor_index) pairs, transitively
    closed, derived from each event's real `temporal_relation` +
    `reference_event_id` - the direction comes from the relation the rewrite
    stage actually classified, not from an event's position in the query.

    - `"after"` (event $i$, reference $r$): direct edge $r \\to i$ ($r$
      precedes $i$).
    - `"before"` (event $i$, reference $r$): direct edge $i \\to r$.
    - `"sequence_start"` / `"independent"` / `"unknown"`: no edge from this
      event (no reference to derive one from).
    - `"during"` / `"simultaneous"`: **deliberately** no edge - not a gap.
      `_order_score` can only express strict pairwise precedence
      ("predecessor's timestamp must be smaller"); "during" and
      "simultaneous" describe temporal overlap/proximity, not precedence.
      Asserting either direction here would claim something the relation
      does not say, and would be wrong exactly as often as the two events'
      true positions resolve the other way - no constraint is the correct
      encoding of these two relations under this pairwise-precedence model,
      not a missing case. A proximity- or range-aware scoring extension
      would be the right place to use these relations, not this function.

    Direct edges are closed transitively (if $A$ precedes $B$ and $B$
    precedes $C$, $A$ is also constrained to precede $C$) via reachability
    over the (small, per-query) event graph - a direct-edges-only constraint
    set would otherwise miss ordering the relations jointly imply between
    non-adjacent events.

    Returns `None` (defer to `_order_score`'s own adjacent-chain default)
    only when no event carries any relation data at all, e.g.\\ these
    `EventDefinition`s never went through `rewrite_bridge.py`.
    """

    index_by_id = {event.event_id: index for index, event in enumerate(events)}
    if not any(
        event.temporal_relation != "unknown" or event.reference_event_id is not None
        for event in events
    ):
        return None

    direct_edges: set[tuple[int, int]] = set()
    for index, event in enumerate(events):
        reference = event.reference_event_id
        if reference is None or reference not in index_by_id:
            continue
        reference_index = index_by_id[reference]
        if event.temporal_relation == "after":
            direct_edges.add((reference_index, index))
        elif event.temporal_relation == "before":
            direct_edges.add((index, reference_index))
        # "during" / "simultaneous": deliberately no edge - see docstring.

    count = len(events)
    reachable = [[False] * count for _ in range(count)]
    for predecessor, successor in direct_edges:
        reachable[predecessor][successor] = True
    for via in range(count):
        for i in range(count):
            if not reachable[i][via]:
                continue
            for j in range(count):
                if reachable[via][j]:
                    reachable[i][j] = True

    return [
        (i, j)
        for i in range(count)
        for j in range(count)
        if reachable[i][j]
    ]
