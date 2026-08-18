"""Multi-region pooling + order-aware tuple video ranking.

An opt-in alternative to `algorithms.py::prioritize_videos()`'s independent
per-event max: instead of picking each event's best-scoring region in
isolation, pool up to `max_regions_per_event` candidate regions per event,
assemble same-video combinations across events (bounded backtracking - see
`assemble_region_tuples_for_video`), and score each combination by its mean
region score plus a confidence-gated bonus/penalty for whether its per-event
timestamps land in the expected order.

Ported from `research_tools/benchmarks/youcook2/region_tuple_ranking.py`,
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

import heapq
from dataclasses import dataclass
from itertools import count
from statistics import fmean
from typing import Mapping, Sequence

from .algorithms import _region_allowed, adjacent_hinge_penalty
from .schemas import EventDefinition, SearchConstraints, SparseCandidate, TemporalRegion, TupleRankingHyperparameters


@dataclass(frozen=True)
class TupleRankingResult:
    video_id: str
    score: float
    region_mean_score: float
    margin_score: float
    order_score: float
    region_ids: tuple[str | None, ...]
    timestamps: tuple[float | None, ...]
    # Per-event score of the region this tuple actually chose, zero-filled
    # for an uncovered event (same convention region_mean_score already
    # uses) - lets a caller compute e.g. min_best_event_score for *this*
    # combination, not the independent per-event argmax prioritize_videos()
    # would use.
    region_scores: tuple[float, ...]


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


def _adjacent_gap_penalty_or_rejection(
    event_ids: Sequence[str],
    timestamps: Sequence[float | None],
    constraints: SearchConstraints | None,
) -> tuple[bool, float]:
    """`(hard_reject, mean_soft_penalty)` for `constraints.adjacent_gap_constraints`,
    checked over pairs positionally adjacent in `event_ids` - what
    "adjacent" means for this constraint type (a constraint's
    `before_event_id`/`after_event_id` only takes effect when those two
    events also happen to be list-adjacent), the same convention the
    retired proposal-based `assemble_ordered_tuples` mechanism used before
    it was removed. A pair with either side uncovered (`None` timestamp)
    imposes no gap check, matching `_order_score`'s existing "an uncovered
    event drops any constraint touching it" convention. `min_gap_seconds`/
    `max_gap_seconds` are hard bounds (violating either rejects the whole
    combination); `gap_lambda` (when set) additionally applies
    `adjacent_hinge_penalty` as a soft, mean-pooled penalty subtracted from
    the combination's score - never negative by the time it reaches that
    call, since any gap small enough to be negative is already caught by
    `min_gap_seconds >= 0`."""

    if constraints is None or not constraints.adjacent_gap_constraints:
        return False, 0.0
    gap_map = {
        (c.before_event_id, c.after_event_id): c
        for c in constraints.adjacent_gap_constraints
    }
    penalties: list[float] = []
    for i in range(len(event_ids) - 1):
        before, after = timestamps[i], timestamps[i + 1]
        if before is None or after is None:
            continue
        constraint = gap_map.get((event_ids[i], event_ids[i + 1]))
        if constraint is None:
            continue
        gap = after - before
        if gap < constraint.min_gap_seconds:
            return True, 0.0
        if constraint.max_gap_seconds is not None and gap > constraint.max_gap_seconds:
            return True, 0.0
        if constraint.gap_lambda is not None:
            tau = constraint.hinge_tau_seconds if constraint.hinge_tau_seconds is not None else 0.0
            penalties.append(adjacent_hinge_penalty(gap, tau_seconds=tau, gap_lambda=constraint.gap_lambda))
    return False, (fmean(penalties) if penalties else 0.0)


def _tuple_span_violation(
    timestamps: Sequence[float | None],
    constraints: SearchConstraints | None,
) -> bool:
    """True if the combination's covered timestamps span more than
    `constraints.max_tuple_span_seconds`. Uses min/max over covered
    (non-`None`) timestamps rather than the retired `assemble_ordered_tuples`
    mechanism's first-minus-last convention, since that convention assumed
    every event was covered - not true here, where an uncovered event
    zero-fills instead of eliminating the video (see module-level zero-fill
    note)."""

    if constraints is None or constraints.max_tuple_span_seconds is None:
        return False
    covered = [t for t in timestamps if t is not None]
    if len(covered) < 2:
        return False
    return max(covered) - min(covered) > constraints.max_tuple_span_seconds


def assemble_region_tuples_for_video(
    video_id: str,
    event_ids: Sequence[str],
    regions: Sequence[TemporalRegion],
    candidates_by_id: Mapping[str, SparseCandidate],
    params: TupleRankingHyperparameters,
    order_constraints: Sequence[tuple[int, int]] | None = None,
    constraints: SearchConstraints | None = None,
) -> list[TupleRankingResult]:
    """All (bounded) same-video region combinations, one region per event,
    scored by mean region score plus a confidence-gated order-agreement
    bonus, minus any soft adjacent-gap penalty (`constraints`).

    Explored best-first via branch-and-bound over an admissible upper bound
    on each partial assignment's best possible completion (own achieved
    score, plus every remaining event's own best-pool score, plus the
    maximum possible order-agreement bonus - ignoring any eventual gap
    penalty, which can only lower a completion's true score and so never
    invalidates the bound), not by lexicographic enumeration. This
    guarantees the up-to-`max_tuples_per_video` results returned are the
    true best found, not an artifact of enumeration order:
    `max_combinations_per_video` bounds total search-node expansions
    (floored at `len(event_ids) + 1`, the minimum possible to ever produce
    even one complete result - see the floor's own comment below), and if
    that cap is hit before finding enough valid (non-rejected) complete
    combinations, everything already found is still provably at least as
    good as anything left unexplored - never a low-scoring combination
    standing in for a much better one that entered the search space too
    late to be reached, which naive lexicographic truncation could not
    guarantee. Ties in priority are broken in favor of depth (see the
    heap-entry comment below), so a heavily-tied pool still reliably yields
    results under a modest budget instead of exhausting it on width.

    A combination whose covered timestamps violate `constraints.
    max_tuple_span_seconds` or a hard `adjacent_gap_constraints` bound is
    rejected outright (not scored, not counted toward
    `max_tuples_per_video`) but does not stop the search - only that one
    leaf is skipped.

    An event with zero surviving regions in this video is not fatal to the
    whole video (matching `prioritize_videos()`'s zero-fill semantics for an
    uncovered event): it contributes a fixed 0.0 to the mean and is skipped
    when checking order or gap constraints, rather than eliminating the
    video from tuple ranking entirely. A video with zero covered events in
    total still yields no tuples, same as `prioritize_videos()` never
    ranking it at all.
    """

    pools: list[list[TemporalRegion | None]] = []
    for event_id in event_ids:
        pool = pool_event_regions(regions, event_id=event_id, video_id=video_id, params=params)
        pools.append(list(pool) if pool else [None])
    if all(pool == [None] for pool in pools):
        return []

    n = len(event_ids)
    pool_best_score = [
        (pools[i][0].normalized_coarse_score or 0.0) if pools[i][0] is not None else 0.0
        for i in range(n)
    ]
    remaining_best_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        remaining_best_sum[i] = remaining_best_sum[i + 1] + pool_best_score[i]
    max_order_bonus = params.order_weight

    def upper_bound(achieved_sum: float, next_index: int) -> float:
        return (achieved_sum + remaining_best_sum[next_index]) / n + max_order_bonus

    def finalize(selected: tuple[TemporalRegion | None, ...]) -> TupleRankingResult | None:
        timestamps = tuple(
            _region_timestamp(region, candidates_by_id) if region is not None else None
            for region in selected
        )
        if _tuple_span_violation(timestamps, constraints):
            return None
        rejected, gap_penalty = _adjacent_gap_penalty_or_rejection(event_ids, timestamps, constraints)
        if rejected:
            return None
        scores = [region.normalized_coarse_score or 0.0 if region is not None else 0.0 for region in selected]
        region_mean = fmean(scores)
        margins = [
            _region_margin(region, pools[i]) if region is not None else 0.0
            for i, region in enumerate(selected)
        ]
        margin_mean = fmean(margins)
        order = _order_score(timestamps, order_constraints)
        confidence_value = margin_mean if params.confidence_gate == "margin" else region_mean
        final = region_mean + _effective_order_weight(params, confidence_value) * order - gap_penalty
        return TupleRankingResult(
            video_id=video_id,
            score=final,
            region_mean_score=region_mean,
            margin_score=margin_mean,
            order_score=order,
            region_ids=tuple(region.id if region is not None else None for region in selected),
            timestamps=timestamps,
            region_scores=tuple(scores),
        )

    # Heap entries: (-priority, -depth, tie_breaker, is_partial, payload).
    #
    # A *partial* entry's payload is (achieved_sum, selected-so-far) and its
    # priority is the still-optimistic upper_bound(). A *complete* entry's
    # payload is its already-finalized TupleRankingResult and its priority
    # is that result's own exact score - critically NOT upper_bound() again,
    # which would keep adding the optimistic max_order_bonus even though
    # the order term (and any gap penalty) is now fully known, no longer
    # uncertain. Without this distinction every completed leaf would look
    # equally attractive from the order-bonus's perspective regardless of
    # whether it actually satisfied the order constraint, defeating the
    # point of best-first search for exactly the order-term-driven case
    # this rewrite exists to get right - the search would still expand in a
    # roughly region-mean-only order, only discovering a genuinely
    # order-satisfying but lower-region-mean combination after separately
    # exhausting every higher-region-mean, order-violating alternative
    # first. Finalizing eagerly at push time (not at pop time) is what
    # makes this possible: a gap/span-violating combination is pruned right
    # here and never even enters the heap.
    #
    # `-depth` breaks ties among equal-priority entries in favor of the
    # deeper (closer to complete) one - a pure tie-break, since it only
    # applies once -priority already matches, so it cannot change which
    # result best-first finds first when priorities differ. Without it, a
    # heavily-tied pool (many candidates at the same score) degenerates
    # into near-breadth-first expansion under heapq's FIFO tie-break: every
    # sibling at a shallow level pops before any of them reaches depth n,
    # so max_combinations_per_video can be exhausted expanding width with
    # zero completed results even though hundreds of valid combinations
    # exist. Preferring depth means the search commits to finishing one
    # path before fanning out into the next, so a modest budget still
    # reliably produces results under heavy ties. A complete entry's
    # payload has no directly-stored depth, but it is always exactly `n`
    # (finalize() only ever runs on a fully-selected tuple), which also
    # correctly ranks a tied complete result above a same-priority partial
    # one - a real find over a still-uncertain estimate.
    tie_breaker = count()
    heap: list[tuple[float, int, int, bool, object]] = []

    def push(achieved_sum: float, index: int, selected: tuple[TemporalRegion | None, ...]) -> None:
        if index < n:
            heapq.heappush(
                heap,
                (-upper_bound(achieved_sum, index), -index, next(tie_breaker), True, (achieved_sum, selected)),
            )
            return
        candidate = finalize(selected)
        if candidate is None:
            return  # gap/span violation - pruned, never enters the heap
        heapq.heappush(heap, (-candidate.score, -n, next(tie_breaker), False, candidate))

    push(0.0, 0, ())

    # A floor beneath the configured cap: reaching even one complete result
    # requires walking root-to-leaf (n pops, one per event) and then popping
    # that finalized leaf itself (+1) - n+1 pops minimum, regardless of how
    # good the search's ordering is. A caller-supplied cap smaller than that
    # would make it structurally impossible to ever return a single result
    # even when a valid combination plainly exists (e.g. cap=1 with any
    # events at all), which is not "safely bounded," just silently empty.
    # The configured cap otherwise still applies unchanged - this only
    # raises a cap that was too small to ever succeed at all.
    effective_max_combinations = max(params.max_combinations_per_video, n + 1)

    results: list[TupleRankingResult] = []
    expansions = 0
    while heap:
        if len(results) >= params.max_tuples_per_video:
            break
        if expansions >= effective_max_combinations:
            break
        expansions += 1
        _, _, _, is_partial, payload = heapq.heappop(heap)
        if not is_partial:
            results.append(payload)  # already a finalized TupleRankingResult
            continue
        achieved_sum, selected = payload
        index = len(selected)
        for region in pools[index]:
            region_score = (region.normalized_coarse_score or 0.0) if region is not None else 0.0
            push(achieved_sum + region_score, index + 1, selected + (region,))

    results.sort(key=lambda item: (-item.score, item.region_ids))
    return results[: params.max_tuples_per_video]


def rank_videos_by_region_tuples(
    regions: Sequence[TemporalRegion],
    candidates_by_id: Mapping[str, SparseCandidate],
    event_ids: Sequence[str],
    params: TupleRankingHyperparameters,
    order_constraints: Sequence[tuple[int, int]] | None = None,
    constraints: SearchConstraints | None = None,
) -> tuple[list[tuple[str, float]], dict[str, list[TupleRankingResult]]]:
    """Videos ranked by pooled tuple score (raw, not yet normalized to
    [0,1] - callers writing this into a `UnitScore` field should
    `algorithms.robust_sigmoid()` the scores first), plus
    each video's kept tuples (so a caller can also read off the winning
    tuple's per-event anchors for e.g. seeding boundary refinement).

    `constraints` (rejected/fixed videos and regions from Stage 6's
    interactive correction) is applied the same way `prioritize_videos()`
    applies it - via `_region_allowed`, filtering `regions` before pooling -
    so a region a user explicitly rejected or a different, unpinned region
    can never win a tuple. Before this, tuple ranking read the raw,
    unfiltered region set directly; a whole rejected *video* happened to
    still disappear from the final API response (the router's merge step
    only keeps videos also present in the constraint-respecting baseline
    list), but a rejected *region* or an unmet `fixed_region_id` pin inside
    an otherwise-valid video was not honored - confirmed by grep, not
    assumption: this module had no reference to `SearchConstraints` or
    `_region_allowed` at all until this fix.

    An `EventConstraint` whose `fixed_video_id` + `fixed_timestamp_seconds`
    name an exact frame with no corresponding region (`commands/fix-frame`
    on an arbitrary timestamp, not a pinned existing region) is synthesized
    into a real region before filtering (`_synthetic_fixed_regions`) -
    without this, a user-confirmed frame had no region for pooling to ever
    select, so it was silently ignored by ranking regardless of
    `fixed_video_id`/`fixed_timestamp_seconds` being set. `adjacent_gap_
    constraints` and `max_tuple_span_seconds` are enforced per combination
    inside `assemble_region_tuples_for_video` itself - the retired
    proposal-based `assemble_ordered_tuples` mechanism enforced the same two
    constraint types for its own (separate) dense per-frame tuple assembly,
    but its output had no HTTP reader and nothing internal ever consulted
    it beyond a bare count, so it was removed rather than kept as a second,
    parallel implementation of the same constraints."""

    if constraints is not None:
        regions = [*regions, *_synthetic_fixed_regions(regions, constraints)]
        regions = [region for region in regions if _region_allowed(region, constraints)]
    video_ids = sorted({region.video_id for region in regions})
    video_scores: list[tuple[str, float]] = []
    tuples_by_video: dict[str, list[TupleRankingResult]] = {}
    for video_id in video_ids:
        tuples = assemble_region_tuples_for_video(
            video_id, event_ids, regions, candidates_by_id, params, order_constraints, constraints
        )
        if not tuples:
            continue
        tuples_by_video[video_id] = tuples
        score = tuples[0].score if params.pooling == "max" else fmean(t.score for t in tuples)
        video_scores.append((video_id, score))
    video_scores.sort(key=lambda item: (-item[1], item[0]))
    return video_scores, tuples_by_video


def _synthetic_fixed_regions(
    regions: Sequence[TemporalRegion],
    constraints: SearchConstraints,
) -> list[TemporalRegion]:
    """One zero-width `TemporalRegion` per event whose constraint pins an
    exact timestamp (`fixed_video_id` + `fixed_timestamp_seconds`, set via
    `commands/fix-frame`) that doesn't already correspond to a real region.

    `fixed_region_id` is deliberately ignored whenever `fixed_frame_id` is
    also set - "explicit frame selection is strongest, deliberately ignores
    a stale fixed-region field" is the same precedence the retired
    proposal-based mechanism's own `_proposal_allowed` used. This matters in
    practice, not just for consistency: `service.py::fix_frame()` always
    populates `fixed_region_id` too, defaulting it to a synthetic
    `f"user:{event_id}:{video_id}"` placeholder that names no real
    `TemporalRegion` at all - treating that as a genuine region pin would
    silently skip synthesis on every real `commands/fix-frame` call, the
    exact call this function exists to support. `fixed_region_id` is only
    respected as a real pin (skip synthesis, defer entirely to
    `_region_allowed`'s own handling) when `fixed_frame_id` is `None` - a
    constraint that named an existing region without ever naming a frame.

    Scored at the top of `[0,1]` (`normalized_coarse_score=1.0`) since the
    user has already confirmed this exact frame is correct - it should win
    pooling on merit, not only by being force-included, and
    `_region_allowed`'s existing `fixed_video_id` handling (scoped to this
    region's own event) already rejects every other video's candidates for
    this event, so a real competing region from a different video cannot
    accidentally outrank it. `candidate_ids` references a synthetic,
    deliberately unresolvable ID rather than a real candidate -
    `_region_timestamp` falls back to the region's own span midpoint (this
    exact timestamp) whenever none of a region's `candidate_ids` resolve,
    so the fixed timestamp still comes through correctly without needing a
    real backing candidate to exist."""

    if not constraints.event_constraints:
        return []
    session_id = regions[0].session_id if regions else "fixed"
    synthetic: list[TemporalRegion] = []
    for event_id, event_constraint in constraints.event_constraints.items():
        if event_constraint.fixed_video_id is None or event_constraint.fixed_timestamp_seconds is None:
            continue
        if event_constraint.fixed_frame_id is None and event_constraint.fixed_region_id is not None:
            continue  # a genuine "pin this existing region" constraint - _region_allowed handles it
        timestamp = event_constraint.fixed_timestamp_seconds
        already_covered = any(
            region.event_id == event_id
            and region.video_id == event_constraint.fixed_video_id
            and region.start_seconds <= timestamp <= region.end_seconds
            for region in regions
        )
        if already_covered:
            continue
        placeholder_id = f"fixed:{event_id}:{event_constraint.fixed_video_id}"
        synthetic.append(
            TemporalRegion(
                id=placeholder_id,
                session_id=session_id,
                event_id=event_id,
                video_id=event_constraint.fixed_video_id,
                start_seconds=timestamp,
                end_seconds=timestamp,
                candidate_ids=(placeholder_id,),
                raw_coarse_score=1.0,
                normalized_coarse_score=1.0,
            )
        )
    return synthetic


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
