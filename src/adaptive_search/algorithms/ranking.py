"""Independent per-event video ranking (`prioritize_videos` - no longer
called from `GET /video-priorities`, which uses `tuple_ranking.py`
exclusively, but still real, tested, and the standing comparison baseline
every tuple-ranking benchmark round measures against) and refinement
frontier selection."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Iterable, Sequence

from ..schemas import RefinementHyperparameters, SearchConstraints, TemporalRegion, VideoPriority
from .constraints import _region_allowed
from .scoring import _region_score_map


def _min_pairwise_gap_seconds(timestamps: Sequence[float]) -> float | None:
    """The smallest gap between any two of `timestamps`. Always occurs
    between two *adjacent* values in sorted order (a non-adjacent pair can
    never be closer than the adjacent pairs between them), so sorting once
    and diffing neighbors is enough - no need to check every pair."""

    if len(timestamps) < 2:
        return None
    ordered = sorted(timestamps)
    return min(b - a for a, b in zip(ordered, ordered[1:]))


def distinctness_from_timestamps(
    covered_timestamps: Sequence[float], norm_seconds: float
) -> float:
    """1.0 if fewer than 2 covered timestamps (nothing to compare against -
    not a penalty for coverage, that's a separate term); otherwise the
    smallest pairwise gap among them, normalized to [0,1] against
    `norm_seconds`. Shared by `prioritize_videos()` (below) and
    `router.py`'s tuple-ranking response construction, so both branches'
    `distinctness` field means the same thing computed the same way."""

    min_gap = _min_pairwise_gap_seconds(covered_timestamps)
    return 1.0 if min_gap is None else min(1.0, min_gap / norm_seconds)


def prioritize_videos(
    regions: Sequence[TemporalRegion],
    event_ids: Sequence[str],
    parameters: RefinementHyperparameters | None = None,
    constraints: SearchConstraints | None = None,
) -> list[VideoPriority]:
    """Rank videos by event coverage, mean evidence, weakest event, and
    distinctness (whether covered events' chosen timestamps are actually
    spread out, not collapsed onto the same physical moment)."""

    parameters = parameters or RefinementHyperparameters()
    constraints = constraints or SearchConstraints()
    ordered_event_ids = tuple(dict.fromkeys(event_ids))
    if not ordered_event_ids:
        return []

    active = [region for region in regions if _region_allowed(region, constraints)]
    if not active:
        return []
    scores = _region_score_map(active)
    # (score, representative_timestamp) per (video, event) - the timestamp is
    # the region's own span midpoint, which recovers the source candidate's
    # exact timestamp for atomic (single-candidate, symmetrically padded)
    # regions; ties keep the last-seen region, matching this function's
    # pre-existing max()-based tie behavior (which never had to pick a
    # specific region for a tie, only its score).
    by_video: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for region in active:
        if region.event_id not in ordered_event_ids:
            continue
        region_score = scores[region.id]
        current_score, _ = by_video[region.video_id].get(region.event_id, (0.0, 0.0))
        if region_score >= current_score:
            timestamp = (region.start_seconds + region.end_seconds) / 2.0
            by_video[region.video_id][region.event_id] = (region_score, timestamp)

    weight_sum = (
        parameters.video_coverage_weight
        + parameters.video_mean_weight
        + parameters.video_min_weight
        + parameters.video_distinctness_weight
    )
    priorities: list[VideoPriority] = []
    for video_id in sorted(by_video):
        best_by_event = by_video[video_id]
        score_vector = [
            best_by_event.get(event_id, (0.0, 0.0))[0] for event_id in ordered_event_ids
        ]
        coverage = sum(event_id in best_by_event for event_id in ordered_event_ids)
        normalized_coverage = coverage / len(ordered_event_ids)
        mean_score = fmean(score_vector)
        minimum_score = min(score_vector)
        covered_timestamps = [timestamp for _, timestamp in best_by_event.values()]
        distinctness = distinctness_from_timestamps(
            covered_timestamps, parameters.distinctness_norm_seconds
        )
        priority_score = (
            parameters.video_coverage_weight * normalized_coverage
            + parameters.video_mean_weight * mean_score
            + parameters.video_min_weight * minimum_score
            + parameters.video_distinctness_weight * distinctness
        ) / weight_sum
        priorities.append(
            VideoPriority(
                video_id=video_id,
                event_coverage=coverage,
                normalized_coverage=normalized_coverage,
                mean_best_event_score=mean_score,
                min_best_event_score=minimum_score,
                distinctness=distinctness,
                priority_score=priority_score,
            )
        )
    return sorted(
        priorities,
        key=lambda item: (
            -item.priority_score,
            -item.event_coverage,
            item.video_id,
        ),
    )


def select_refinement_frontier(
    regions: Sequence[TemporalRegion],
    video_priorities: Sequence[VideoPriority],
    event_ids: Sequence[str],
    parameters: RefinementHyperparameters | None = None,
    constraints: SearchConstraints | None = None,
    *,
    forced_region_ids: Iterable[str] = (),
) -> list[TemporalRegion]:
    """Select a deterministic, budgeted mix of priority and exploration regions."""

    parameters = parameters or RefinementHyperparameters()
    constraints = constraints or SearchConstraints()
    ordered_events = tuple(dict.fromkeys(event_ids))
    event_rank = {event_id: rank for rank, event_id in enumerate(ordered_events)}
    active = [region for region in regions if _region_allowed(region, constraints)]
    if not active:
        return []
    scores = _region_score_map(active)
    priority_rank = {
        item.video_id: rank for rank, item in enumerate(video_priorities)
    }

    forced_ids = set(forced_region_ids)
    forced_ids.update(
        event_constraint.fixed_region_id
        for event_constraint in constraints.event_constraints.values()
        if event_constraint.fixed_region_id is not None
    )
    forced = sorted(
        [region for region in active if region.id in forced_ids],
        key=lambda item: (
            event_rank.get(item.event_id, len(event_rank)),
            item.video_id,
            item.start_seconds,
            item.id,
        ),
    )
    selected: list[TemporalRegion] = list(forced)
    selected_ids = {region.id for region in selected}

    top_video_ids = {
        item.video_id
        for item in video_priorities[: parameters.max_initial_videos]
    }
    grouped: dict[tuple[str, str], list[TemporalRegion]] = defaultdict(list)
    for region in active:
        if region.video_id in top_video_ids and region.event_id in event_rank:
            grouped[(region.video_id, region.event_id)].append(region)

    # Guarantee independent evidence for each event before filling by video rank.
    independent: list[TemporalRegion] = []
    for event_id in ordered_events:
        matches = [region for region in active if region.event_id == event_id]
        if matches:
            independent.append(
                min(
                    matches,
                    key=lambda item: (
                        -scores[item.id],
                        priority_rank.get(item.video_id, len(priority_rank)),
                        item.start_seconds,
                        item.id,
                    ),
                )
            )

    core: list[TemporalRegion] = independent
    for group_key in sorted(
        grouped,
        key=lambda key: (
            priority_rank.get(key[0], len(priority_rank)),
            event_rank.get(key[1], len(event_rank)),
            key,
        ),
    ):
        ranked = sorted(
            grouped[group_key],
            key=lambda item: (-scores[item.id], item.start_seconds, item.id),
        )
        core.extend(ranked[: parameters.max_regions_per_event_per_video])
    core = list(dict.fromkeys(region.id for region in core))
    region_by_id = {region.id: region for region in active}
    core_regions = [region_by_id[region_id] for region_id in core]

    total_budget = parameters.max_total_regions
    if len(selected) >= total_budget:
        # Explicit user selections take precedence over an automatic budget.
        return selected
    remaining_budget = total_budget - len(selected)
    exploration_slots = min(
        remaining_budget,
        int(total_budget * parameters.exploration_region_ratio),
    )
    core_slots = remaining_budget - exploration_slots
    for region in core_regions:
        if region.id in selected_ids:
            continue
        if core_slots <= 0:
            break
        selected.append(region)
        selected_ids.add(region.id)
        core_slots -= 1

    # Stable round-robin order by event/video/time makes exploration replayable.
    exploration = sorted(
        [region for region in active if region.id not in selected_ids],
        key=lambda item: (
            event_rank.get(item.event_id, len(event_rank)),
            item.video_id,
            item.start_seconds,
            item.id,
        ),
    )
    for region in exploration[:exploration_slots]:
        selected.append(region)
        selected_ids.add(region.id)

    # If rounding or duplicate core regions left room, fill with strongest evidence.
    if len(selected) < total_budget:
        remainder = sorted(
            [region for region in active if region.id not in selected_ids],
            key=lambda item: (
                -scores[item.id],
                priority_rank.get(item.video_id, len(priority_rank)),
                event_rank.get(item.event_id, len(event_rank)),
                item.start_seconds,
                item.id,
            ),
        )
        for region in remainder[: total_budget - len(selected)]:
            selected.append(region)
            selected_ids.add(region.id)
    return selected
