"""Deterministic, model-free algorithms for adaptive temporal search.

Embedding and video decoding intentionally live outside this module.  Every
function here consumes validated domain objects, has deterministic tie breaks,
and can be tested without a GPU, a model download, or network access.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from hashlib import sha256
from math import exp, isclose
from statistics import fmean, median

from .schemas import (
    AdjacentGapConstraint,
    BoundaryHyperparameters,
    ClusteringHyperparameters,
    EventConstraint,
    EventDefinition,
    EventProposal,
    FrameScoreSample,
    RefinementHyperparameters,
    SearchConstraints,
    SparseCandidate,
    TemporalRegion,
    TupleResult,
    VideoPriority,
)


_EPSILON = 1e-12


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{sha256(payload).hexdigest()[:20]}"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    forward = exp(value)
    return forward / (1.0 + forward)


def robust_sigmoid(
    values: Sequence[float],
    *,
    clip_z: float = 8.0,
) -> list[float]:
    """Map arbitrary finite scores to ``[0, 1]`` using median/MAD scaling.

    When MAD is zero, values at the median remain neutral (0.5), while values
    on either side get the corresponding clipped tail probability.  This is
    preferable to allowing a single outlier to define the scale.
    """

    if clip_z <= 0.0:
        raise ValueError("clip_z must be positive")
    if not values:
        return []

    center = float(median(values))
    mad = float(median(abs(value - center) for value in values))
    scale = 1.4826 * mad
    if scale <= _EPSILON:
        if all(isclose(value, center, abs_tol=_EPSILON) for value in values):
            return [0.5] * len(values)
        return [
            0.5
            if isclose(value, center, abs_tol=_EPSILON)
            else _sigmoid(clip_z if value > center else -clip_z)
            for value in values
        ]

    normalized: list[float] = []
    for value in values:
        z_score = max(-clip_z, min(clip_z, (value - center) / scale))
        normalized.append(_sigmoid(z_score))
    return normalized


def pairwise_softmax(
    first_score: float,
    second_score: float,
    *,
    temperature: float = 1.0,
) -> tuple[float, float]:
    """Stable two-way softmax used to make pre/post scores comparable."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled_first = first_score / temperature
    scaled_second = second_score / temperature
    maximum = max(scaled_first, scaled_second)
    first_exp = exp(scaled_first - maximum)
    second_exp = exp(scaled_second - maximum)
    denominator = first_exp + second_exp
    return first_exp / denominator, second_exp / denominator


def calibrate_frame_scores(
    samples: Sequence[FrameScoreSample],
    parameters: BoundaryHyperparameters | None = None,
) -> list[FrameScoreSample]:
    """Calibrate raw scores independently inside each event/video/region.

    Pre and post states are normalized as a pair at each frame.  Anchor and
    motion signals use robust sigmoid calibration across the local region.
    Raw fields are copied unchanged into the returned samples.
    """

    parameters = parameters or BoundaryHyperparameters()
    calibrated = list(samples)
    grouped_indices: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped_indices[
            (
                sample.session_id,
                sample.event_id,
                sample.video_id,
                sample.region_id,
            )
        ].append(index)

    for indices in grouped_indices.values():
        anchor_values = [samples[index].raw_anchor_score for index in indices]
        motion_values = [samples[index].raw_motion_score for index in indices]
        normalized_anchors = robust_sigmoid(
            anchor_values, clip_z=parameters.anchor_clip_z
        )
        normalized_motion = robust_sigmoid(
            motion_values, clip_z=parameters.anchor_clip_z
        )
        for local_index, sample_index in enumerate(indices):
            sample = samples[sample_index]
            pre_score, post_score = pairwise_softmax(
                sample.raw_pre_score,
                sample.raw_post_score,
                temperature=parameters.pairwise_temperature,
            )
            calibrated[sample_index] = sample.model_copy(
                update={
                    "normalized_anchor_score": normalized_anchors[local_index],
                    "normalized_pre_score": pre_score,
                    "normalized_post_score": post_score,
                    "normalized_motion_score": normalized_motion[local_index],
                }
            )
    return calibrated


def _candidate_normalized_scores(
    candidates: Sequence[SparseCandidate],
) -> dict[str, float]:
    calibrated = robust_sigmoid(
        [candidate.raw_relevance_score for candidate in candidates]
    )
    return {
        candidate.id: (
            candidate.normalized_relevance_score
            if candidate.normalized_relevance_score is not None
            else calibrated[index]
        )
        for index, candidate in enumerate(candidates)
    }


def _candidate_normalized_scores_by_event(
    candidates: Sequence[SparseCandidate],
) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[SparseCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.session_id, candidate.event_id)].append(candidate)

    normalized: dict[str, float] = {}
    for group in grouped.values():
        normalized.update(_candidate_normalized_scores(group))
    return normalized

def cluster_temporal_regions(
    candidates: Sequence[SparseCandidate],
    parameters: ClusteringHyperparameters | None = None,
) -> list[TemporalRegion]:
    """Cluster candidates per event/video, expand margins, then merge overlaps."""

    parameters = parameters or ClusteringHyperparameters()
    if not candidates:
        return []

    normalized_scores = _candidate_normalized_scores_by_event(candidates)
    grouped: dict[tuple[str, str, str], list[SparseCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.session_id, candidate.event_id, candidate.video_id)].append(
            candidate
        )

    provisional: list[TemporalRegion] = []
    max_core_span = parameters.max_region_seconds - 2.0 * parameters.margin_seconds
    for group_key in sorted(grouped):
        group = sorted(
            grouped[group_key],
            key=lambda item: (item.timestamp_seconds, item.frame_id, item.id),
        )
        clusters: list[list[SparseCandidate]] = []
        current: list[SparseCandidate] = []
        for candidate in group:
            if current and (
                candidate.timestamp_seconds - current[-1].timestamp_seconds
                > parameters.gap_seconds
                or candidate.timestamp_seconds - current[0].timestamp_seconds
                > max_core_span
            ):
                clusters.append(current)
                current = []
            current.append(candidate)
        if current:
            clusters.append(current)

        for cluster in clusters:
            session_id, event_id, video_id = group_key
            candidate_ids = tuple(item.id for item in cluster)
            start_seconds = max(
                0.0, cluster[0].timestamp_seconds - parameters.margin_seconds
            )
            end_seconds = cluster[-1].timestamp_seconds + parameters.margin_seconds
            provisional.append(
                TemporalRegion(
                    id=_stable_id(
                        "region",
                        session_id,
                        event_id,
                        video_id,
                        *candidate_ids,
                    ),
                    session_id=session_id,
                    event_id=event_id,
                    video_id=video_id,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    candidate_ids=candidate_ids,
                    raw_coarse_score=max(
                        item.raw_relevance_score for item in cluster
                    ),
                    normalized_coarse_score=max(
                        normalized_scores[item.id] for item in cluster
                    ),
                )
            )

    return merge_overlapping_regions(
        provisional,
        max_region_seconds=parameters.max_region_seconds,
    )


def merge_overlapping_regions(
    regions: Sequence[TemporalRegion],
    *,
    max_region_seconds: float | None = None,
) -> list[TemporalRegion]:
    """Union overlapping regions without crossing event/video or status scopes."""
    if max_region_seconds is not None and max_region_seconds <= 0.0:
        raise ValueError("max_region_seconds must be positive")


    grouped: dict[
        tuple[str, str, str, str, str], list[TemporalRegion]
    ] = defaultdict(list)
    for region in regions:
        grouped[
            (
                region.session_id,
                region.event_id,
                region.video_id,
                region.refinement_status,
                region.user_status,
            )
        ].append(region)

    merged_regions: list[TemporalRegion] = []
    for group_key in sorted(grouped):
        ordered = sorted(
            grouped[group_key],
            key=lambda item: (item.start_seconds, item.end_seconds, item.id),
        )
        active: list[TemporalRegion] = []
        for region in ordered:
            would_exceed_limit = (
                bool(active)
                and max_region_seconds is not None
                and max(region.end_seconds, active[-1].end_seconds)
                - min(region.start_seconds, active[-1].start_seconds)
                > max_region_seconds
            )
            if (
                not active
                or region.start_seconds > active[-1].end_seconds
                or would_exceed_limit
            ):
                active.append(region)
                continue

            previous = active.pop()
            candidate_ids = tuple(
                dict.fromkeys((*previous.candidate_ids, *region.candidate_ids))
            )
            normalized_values = [
                value
                for value in (
                    previous.normalized_coarse_score,
                    region.normalized_coarse_score,
                )
                if value is not None
            ]
            active.append(
                previous.model_copy(
                    update={
                        "id": _stable_id(
                            "region",
                            previous.session_id,
                            previous.event_id,
                            previous.video_id,
                            *candidate_ids,
                        ),
                        "start_seconds": min(
                            previous.start_seconds, region.start_seconds
                        ),
                        "end_seconds": max(
                            previous.end_seconds, region.end_seconds
                        ),
                        "candidate_ids": candidate_ids,
                        "raw_coarse_score": max(
                            previous.raw_coarse_score, region.raw_coarse_score
                        ),
                        "normalized_coarse_score": (
                            max(normalized_values) if normalized_values else None
                        ),
                    }
                )
            )
        merged_regions.extend(active)

    return sorted(
        merged_regions,
        key=lambda item: (
            item.session_id,
            item.event_id,
            item.video_id,
            item.start_seconds,
            item.end_seconds,
            item.id,
        ),
    )


def _mean(samples: Sequence[FrameScoreSample], attribute: str) -> float:
    values = [getattr(sample, attribute) for sample in samples]
    if any(value is None for value in values):
        raise ValueError("frame scores must be calibrated before proposal generation")
    return fmean(float(value) for value in values)


def generate_boundary_proposals(
    samples: Sequence[FrameScoreSample],
    parameters: BoundaryHyperparameters | None = None,
    *,
    source: str = "dense",
    apply_nms: bool = True,
) -> list[EventProposal]:
    """Score finite left/right windows and emit locally calibrated proposals.

    A timestamp is skipped unless both sides contain ``min_samples_per_side``.
    Only configured window sizes are considered, preventing unconstrained
    maximization over arbitrary temporal windows.
    """

    parameters = parameters or BoundaryHyperparameters()
    calibrated = calibrate_frame_scores(samples, parameters)
    grouped: dict[tuple[str, str, str, str], list[FrameScoreSample]] = defaultdict(
        list
    )
    for sample in calibrated:
        grouped[
            (
                sample.session_id,
                sample.event_id,
                sample.video_id,
                sample.region_id,
            )
        ].append(sample)

    proposals: list[EventProposal] = []
    maximum_window = max(parameters.window_options_seconds)
    final_weight = (
        parameters.semantic_weight
        + parameters.boundary_weight
        + parameters.pre_weight
        + parameters.post_weight
    )

    for group_key in sorted(grouped):
        timeline = sorted(
            grouped[group_key],
            key=lambda item: (item.timestamp_seconds, item.frame_id),
        )
        scored_centers: list[dict[str, object]] = []
        for center in timeline:
            best: dict[str, object] | None = None
            for left_window in parameters.window_options_seconds:
                left = [
                    sample
                    for sample in timeline
                    if center.timestamp_seconds - left_window
                    <= sample.timestamp_seconds
                    < center.timestamp_seconds
                ]
                if len(left) < parameters.min_samples_per_side:
                    continue
                for right_window in parameters.window_options_seconds:
                    right = [
                        sample
                        for sample in timeline
                        if center.timestamp_seconds
                        < sample.timestamp_seconds
                        <= center.timestamp_seconds + right_window
                    ]
                    if len(right) < parameters.min_samples_per_side:
                        continue

                    pre_consistency = _mean(left, "normalized_pre_score")
                    post_persistence = _mean(right, "normalized_post_score")
                    post_contrast = post_persistence - _mean(
                        left, "normalized_post_score"
                    )
                    pre_contrast = pre_consistency - _mean(
                        right, "normalized_pre_score"
                    )
                    motion_score = float(center.normalized_motion_score)
                    raw_boundary = (
                        parameters.post_contrast_weight * post_contrast
                        + parameters.pre_contrast_weight * pre_contrast
                        + parameters.motion_contrast_weight * motion_score
                    )
                    raw_boundary -= parameters.window_length_regularization * (
                        (left_window + right_window) / (2.0 * maximum_window)
                    )
                    raw_boundary -= parameters.window_asymmetry_regularization * (
                        abs(left_window - right_window) / maximum_window
                    )
                    candidate = {
                        "center": center,
                        "left_window": left_window,
                        "right_window": right_window,
                        "raw_boundary": raw_boundary,
                        "pre_consistency": pre_consistency,
                        "post_persistence": post_persistence,
                    }
                    if best is None:
                        best = candidate
                        continue
                    candidate_key = (
                        float(candidate["raw_boundary"]),
                        -float(candidate["left_window"])
                        - float(candidate["right_window"]),
                        -abs(
                            float(candidate["left_window"])
                            - float(candidate["right_window"])
                        ),
                        -float(candidate["left_window"]),
                    )
                    best_key = (
                        float(best["raw_boundary"]),
                        -float(best["left_window"]) - float(best["right_window"]),
                        -abs(
                            float(best["left_window"])
                            - float(best["right_window"])
                        ),
                        -float(best["left_window"]),
                    )
                    if candidate_key > best_key:
                        best = candidate
            if best is not None:
                scored_centers.append(best)

        normalized_boundaries = robust_sigmoid(
            [float(item["raw_boundary"]) for item in scored_centers],
            clip_z=parameters.anchor_clip_z,
        )
        for item, normalized_boundary in zip(
            scored_centers, normalized_boundaries, strict=True
        ):
            center = item["center"]
            if not isinstance(center, FrameScoreSample):
                raise TypeError("internal boundary proposal state is invalid")
            semantic_score = float(center.normalized_anchor_score)
            pre_consistency = float(item["pre_consistency"])
            post_persistence = float(item["post_persistence"])
            final_score = (
                parameters.semantic_weight * semantic_score
                + parameters.boundary_weight * normalized_boundary
                + parameters.pre_weight * pre_consistency
                + parameters.post_weight * post_persistence
            ) / final_weight
            left_window = float(item["left_window"])
            right_window = float(item["right_window"])
            proposals.append(
                EventProposal(
                    id=_stable_id(
                        "proposal",
                        *group_key,
                        center.frame_id,
                        center.timestamp_seconds,
                        left_window,
                        right_window,
                    ),
                    session_id=center.session_id,
                    event_id=center.event_id,
                    video_id=center.video_id,
                    region_id=center.region_id,
                    timestamp_seconds=center.timestamp_seconds,
                    frame_id=center.frame_id,
                    raw_semantic_score=center.raw_anchor_score,
                    normalized_semantic_score=semantic_score,
                    raw_boundary_score=float(item["raw_boundary"]),
                    normalized_boundary_score=normalized_boundary,
                    raw_motion_score=center.raw_motion_score,
                    normalized_motion_score=float(center.normalized_motion_score),
                    pre_consistency_score=pre_consistency,
                    post_persistence_score=post_persistence,
                    final_event_score=final_score,
                    left_window_seconds=left_window,
                    right_window_seconds=right_window,
                    source=source,
                )
            )

    if apply_nms:
        return temporal_nms(
            proposals,
            radius_seconds=parameters.nms_radius_seconds,
            max_per_group=parameters.max_proposals_per_region,
        )
    return sorted(proposals, key=_proposal_output_key)


def _proposal_rank_key(proposal: EventProposal) -> tuple[object, ...]:
    user_rank = {
        "confirmed": 0,
        "fixed": 1,
        "active": 2,
        "rejected": 3,
    }[proposal.user_status]
    return (
        user_rank,
        -proposal.final_event_score,
        -proposal.normalized_boundary_score,
        proposal.timestamp_seconds,
        proposal.frame_id,
        proposal.id,
    )


def _proposal_output_key(proposal: EventProposal) -> tuple[object, ...]:
    return (
        proposal.event_id,
        proposal.video_id,
        proposal.region_id,
        -proposal.final_event_score,
        proposal.timestamp_seconds,
        proposal.frame_id,
        proposal.id,
    )


def temporal_nms(
    proposals: Sequence[EventProposal],
    *,
    radius_seconds: float,
    max_per_group: int | None = None,
) -> list[EventProposal]:
    """Temporal non-maximum suppression within each event/video/region."""

    if radius_seconds < 0.0:
        raise ValueError("radius_seconds must be non-negative")
    if max_per_group is not None and max_per_group <= 0:
        raise ValueError("max_per_group must be positive")

    grouped: dict[tuple[str, str, str, str], list[EventProposal]] = defaultdict(list)
    for proposal in proposals:
        if proposal.user_status != "rejected":
            grouped[
                (
                    proposal.session_id,
                    proposal.event_id,
                    proposal.video_id,
                    proposal.region_id,
                )
            ].append(proposal)

    kept: list[EventProposal] = []
    for group_key in sorted(grouped):
        group_kept: list[EventProposal] = []
        for proposal in sorted(grouped[group_key], key=_proposal_rank_key):
            if any(
                abs(proposal.timestamp_seconds - selected.timestamp_seconds)
                <= radius_seconds
                for selected in group_kept
            ):
                continue
            group_kept.append(proposal)
            if max_per_group is not None and len(group_kept) >= max_per_group:
                break
        kept.extend(group_kept)
    return sorted(kept, key=_proposal_output_key)


def _event_constraint(
    constraints: SearchConstraints,
    event_id: str,
) -> EventConstraint:
    return constraints.event_constraints.get(event_id, EventConstraint())


def _region_allowed(
    region: TemporalRegion,
    constraints: SearchConstraints,
) -> bool:
    if region.user_status == "rejected":
        return False
    if region.video_id in constraints.rejected_video_ids:
        return False
    event_constraint = _event_constraint(constraints, region.event_id)
    if region.id in event_constraint.rejected_region_ids:
        return False
    if event_constraint.fixed_video_id is not None:
        if region.video_id != event_constraint.fixed_video_id:
            return False
    if (
        event_constraint.fixed_frame_id is None
        and event_constraint.fixed_region_id is not None
        and region.id != event_constraint.fixed_region_id
    ):
        return False

    allowlist_overridden = (
        event_constraint.fixed_video_id == region.video_id
        or event_constraint.fixed_region_id == region.id
    )
    return (
        not constraints.allowed_video_ids
        or region.video_id in constraints.allowed_video_ids
        or allowlist_overridden
    )


def _region_score_map(regions: Sequence[TemporalRegion]) -> dict[str, float]:
    calibrated = robust_sigmoid([region.raw_coarse_score for region in regions])
    return {
        region.id: (
            region.normalized_coarse_score
            if region.normalized_coarse_score is not None
            else calibrated[index]
        )
        for index, region in enumerate(regions)
    }


def prioritize_videos(
    regions: Sequence[TemporalRegion],
    event_ids: Sequence[str],
    parameters: RefinementHyperparameters | None = None,
    constraints: SearchConstraints | None = None,
) -> list[VideoPriority]:
    """Rank videos by event coverage, mean evidence, and weakest event."""

    parameters = parameters or RefinementHyperparameters()
    constraints = constraints or SearchConstraints()
    ordered_event_ids = tuple(dict.fromkeys(event_ids))
    if not ordered_event_ids:
        return []

    active = [region for region in regions if _region_allowed(region, constraints)]
    if not active:
        return []
    scores = _region_score_map(active)
    by_video: dict[str, dict[str, float]] = defaultdict(dict)
    for region in active:
        if region.event_id not in ordered_event_ids:
            continue
        current = by_video[region.video_id].get(region.event_id, 0.0)
        by_video[region.video_id][region.event_id] = max(current, scores[region.id])

    weight_sum = (
        parameters.video_coverage_weight
        + parameters.video_mean_weight
        + parameters.video_min_weight
    )
    priorities: list[VideoPriority] = []
    for video_id in sorted(by_video):
        best_by_event = by_video[video_id]
        score_vector = [best_by_event.get(event_id, 0.0) for event_id in ordered_event_ids]
        coverage = sum(event_id in best_by_event for event_id in ordered_event_ids)
        normalized_coverage = coverage / len(ordered_event_ids)
        mean_score = fmean(score_vector)
        minimum_score = min(score_vector)
        priority_score = (
            parameters.video_coverage_weight * normalized_coverage
            + parameters.video_mean_weight * mean_score
            + parameters.video_min_weight * minimum_score
        ) / weight_sum
        priorities.append(
            VideoPriority(
                video_id=video_id,
                event_coverage=coverage,
                normalized_coverage=normalized_coverage,
                mean_best_event_score=mean_score,
                min_best_event_score=minimum_score,
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


def adjacent_hinge_penalty(
    gap_seconds: float,
    *,
    tau_seconds: float,
    gap_lambda: float,
) -> float:
    """Penalty only the portion of a non-negative adjacent gap above ``tau``."""

    if gap_seconds < 0.0:
        raise ValueError("gap_seconds must be non-negative")
    if tau_seconds < 0.0:
        raise ValueError("tau_seconds must be non-negative")
    if gap_lambda < 0.0:
        raise ValueError("gap_lambda must be non-negative")
    return gap_lambda * max(0.0, gap_seconds - tau_seconds)


def _proposal_allowed(
    proposal: EventProposal,
    event_constraint: EventConstraint,
) -> bool:
    if proposal.user_status == "rejected":
        return False
    if proposal.id in event_constraint.rejected_proposal_ids:
        return False

    # Explicit frame selection is strongest and deliberately ignores a stale
    # fixed-region field. Rejections remain hard safety constraints.
    if event_constraint.fixed_frame_id is not None:
        if proposal.frame_id != event_constraint.fixed_frame_id:
            return False
        if (
            event_constraint.fixed_video_id is not None
            and proposal.video_id != event_constraint.fixed_video_id
        ):
            return False
        if (
            event_constraint.fixed_timestamp_seconds is not None
            and not isclose(
                proposal.timestamp_seconds,
                event_constraint.fixed_timestamp_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            return False
        return True

    if proposal.region_id in event_constraint.rejected_region_ids:
        return False
    if (
        event_constraint.fixed_region_id is not None
        and proposal.region_id != event_constraint.fixed_region_id
    ):
        return False
    if (
        event_constraint.fixed_video_id is not None
        and proposal.video_id != event_constraint.fixed_video_id
    ):
        return False
    if (
        event_constraint.fixed_timestamp_seconds is not None
        and not isclose(
            proposal.timestamp_seconds,
            event_constraint.fixed_timestamp_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        return False
    return True


def _gap_constraint_map(
    constraints: SearchConstraints,
) -> dict[tuple[str, str], AdjacentGapConstraint]:
    return {
        (constraint.before_event_id, constraint.after_event_id): constraint
        for constraint in constraints.adjacent_gap_constraints
    }


def _video_overrides_allowlist(
    video_id: str,
    proposals: Sequence[EventProposal],
    constraints: SearchConstraints,
) -> bool:
    for event_id, event_constraint in constraints.event_constraints.items():
        if event_constraint.fixed_video_id == video_id:
            return True
        if event_constraint.fixed_region_id is not None and any(
            proposal.event_id == event_id
            and proposal.video_id == video_id
            and proposal.region_id == event_constraint.fixed_region_id
            for proposal in proposals
        ):
            return True
    return False


def assemble_ordered_tuples(
    events: Sequence[EventDefinition],
    proposals: Sequence[EventProposal],
    constraints: SearchConstraints | None = None,
    parameters=None,
) -> list[TupleResult]:
    """Assemble ordered, same-video event tuples under hard and soft limits."""

    # Avoid a circular annotation import solely for a default constructor.
    from .schemas import RankingHyperparameters

    constraints = constraints or SearchConstraints()
    parameters = parameters or RankingHyperparameters()
    event_ids = tuple(event.event_id for event in events)
    if not event_ids:
        return []
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event definitions must have unique event_id values")

    gap_constraints = _gap_constraint_map(constraints)
    active_proposals = [
        proposal
        for proposal in proposals
        if proposal.event_id in event_ids
        and proposal.video_id not in constraints.rejected_video_ids
        and _proposal_allowed(
            proposal, _event_constraint(constraints, proposal.event_id)
        )
    ]
    by_video: dict[str, list[EventProposal]] = defaultdict(list)
    for proposal in active_proposals:
        by_video[proposal.video_id].append(proposal)

    all_results: list[TupleResult] = []
    for video_id in sorted(by_video):
        video_proposals = by_video[video_id]
        if (
            constraints.allowed_video_ids
            and video_id not in constraints.allowed_video_ids
            and not _video_overrides_allowlist(
                video_id, video_proposals, constraints
            )
        ):
            continue

        choices: list[list[EventProposal]] = []
        missing_event = False
        for event_id in event_ids:
            event_choices = sorted(
                [
                    proposal
                    for proposal in video_proposals
                    if proposal.event_id == event_id
                ],
                key=_proposal_rank_key,
            )[: parameters.max_proposals_per_event_per_video]
            if not event_choices:
                missing_event = True
                break
            choices.append(event_choices)
        if missing_event:
            continue

        video_results: list[TupleResult] = []
        completed_combinations = 0

        def backtrack(selected: list[EventProposal]) -> None:
            nonlocal completed_combinations
            if completed_combinations >= parameters.max_combinations_per_video:
                return
            event_index = len(selected)
            if event_index == len(event_ids):
                completed_combinations += 1
                timestamps = [proposal.timestamp_seconds for proposal in selected]
                if (
                    constraints.max_tuple_span_seconds is not None
                    and timestamps[-1] - timestamps[0]
                    > constraints.max_tuple_span_seconds
                ):
                    return

                gaps: list[float] = []
                penalties: list[float] = []
                for index in range(len(selected) - 1):
                    gap = timestamps[index + 1] - timestamps[index]
                    pair = (event_ids[index], event_ids[index + 1])
                    gap_constraint = gap_constraints.get(pair)
                    if gap_constraint is not None:
                        if gap < gap_constraint.min_gap_seconds:
                            return
                        if (
                            gap_constraint.max_gap_seconds is not None
                            and gap > gap_constraint.max_gap_seconds
                        ):
                            return
                    tau = (
                        gap_constraint.hinge_tau_seconds
                        if gap_constraint is not None
                        and gap_constraint.hinge_tau_seconds is not None
                        else parameters.default_gap_tau_seconds
                    )
                    gap_lambda = (
                        gap_constraint.gap_lambda
                        if gap_constraint is not None
                        and gap_constraint.gap_lambda is not None
                        else parameters.gap_lambda
                    )
                    gaps.append(gap)
                    penalties.append(
                        adjacent_hinge_penalty(
                            gap, tau_seconds=tau, gap_lambda=gap_lambda
                        )
                    )

                event_mean = fmean(
                    proposal.final_event_score for proposal in selected
                )
                mean_penalty = fmean(penalties) if penalties else 0.0
                fixed_count = sum(
                    any(
                        value is not None
                        for value in (
                            event_constraint.fixed_video_id,
                            event_constraint.fixed_region_id,
                            event_constraint.fixed_frame_id,
                            event_constraint.fixed_timestamp_seconds,
                        )
                    )
                    for event_constraint in (
                        _event_constraint(constraints, event_id)
                        for event_id in event_ids
                    )
                )
                constraint_bonus = (
                    parameters.fixed_constraint_bonus
                    * fixed_count
                    / len(event_ids)
                )
                final_score = event_mean - mean_penalty + constraint_bonus
                proposal_ids = tuple(proposal.id for proposal in selected)
                video_results.append(
                    TupleResult(
                        id=_stable_id("tuple", video_id, *proposal_ids),
                        video_id=video_id,
                        proposals=tuple(selected),
                        adjacent_gaps_seconds=tuple(gaps),
                        adjacent_gap_penalties=tuple(penalties),
                        raw_event_mean_score=event_mean,
                        raw_gap_penalty=mean_penalty,
                        raw_constraint_bonus=constraint_bonus,
                        raw_final_score=final_score,
                    )
                )
                return

            for proposal in choices[event_index]:
                if selected:
                    gap = proposal.timestamp_seconds - selected[-1].timestamp_seconds
                    if parameters.require_strict_order and gap <= 0.0:
                        continue
                    if not parameters.require_strict_order and gap < 0.0:
                        continue
                    pair = (event_ids[event_index - 1], event_ids[event_index])
                    gap_constraint = gap_constraints.get(pair)
                    if gap_constraint is not None:
                        if gap < gap_constraint.min_gap_seconds:
                            continue
                        if (
                            gap_constraint.max_gap_seconds is not None
                            and gap > gap_constraint.max_gap_seconds
                        ):
                            continue
                selected.append(proposal)
                backtrack(selected)
                selected.pop()
                if completed_combinations >= parameters.max_combinations_per_video:
                    break

        backtrack([])
        video_results.sort(
            key=lambda item: (
                -item.raw_final_score,
                -item.raw_event_mean_score,
                item.raw_gap_penalty,
                item.id,
            )
        )
        all_results.extend(video_results[: parameters.max_tuples_per_video])

    all_results.sort(
        key=lambda item: (
            -item.raw_final_score,
            -item.raw_event_mean_score,
            item.raw_gap_penalty,
            item.video_id,
            item.id,
        )
    )
    all_results = all_results[: parameters.max_total_tuples]
    normalized_scores = robust_sigmoid(
        [result.raw_final_score for result in all_results]
    )
    normalized_results = [
        result.model_copy(update={"normalized_final_score": normalized_scores[index]})
        for index, result in enumerate(all_results)
    ]
    return normalized_results[: parameters.top_k]
