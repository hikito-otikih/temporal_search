"""Boundary proposal generation (finite left/right window scoring) and
temporal non-maximum suppression."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..schemas import BoundaryHyperparameters, EventProposal, FrameScoreSample
from .scoring import _mean, _stable_id, calibrate_frame_scores, robust_sigmoid


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
