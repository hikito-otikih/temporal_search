"""Boundary-profile-aware event proposal generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256
from statistics import fmean

from .algorithms import (
    calibrate_frame_scores,
    generate_boundary_proposals,
    robust_sigmoid,
    temporal_nms,
)
from .models import (
    BoundaryHyperparameters,
    EventDefinition,
    EventProposal,
    FrameScoreSample,
    ProposalSource,
)


_TRANSITION_PROFILES = {"onset", "offset", "transition", "plateau_start"}


def generate_profiled_proposals(
    events: Sequence[EventDefinition],
    samples: Sequence[FrameScoreSample],
    parameters: BoundaryHyperparameters | None = None,
    *,
    source: ProposalSource = "dense",
) -> list[EventProposal]:
    """Dispatch transition, persistent-state and symmetric-peak scorers.

    The original left/right pre-to-post contrast is only valid for transition
    profiles. State events use local anchor persistence. Symmetric peaks use
    center prominence relative to both flanks.
    """

    parameters = parameters or BoundaryHyperparameters()
    event_by_id = {event.event_id: event for event in events}
    if len(event_by_id) != len(events):
        raise ValueError("event definitions must have unique event_id values")

    grouped: dict[str, list[FrameScoreSample]] = defaultdict(list)
    for sample in samples:
        if sample.event_id not in event_by_id:
            raise ValueError(f"frame score references unknown event: {sample.event_id}")
        grouped[sample.event_id].append(sample)

    proposals: list[EventProposal] = []
    for event_id in sorted(grouped):
        event = event_by_id[event_id]
        profile = event.boundary_type
        if profile == "unknown":
            profile = (
                "transition"
                if event.pre_state is not None and event.post_state is not None
                else "state"
            )

        event_samples = grouped[event_id]
        if profile in _TRANSITION_PROFILES:
            proposals.extend(
                generate_boundary_proposals(
                    event_samples,
                    parameters,
                    source=source,
                    apply_nms=False,
                )
            )
        elif profile == "state":
            proposals.extend(
                _generate_state_proposals(event_samples, parameters, source)
            )
        elif profile == "symmetric_peak":
            proposals.extend(
                _generate_peak_proposals(event_samples, parameters, source)
            )
        else:  # pragma: no cover - guarded by the Pydantic BoundaryType literal
            raise ValueError(f"unsupported boundary profile: {profile}")

    return temporal_nms(
        proposals,
        radius_seconds=parameters.nms_radius_seconds,
        max_per_group=parameters.max_proposals_per_region,
    )


def _generate_state_proposals(
    samples: Sequence[FrameScoreSample],
    parameters: BoundaryHyperparameters,
    source: ProposalSource,
) -> list[EventProposal]:
    calibrated = calibrate_frame_scores(samples, parameters)
    grouped = _group_samples(calibrated)
    proposals: list[EventProposal] = []
    radius = max(parameters.window_options_seconds)
    total_weight = (
        parameters.semantic_weight
        + parameters.boundary_weight
        + parameters.pre_weight
        + parameters.post_weight
    )

    for group_key in sorted(grouped):
        timeline = grouped[group_key]
        for center in timeline:
            neighborhood = [
                sample
                for sample in timeline
                if abs(sample.timestamp_seconds - center.timestamp_seconds) <= radius
            ]
            persistence = fmean(
                float(sample.normalized_anchor_score) for sample in neighborhood
            )
            semantic = float(center.normalized_anchor_score)
            final_score = (
                parameters.semantic_weight * semantic
                + (
                    parameters.boundary_weight
                    + parameters.pre_weight
                    + parameters.post_weight
                )
                * persistence
            ) / total_weight
            proposals.append(
                _proposal(
                    center,
                    profile="state",
                    raw_boundary=0.0,
                    normalized_boundary=persistence,
                    pre_consistency=persistence,
                    post_persistence=persistence,
                    final_score=final_score,
                    left_window=radius,
                    right_window=radius,
                    source=source,
                )
            )
    return proposals


def _generate_peak_proposals(
    samples: Sequence[FrameScoreSample],
    parameters: BoundaryHyperparameters,
    source: ProposalSource,
) -> list[EventProposal]:
    calibrated = calibrate_frame_scores(samples, parameters)
    grouped = _group_samples(calibrated)
    proposals: list[EventProposal] = []
    total_weight = (
        parameters.semantic_weight
        + parameters.boundary_weight
        + parameters.pre_weight
        + parameters.post_weight
    )

    for group_key in sorted(grouped):
        timeline = grouped[group_key]
        scored: list[tuple[FrameScoreSample, float, float, float, float]] = []
        for center in timeline:
            best = None
            for window in parameters.window_options_seconds:
                left = [
                    sample
                    for sample in timeline
                    if center.timestamp_seconds - window
                    <= sample.timestamp_seconds
                    < center.timestamp_seconds
                ]
                right = [
                    sample
                    for sample in timeline
                    if center.timestamp_seconds
                    < sample.timestamp_seconds
                    <= center.timestamp_seconds + window
                ]
                if (
                    len(left) < parameters.min_samples_per_side
                    or len(right) < parameters.min_samples_per_side
                ):
                    continue
                left_mean = fmean(
                    float(sample.normalized_anchor_score) for sample in left
                )
                right_mean = fmean(
                    float(sample.normalized_anchor_score) for sample in right
                )
                prominence = float(center.normalized_anchor_score) - 0.5 * (
                    left_mean + right_mean
                )
                candidate = (prominence, -window, left_mean, right_mean, window)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if best is not None:
                prominence, _, left_mean, right_mean, window = best
                scored.append((center, prominence, left_mean, right_mean, window))

        normalized_prominence = robust_sigmoid([item[1] for item in scored])
        for item, normalized_boundary in zip(
            scored, normalized_prominence, strict=True
        ):
            center, prominence, left_mean, right_mean, window = item
            semantic = float(center.normalized_anchor_score)
            left_low = 1.0 - left_mean
            right_low = 1.0 - right_mean
            final_score = (
                parameters.semantic_weight * semantic
                + parameters.boundary_weight * normalized_boundary
                + parameters.pre_weight * left_low
                + parameters.post_weight * right_low
            ) / total_weight
            proposals.append(
                _proposal(
                    center,
                    profile="symmetric_peak",
                    raw_boundary=prominence,
                    normalized_boundary=normalized_boundary,
                    pre_consistency=left_low,
                    post_persistence=right_low,
                    final_score=final_score,
                    left_window=window,
                    right_window=window,
                    source=source,
                )
            )
    return proposals


def _group_samples(
    samples: Sequence[FrameScoreSample],
) -> dict[tuple[str, str, str, str], list[FrameScoreSample]]:
    grouped: dict[tuple[str, str, str, str], list[FrameScoreSample]] = defaultdict(
        list
    )
    for sample in samples:
        grouped[
            (sample.session_id, sample.event_id, sample.video_id, sample.region_id)
        ].append(sample)
    for key in grouped:
        grouped[key].sort(key=lambda item: (item.timestamp_seconds, item.frame_id))
    return grouped


def _proposal(
    center: FrameScoreSample,
    *,
    profile: str,
    raw_boundary: float,
    normalized_boundary: float,
    pre_consistency: float,
    post_persistence: float,
    final_score: float,
    left_window: float,
    right_window: float,
    source: ProposalSource,
) -> EventProposal:
    payload = "\x1f".join(
        (
            center.session_id,
            center.event_id,
            center.video_id,
            center.region_id,
            str(center.frame_id),
            str(center.timestamp_seconds),
            profile,
        )
    ).encode("utf-8")
    return EventProposal(
        id=f"proposal_{sha256(payload).hexdigest()[:20]}",
        session_id=center.session_id,
        event_id=center.event_id,
        video_id=center.video_id,
        region_id=center.region_id,
        timestamp_seconds=center.timestamp_seconds,
        frame_id=center.frame_id,
        raw_semantic_score=center.raw_anchor_score,
        normalized_semantic_score=float(center.normalized_anchor_score),
        raw_boundary_score=raw_boundary,
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
