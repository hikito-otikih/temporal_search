"""Pure scoring/calibration math: robust normalization, pairwise softmax,
frame-score calibration, and the population-relative candidate/region score
maps everything else in this package builds on. No domain-object assembly
(regions, proposals, rankings) lives here."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from math import exp, isclose
from statistics import fmean, median
from typing import Sequence

from ..schemas import BoundaryHyperparameters, FrameScoreSample, SparseCandidate, TemporalRegion

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


def _mean(samples: Sequence[FrameScoreSample], attribute: str) -> float:
    values = [getattr(sample, attribute) for sample in samples]
    if any(value is None for value in values):
        raise ValueError("frame scores must be calibrated before proposal generation")
    return fmean(float(value) for value in values)


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


def adjacent_hinge_penalty(
    gap_seconds: float,
    *,
    tau_seconds: float,
    gap_lambda: float,
) -> float:
    """Penalty only the portion of a non-negative adjacent gap above ``tau``.

    Used by `tuple_ranking.py`'s `adjacent_gap_constraints` soft-penalty
    enforcement - a general-purpose utility, not exclusive to the retired
    proposal-based `assemble_ordered_tuples` mechanism it originally shipped
    alongside."""

    if gap_seconds < 0.0:
        raise ValueError("gap_seconds must be non-negative")
    if tau_seconds < 0.0:
        raise ValueError("tau_seconds must be non-negative")
    if gap_lambda < 0.0:
        raise ValueError("gap_lambda must be non-negative")
    return gap_lambda * max(0.0, gap_seconds - tau_seconds)
