"""Pure scoring/calibration math: robust normalization and the
population-relative candidate/region score maps everything else in this
package builds on. No domain-object assembly (regions, rankings) lives
here."""

from __future__ import annotations

from collections import defaultdict
from math import exp, isclose
from statistics import median
from typing import Sequence

from ..schemas import SparseCandidate, TemporalRegion

_EPSILON = 1e-12


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
