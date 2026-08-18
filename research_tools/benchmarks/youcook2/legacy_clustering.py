"""Merged-candidate region formation - superseded in production by
`adaptive_search.algorithms.atomic_regions` (one region per candidate, never
merged; see that function's docstring for the full comparison this module's
functions were kept to support).

Relocated here (not deleted) because `region_tuple_experiment.py` and
`coarse_anchor.py` still use it as the "clustered" arm of their atomic-vs-
clustered ablation - the real n=60 comparison
(region_tuple_ranking_results.md) that justified retiring clustering from
`src/adaptive_search/` in the first place. Reuses production's own stable-ID
and population-relative score normalization helpers directly
(`adaptive_search.algorithms._stable_id`,
`_candidate_normalized_scores_by_event`) rather than re-implementing them, so
this stays a faithful comparison arm, not a parallel reimplementation that
could quietly drift from what production actually did before atomic regions
replaced it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from adaptive_search.algorithms import _candidate_normalized_scores_by_event, _stable_id
from adaptive_search.schemas import ClusteringHyperparameters, SparseCandidate, TemporalRegion


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
