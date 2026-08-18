"""Bypass-HTTP `adaptive_coarse` "before" state: video rank + per-event seed anchors.

`adaptive_coarse` has no existing per-event frame concept in production - it
only ranks whole videos via `prioritize_videos()` over `TemporalRegion` spans
(no single frame). There's also no HTTP endpoint that exposes the underlying
`SparseCandidate` objects (only `TemporalRegion.candidate_ids`, bare id
strings with no join back to their scores/timestamps). So this module calls
the retrieval/ranking primitives directly in Python - the same pattern this
session's earlier hyperparameter sweeps already used.

Seed selection is region-based, not a flat candidate-pool score cutoff: for a
given (event, video), rank the `TemporalRegion`s that `cluster_temporal_regions()`
already produces (the same regions `prioritize_videos()` uses for `rank`) by
`raw_coarse_score`, keep whichever regions are within `SCORE_THRESHOLD_RATIO`
of the best one (capped at `MAX_REGIONS_PER_EVENT`), then use every candidate
timestamp belonging to those surviving regions as a seed (capped overall at
`MAX_SEEDS_PER_EVENT`). `raw_coarse_score` is already a `robust_sigmoid`-
calibrated aggregate over a temporally-coherent cluster of candidates
(algorithms.py) - a materially different, arguably more robust signal than
any single candidate's own RRF score, and it's the exact signal that already
determines the video's rank - so anchor selection is now consistent with
*why* the video ranked where it did, instead of an unrelated secondary
signal. (Earlier version of this module ignored region membership entirely
and just score-thresholded the flat candidate pool; that missed exactly the
information this docstring now uses.) This does not exist in production; it
exists only to give the boundary refiner something to start from for a
pipeline that has never picked individual frames before.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import VideoQueryGroup, canonical_video_id
from .legacy_clustering import ClusteringHyperparameters, cluster_temporal_regions
from adaptive_search.algorithms import prioritize_videos
from adaptive_search.client import QueryVariant, UpstreamSearchClient
from adaptive_search.retrieval import fuse_candidates_rrf
from adaptive_search.schemas import (
    RefinementHyperparameters,
    RetrievalHyperparameters,
)

# The winner config from this session's Table 1 hyperparameter sweep
# (hyperparameter_sweep_results.md): wide retrieval + mean-only weights.
WINNER_RETRIEVAL_PARAMS = RetrievalHyperparameters(
    top_n_per_variant=500, top_n_fused=1000, rrf_k=60
)
WINNER_REFINEMENT_PARAMS = RefinementHyperparameters(
    video_coverage_weight=0.0, video_mean_weight=1.0, video_min_weight=0.0
)
# top_n_per_variant=500 only binds if the raw upstream fetch is at least that
# large - matches the sweep's own fixed UPSTREAM_TOP_K.
WINNER_UPSTREAM_TOP_K = 500

# A region survives as a seed source if its raw_coarse_score is within this
# fraction of the best region's score for the same (event, video) pair.
SCORE_THRESHOLD_RATIO = 0.1
# How many distinct temporal clusters (occurrences) to draw seeds from. Kept
# small on purpose - this experiment targets boundary precision within one
# occurrence, not disambiguating between several separate ones; it exists
# mainly so a couple of near-tied regions can both contribute, not to chase
# every region that clears the threshold.
MAX_REGIONS_PER_EVENT = 2
# Overall cap on total seeds (summed across surviving regions), best-score
# first. The real region-size distribution (verified this session) has
# median 1, mean 1.65, max 16 candidates per region - raised from the old
# flat-pool cap of 3 to comfortably cover "2 regions x a few candidates each"
# without letting a rare oversized region blow up the refinement pass.
MAX_SEEDS_PER_EVENT = 6


@dataclass(frozen=True)
class CoarseBeforeResult:
    rank: int | None
    unique_video_count: int
    # event_id -> seed timestamps, best-score-first, GT video only
    event_anchors: dict[str, tuple[float, ...]]


def adaptive_coarse_before(
    group: VideoQueryGroup,
    *,
    upstream: UpstreamSearchClient,
    retrieval_params: RetrievalHyperparameters = WINNER_RETRIEVAL_PARAMS,
    clustering_params: ClusteringHyperparameters | None = None,
    refinement_params: RefinementHyperparameters = WINNER_REFINEMENT_PARAMS,
    top_k: int = WINNER_UPSTREAM_TOP_K,
) -> CoarseBeforeResult:
    session_id = f"boundary_exp_coarse_{group.video_id}"
    variants = [
        QueryVariant(event_id=event_id, variant_id=f"{event_id}:v0", text=text)
        for event_id, text in group.events
    ]
    candidates = upstream.retrieve_candidates(
        session_id=session_id, variants=variants, top_k=top_k
    )
    fused = fuse_candidates_rrf(candidates, retrieval_params)

    regions = cluster_temporal_regions(fused, clustering_params)
    event_ids = [event_id for event_id, _ in group.events]
    priorities = prioritize_videos(regions, event_ids, refinement_params)
    rank = next(
        (
            position
            for position, priority in enumerate(priorities, 1)
            if canonical_video_id(priority.video_id) == group.video_id
        ),
        None,
    )

    candidates_by_id = {candidate.id: candidate for candidate in fused}

    event_anchors: dict[str, tuple[float, ...]] = {}
    for event_id, _ in group.events:
        matching_regions = [
            region
            for region in regions
            if region.event_id == event_id
            and canonical_video_id(region.video_id) == group.video_id
        ]
        if not matching_regions:
            continue

        ranked_regions = sorted(matching_regions, key=lambda region: -region.raw_coarse_score)
        best_region_score = ranked_regions[0].raw_coarse_score
        region_threshold = best_region_score * (1.0 - SCORE_THRESHOLD_RATIO)
        surviving_regions = [
            region
            for region in ranked_regions
            if region.raw_coarse_score >= region_threshold
        ][:MAX_REGIONS_PER_EVENT]

        seed_candidates = [
            candidates_by_id[candidate_id]
            for region in surviving_regions
            for candidate_id in region.candidate_ids
        ]
        seed_candidates.sort(key=lambda candidate: -candidate.raw_relevance_score)
        seeds = [candidate.timestamp_seconds for candidate in seed_candidates][:MAX_SEEDS_PER_EVENT]
        event_anchors[event_id] = tuple(seeds)

    return CoarseBeforeResult(
        rank=rank, unique_video_count=len(priorities), event_anchors=event_anchors
    )
