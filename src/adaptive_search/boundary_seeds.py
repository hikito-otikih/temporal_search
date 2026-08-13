"""Region-based seed selection for the boundary-refinement stage.

`prioritize_videos()` ranks whole videos from `TemporalRegion` spans - it has
no notion of "which single frame represents this event for this video" at
all. This module supplies that: for a given (event, video) pair, rank that
pair's regions by `raw_coarse_score` (a plain max over the region's clustered
candidates - the same signal that already determines the video's rank, not
an unrelated secondary one), keep whichever regions are within
`SCORE_THRESHOLD_RATIO` of the best (capped at `MAX_REGIONS_PER_EVENT`), and
return every candidate timestamp belonging to those surviving regions as a
seed, best-score-first, capped at `MAX_SEEDS_PER_EVENT`.

Ported from `irrelevant_things/benchmarks/youcook2/coarse_anchor.py`,
validated there against real YouCook2 queries this session -
`SCORE_THRESHOLD_RATIO`/`MAX_REGIONS_PER_EVENT`/`MAX_SEEDS_PER_EVENT` are
unchanged, tuned there, not re-derived here. The one real difference: this
module reads data already present in a session's `bundle.artifacts.regions`/
`.candidates` - it never re-derives them via a fresh retrieval call, unlike
the benchmark version, which bypassed this repo's own HTTP API on purpose to
sweep many hyperparameter configurations cheaply.

Only the first (best-ranked) seed a caller receives is meant to be passed to
`boundary_refinement.refine_event_boundary()` - comparing scores across
several independently-swept seed neighborhoods was tried and found
structurally invalid this session (see `boundary_refinement.py`'s
docstring). The remaining seeds exist only to make that top seed more likely
correct in the first place, not to be refined themselves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .schemas import SparseCandidate, TemporalRegion

# A region survives as a seed source if its raw_coarse_score is within this
# fraction of the best region's score for the same (event, video) pair.
SCORE_THRESHOLD_RATIO = 0.1
# How many distinct temporal clusters (occurrences) to draw seeds from. Kept
# small on purpose - boundary refinement targets precision within one
# occurrence, not disambiguating between several separate ones.
MAX_REGIONS_PER_EVENT = 2
# Overall cap on total seeds (summed across surviving regions), best-score
# first. Real region-size distribution (measured this session): median 1,
# mean 1.65, max 16 candidates per region.
MAX_SEEDS_PER_EVENT = 6


def select_event_seeds(
    *,
    regions: Sequence[TemporalRegion],
    candidates_by_id: Mapping[str, SparseCandidate],
    event_id: str,
    video_id: str,
) -> list[float]:
    """Best-score-first candidate timestamps for one (event, video) pair."""

    matching_regions = [
        region
        for region in regions
        if region.event_id == event_id and region.video_id == video_id
    ]
    if not matching_regions:
        return []

    ranked_regions = sorted(matching_regions, key=lambda region: -region.raw_coarse_score)
    best_region_score = ranked_regions[0].raw_coarse_score
    region_threshold = best_region_score * (1.0 - SCORE_THRESHOLD_RATIO)
    surviving_regions = [
        region for region in ranked_regions if region.raw_coarse_score >= region_threshold
    ][:MAX_REGIONS_PER_EVENT]

    seed_candidates = [
        candidates_by_id[candidate_id]
        for region in surviving_regions
        for candidate_id in region.candidate_ids
        if candidate_id in candidates_by_id
    ]
    seed_candidates.sort(key=lambda candidate: -candidate.raw_relevance_score)
    return [candidate.timestamp_seconds for candidate in seed_candidates][:MAX_SEEDS_PER_EVENT]
