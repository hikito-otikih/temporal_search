"""Temporal region formation - the tail of `commands/retrieve`'s commit, not
a separate pipeline stage (see the paper's "note on stage boundaries")."""

from __future__ import annotations

from typing import Sequence

from ..schemas import SparseCandidate, TemporalRegion
from .scoring import _candidate_normalized_scores_by_event


def atomic_regions(
    candidates: Sequence[SparseCandidate], *, margin_seconds: float = 0.0
) -> list[TemporalRegion]:
    """One `TemporalRegion` per candidate - no merging of nearby candidates,
    ever - zero-width by default (`start_seconds == end_seconds ==
    candidate.timestamp_seconds`).

    The production region-formation path (`service.py::retrieve_candidates`,
    called with no margin - true zero-width). Merged-candidate clustering
    (`research_tools/benchmarks/youcook2/legacy_clustering.py`, relocated
    out of this module once nothing in production called it any more) was
    verified, not assumed, to have zero effect on
    `prioritize_videos()` and `select_event_seeds()` (each reduces to "the
    single best-scoring raw candidate," an identity merging can regroup but
    not change - every (event, video) pair's top seed was byte-identical
    between clustered and atomic input across a real n=60 corpus) and a
    small net *positive* for order-aware tuple ranking
    (`region_tuple_ranking_results.md`) - `_region_timestamp`
    (`tuple_ranking.py`) and `select_event_seeds` both read only a region's
    member candidates' own `timestamp_seconds`, never
    `start_seconds`/`end_seconds`, so region width was never doing anything
    for either. `replace_frame_scores` is the one exception - it needs a
    real acceptance window around a region's timestamp, but that window is
    now computed at validation time by `service.py::_frame_score_
    acceptance_window` (margin applied there, with a cross-event-overlap
    guard) rather than baked into every region up front via this
    function's `margin_seconds` parameter, which exists for callers that
    genuinely want padded regions (e.g. direct benchmark use) but is not
    how production constructs them.
    """

    normalized_by_id = _candidate_normalized_scores_by_event(candidates)
    return [
        TemporalRegion(
            id=f"atomic_{candidate.id}",
            session_id=candidate.session_id,
            event_id=candidate.event_id,
            video_id=candidate.video_id,
            start_seconds=max(0.0, candidate.timestamp_seconds - margin_seconds),
            end_seconds=candidate.timestamp_seconds + margin_seconds,
            candidate_ids=(candidate.id,),
            raw_coarse_score=candidate.raw_relevance_score,
            normalized_coarse_score=normalized_by_id[candidate.id],
        )
        for candidate in candidates
    ]
