"""The video-priorities read endpoint: order-aware tuple ranking over a session's artifacts."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..algorithms import distinctness_from_timestamps, robust_sigmoid
from ..dependencies import adaptive_service
from ..schemas import EventConstraint, SparseCandidate, TemporalRegion, VideoPriority
from ..session import SessionBundle
from ..tuple_ranking import TupleRankingResult, build_order_constraints, rank_videos_by_region_tuples

from .errors import _raise_api_error

router = APIRouter()

# Fixed reporting scale for VideoPriority.distinctness - not a session
# hyperparameter (it never affected ranking at this endpoint; see
# _compute_priorities()'s comment). Matches the pipeline's existing 3.0s
# margin/gap scale rather than introducing a new constant.
DISTINCTNESS_NORM_SECONDS = 3.0


def _matched_frame_ids(
    event_ids: list[str],
    winner: TupleRankingResult,
    regions_by_id: dict[str, TemporalRegion],
    candidates_by_id: dict[str, SparseCandidate],
    event_constraints: dict[str, EventConstraint],
    video_id: str,
) -> tuple[str, ...]:
    """One frame_id per session event, in event order, "?" wherever it can't
    be resolved. A commands/fix-frame pin for (event, video_id) always wins,
    checked fresh on every call - not baked into the tuple at ranking time -
    so the list reflects the current constraint state immediately. Otherwise
    resolves through the winning tuple's region_id -> its one backing
    candidate (atomic_regions() guarantees exactly one - see that function's
    docstring) -> SparseCandidate.frame_id. A synthetic fixed region
    (tuple_ranking._synthetic_fixed_regions) has a deliberately unresolvable
    candidate_id; that case is already caught by the fix-frame check above
    before reaching this fallback."""

    frame_ids: list[str] = []
    for event_id, region_id in zip(event_ids, winner.region_ids):
        constraint = event_constraints.get(event_id)
        if (
            constraint is not None
            and constraint.fixed_video_id == video_id
            and constraint.fixed_frame_id is not None
        ):
            frame_ids.append(str(constraint.fixed_frame_id))
            continue
        region = regions_by_id.get(region_id) if region_id is not None else None
        candidate = (
            candidates_by_id.get(region.candidate_ids[0])
            if region is not None and region.candidate_ids
            else None
        )
        frame_ids.append(str(candidate.frame_id) if candidate is not None else "?")
    return tuple(frame_ids)


def _compute_priorities(bundle: SessionBundle) -> list[VideoPriority]:
    """Rank every video in the session and build one VideoPriority per video,
    sorted, with `prioritized_video_ids` reordering already applied."""

    event_ids = [event.event_id for event in bundle.session.events]
    candidates_by_id = {c.id: c for c in bundle.artifacts.candidates}
    regions_by_id = {r.id: r for r in bundle.artifacts.regions}
    event_constraints = bundle.session.constraints.event_constraints
    # watch_url is a video-level property, not per-candidate/region - any
    # one of a video's candidates carries it, so the first one seen per
    # video_id is as good as any other.
    watch_url_by_video: dict[str, str] = {}
    for candidate in bundle.artifacts.candidates:
        if candidate.watch_url is not None:
            watch_url_by_video.setdefault(candidate.video_id, candidate.watch_url)

    # Order-aware tuple ranking is the only ranking algorithm this endpoint
    # runs - prioritize_videos() (independent per-event argmax) is no longer
    # called here at all. It remains real, tested code - the always-available
    # reference baseline every benchmark in region_tuple_ranking_results.md
    # compares against - just not reachable through this endpoint anymore;
    # there is no evidence it ever produces a better result, and threading
    # its own SearchConstraints filtering + VideoPriority construction
    # through two independent code paths was exactly the kind of duplication
    # that let five of seven VideoPriority fields silently go stale (fixed
    # the round before this one).
    order_constraints = build_order_constraints(bundle.session.events)
    tuple_ranking, tuples_by_video = rank_videos_by_region_tuples(
        bundle.artifacts.regions,
        candidates_by_id,
        event_ids,
        bundle.session.hyperparameters.tuple_ranking,
        order_constraints,
        bundle.session.constraints,
    )
    # Raw tuple scores aren't bounded to [0,1] (mean region score can be
    # pushed above 1 or below 0 by the order term) - normalize via
    # robust_sigmoid() before writing into priority_score's UnitScore field.
    # This normalized score is priority_score directly - no further blending
    # against coverage/mean/min/distinctness weights: those were a session-
    # level hyperparameter surface that degenerated to a no-op at this
    # endpoint (video_mean_weight was the only nonzero default, and a single
    # weight always cancels out of its own weighted average), so it was
    # removed from SearchHyperparameters entirely. distinctness is still
    # reported below - purely descriptive now, not a ranking input.
    normalized_by_video = dict(zip(
        (video_id for video_id, _ in tuple_ranking),
        robust_sigmoid([score for _, score in tuple_ranking]),
    ))

    raw_by_video = dict(tuple_ranking)
    blended: list[tuple[str, float, TupleRankingResult, float, float, int]] = []
    for video_id, priority_score in normalized_by_video.items():
        winner = tuples_by_video[video_id][0]
        covered_timestamps = [t for t in winner.timestamps if t is not None]
        distinctness = distinctness_from_timestamps(covered_timestamps, DISTINCTNESS_NORM_SECONDS)
        coverage = sum(1 for region_id in winner.region_ids if region_id is not None)
        blended.append((video_id, priority_score, winner, distinctness, raw_by_video[video_id], coverage))

    priorities: list[VideoPriority] = []
    # priority_score (post robust_sigmoid, which clamps at clip_z=8.0 - see
    # algorithms/scoring.py) ties easily once several videos' raw scores all
    # exceed the clamp threshold. Break ties with the raw, pre-normalization
    # score, then with event coverage, before finally falling back to
    # video_id for determinism.
    for video_id, priority_score, winner, distinctness, raw_score, coverage in sorted(
        blended, key=lambda item: (-item[1], -item[4], -item[5], item[0])
    ):
        priorities.append(
            VideoPriority(
                video_id=video_id,
                event_coverage=coverage,
                normalized_coverage=coverage / len(event_ids),
                mean_best_event_score=winner.region_mean_score,
                min_best_event_score=min(winner.region_scores),
                distinctness=distinctness,
                order_score=winner.order_score,
                raw_score=raw_score,
                priority_score=priority_score,
                watch_url=watch_url_by_video.get(video_id),
                matched_frame_ids=_matched_frame_ids(
                    event_ids, winner, regions_by_id, candidates_by_id, event_constraints, video_id,
                ),
            )
        )

    # prioritized_video_ids reorders the response only - it never touches
    # priority_score or any video's own tuple/region choice (distinct from
    # fix_frame, which is scoped to its own video and deliberately does not
    # affect cross-video ordering at all).
    prioritized_order = bundle.session.constraints.prioritized_video_ids
    if prioritized_order:
        by_video_id = {item.video_id: item for item in priorities}
        prioritized_set = set(prioritized_order)
        front = [by_video_id[vid] for vid in prioritized_order if vid in by_video_id]
        rest = [item for item in priorities if item.video_id not in prioritized_set]
        priorities = front + rest

    return priorities


@router.get("/search-sessions/{session_id}/video-priorities")
def get_video_priorities(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        bundle = adaptive_service.get_session(session_id)
        priorities = _compute_priorities(bundle)
        page = priorities[offset : offset + limit]
        return {
            "items": [priority.model_dump(mode="json") for priority in page],
            "total": len(priorities),
            "offset": offset,
            "limit": limit,
        }
    except Exception as exc:
        _raise_api_error(exc)
