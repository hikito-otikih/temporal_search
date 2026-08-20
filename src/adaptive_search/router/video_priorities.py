"""The video-priorities read endpoint: order-aware tuple ranking plus optional boundary refinement."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ..algorithms import distinctness_from_timestamps, robust_sigmoid
from ..boundary_refinement import refine_event_boundary
from ..dependencies import adaptive_service, boundary_refinement_runtime
from ..providers import VideoAssetNotFoundError
from ..schemas import VideoPriority
from ..tuple_ranking import build_order_constraints, rank_videos_by_region_tuples

from .errors import _raise_api_error

router = APIRouter()


@router.get("/search-sessions/{session_id}/video-priorities")
def get_video_priorities(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    apply_boundary_refinement: bool = Query(default=False),
):
    try:
        bundle = adaptive_service.get_session(session_id)
        event_ids = [event.event_id for event in bundle.session.events]
        candidates_by_id = {c.id: c for c in bundle.artifacts.candidates}

        # Order-aware tuple ranking is the only ranking algorithm this
        # endpoint runs - prioritize_videos() (independent per-event argmax)
        # is no longer called here at all. It remains real, tested code -
        # the always-available reference baseline every benchmark in
        # region_tuple_ranking_results.md compares against - just not
        # reachable through this endpoint anymore; there is no evidence it
        # ever produces a better result, and threading its own SearchConstraints
        # filtering + VideoPriority construction through two independent
        # code paths was exactly the kind of duplication that let five of
        # seven VideoPriority fields silently go stale (fixed the round
        # before this one).
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
        normalized_by_video = dict(zip(
            (video_id for video_id, _ in tuple_ranking),
            robust_sigmoid([score for _, score in tuple_ranking]),
        ))
        refinement_params = bundle.session.hyperparameters.refinement
        distinctness_norm = refinement_params.distinctness_norm_seconds
        # The tuple's own (coverage+order-aware) sigmoid score already plays
        # the role prioritize_videos()'s coverage/mean/min weights jointly
        # play there - there's no separate coverage/mean/min signal here to
        # split video_distinctness_weight against, so it's blended against
        # that combined weight instead. distinctness_weight=0.0 (the
        # default) reduces this exactly to the un-blended score: previously
        # video_distinctness_weight was silently read only by
        # prioritize_videos(), which this endpoint stopped calling entirely
        # once tuple ranking became unconditional - the weight was fully
        # advertised and settable but had zero effect here.
        other_weight = (
            refinement_params.video_coverage_weight
            + refinement_params.video_mean_weight
            + refinement_params.video_min_weight
        )
        distinctness_weight = refinement_params.video_distinctness_weight
        weight_sum = other_weight + distinctness_weight

        raw_by_video = dict(tuple_ranking)
        blended: list[tuple[str, float, object, float, float, int]] = []
        for video_id, raw_priority_score in normalized_by_video.items():
            winner = tuples_by_video[video_id][0]
            covered_timestamps = [t for t in winner.timestamps if t is not None]
            distinctness = distinctness_from_timestamps(covered_timestamps, distinctness_norm)
            priority_score = (
                other_weight * raw_priority_score + distinctness_weight * distinctness
            ) / weight_sum
            coverage = sum(1 for region_id in winner.region_ids if region_id is not None)
            blended.append((video_id, priority_score, winner, distinctness, raw_by_video[video_id], coverage))

        priorities = []
        tuple_anchors_by_video: dict[str, dict[str, float]] = {}
        # priority_score (post robust_sigmoid, which clamps at clip_z=8.0 -
        # see algorithms/scoring.py) ties easily once several videos' raw
        # scores all exceed the clamp threshold. Break ties with the raw,
        # pre-normalization score, then with event coverage, before finally
        # falling back to video_id for determinism.
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
                )
            )
            tuple_anchors_by_video[video_id] = {
                event_id: timestamp
                for event_id, timestamp in zip(event_ids, winner.timestamps)
                if timestamp is not None
            }

        # prioritized_video_ids reorders the response only - it never
        # touches priority_score or any video's own tuple/region choice
        # (distinct from fix_frame, which is scoped to its own video and
        # deliberately does not affect cross-video ordering at all).
        prioritized_order = bundle.session.constraints.prioritized_video_ids
        if prioritized_order:
            by_video_id = {item.video_id: item for item in priorities}
            prioritized_set = set(prioritized_order)
            front = [by_video_id[vid] for vid in prioritized_order if vid in by_video_id]
            rest = [item for item in priorities if item.video_id not in prioritized_set]
            priorities = front + rest

        page = priorities[offset : offset + limit]

        capability_available = False
        capability_reason: str | None = None
        if apply_boundary_refinement:
            runtime_spec = (
                refinement_params.quality_embedding_model
                if refinement_params.quality_profile_enabled
                else refinement_params.embedding_model
            )
            capability = boundary_refinement_runtime.capabilities(runtime_spec)
            capability_available = capability.available
            capability_reason = capability.reason

        items = [
            _video_priority_with_boundary_refinement(
                priority,
                bundle=bundle,
                requested=apply_boundary_refinement,
                capability_available=capability_available,
                capability_reason=capability_reason,
                tuple_anchors=tuple_anchors_by_video[priority.video_id],
            )
            for priority in page
        ]
        return {
            "items": items,
            "total": len(priorities),
            "offset": offset,
            "limit": limit,
            "boundary_refinement_capability": {
                "requested": apply_boundary_refinement,
                "available": capability_available,
                "reason": capability_reason,
            },
        }
    except Exception as exc:
        _raise_api_error(exc)


def _video_priority_with_boundary_refinement(
    priority,
    *,
    bundle,
    requested: bool,
    capability_available: bool,
    capability_reason: str | None,
    tuple_anchors: dict[str, float],
) -> dict[str, Any]:
    payload = priority.model_dump(mode="json")
    if not requested:
        payload["boundary_refinement"] = {"status": "not_requested", "events": None}
        return payload
    if not capability_available:
        payload["boundary_refinement"] = {
            "status": "skipped_runtime_unavailable",
            "events": None,
        }
        return payload

    metadata = boundary_refinement_runtime.frame_provider.catalog.metadata(priority.video_id)
    if metadata is None or metadata.fps is None:
        payload["boundary_refinement"] = {"status": "skipped_no_metadata", "events": None}
        return payload

    events_out: list[dict[str, Any]] = []
    for event in bundle.session.events:
        # A user-confirmed frame (commands/fix-frame) is authoritative: skip
        # both seed-selection and the GPU refine call entirely for this event
        # - there's nothing to refine once the user already picked the exact
        # frame themselves.
        event_constraint = bundle.session.constraints.event_constraints.get(event.event_id)
        if (
            event_constraint is not None
            and event_constraint.fixed_video_id == priority.video_id
            and event_constraint.fixed_timestamp_seconds is not None
        ):
            events_out.append(
                {
                    "event_id": event.event_id,
                    "anchor_seconds": event_constraint.fixed_timestamp_seconds,
                    "refined_seconds": event_constraint.fixed_timestamp_seconds,
                    "used_fallback": False,
                    "boundary_type": event.boundary_type,
                    "sampled_frame_count": 0,
                    "source": "user_fixed",
                }
            )
            continue

        # The winning region-tuple already chose one region per event
        # jointly, order-aware - use its timestamp directly rather than
        # re-deriving an independent per-event argmax, which is exactly the
        # mechanism tuple ranking exists to avoid (see tuple_ranking.py's
        # module docstring). An event absent from tuple_anchors (not
        # covered by the winning tuple) has nothing to refine.
        anchor_seconds = tuple_anchors.get(event.event_id)
        if anchor_seconds is None:
            continue
        try:
            outcome = refine_event_boundary(
                provider=boundary_refinement_runtime.frame_provider,
                embedder=boundary_refinement_runtime.embedder,
                video_id=priority.video_id,
                event_id=event.event_id,
                anchor_query=event.anchor_query,
                pre_state=event.pre_state,
                post_state=event.post_state,
                boundary_type=event.boundary_type,
                anchor_seconds=anchor_seconds,
                fps=metadata.fps,
                duration_ms=metadata.duration_ms,
                parameters=bundle.session.hyperparameters.boundary,
            )
        except VideoAssetNotFoundError:
            payload["boundary_refinement"] = {
                "status": "skipped_video_not_in_catalog",
                "events": None,
            }
            return payload
        events_out.append(
            {
                "event_id": event.event_id,
                "anchor_seconds": outcome.anchor_seconds,
                "refined_seconds": outcome.refined_seconds,
                "used_fallback": outcome.used_fallback,
                "boundary_type": event.boundary_type,
                "sampled_frame_count": outcome.sampled_frame_count,
                "source": "tuple_ranking",
            }
        )
    payload["boundary_refinement"] = {
        "status": "applied" if events_out else "skipped_no_metadata",
        "events": events_out or None,
    }
    return payload
