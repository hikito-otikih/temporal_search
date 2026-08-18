"""Stage 6 (interactive correction) constraint filtering, shared by
`prioritize_videos`/`select_refinement_frontier` (this package) and
`tuple_ranking.rank_videos_by_region_tuples` (imports `_region_allowed`
directly - the same enforcement, not a parallel reimplementation)."""

from __future__ import annotations

from ..schemas import EventConstraint, SearchConstraints, TemporalRegion


def _event_constraint(
    constraints: SearchConstraints,
    event_id: str,
) -> EventConstraint:
    return constraints.event_constraints.get(event_id, EventConstraint())


def _region_allowed(
    region: TemporalRegion,
    constraints: SearchConstraints,
) -> bool:
    if region.user_status == "rejected":
        return False
    if region.video_id in constraints.rejected_video_ids:
        return False
    event_constraint = _event_constraint(constraints, region.event_id)
    if region.id in event_constraint.rejected_region_ids:
        return False
    if event_constraint.fixed_video_id is not None:
        if region.video_id != event_constraint.fixed_video_id:
            return False
    if (
        event_constraint.fixed_frame_id is None
        and event_constraint.fixed_region_id is not None
        and region.id != event_constraint.fixed_region_id
    ):
        return False

    allowlist_overridden = (
        event_constraint.fixed_video_id == region.video_id
        or event_constraint.fixed_region_id == region.id
    )
    return (
        not constraints.allowed_video_ids
        or region.video_id in constraints.allowed_video_ids
        or allowlist_overridden
    )
