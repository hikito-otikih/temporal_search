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
    if (
        event_constraint.fixed_video_id is not None
        and region.video_id == event_constraint.fixed_video_id
    ):
        # A fix_frame pin narrows only the *fixed video's own* region choice
        # for this event - it must never remove other videos' regions from
        # the pool, or every other video would lose all coverage of this
        # event and vanish from ranking entirely (they'd have no region left
        # to build a tuple from). Cross-video ordering is a separate,
        # explicit mechanism (prioritized_video_ids), not a side effect of
        # fixing a frame.
        #
        # An exact-timestamp pin (fixed_timestamp_seconds set) narrows this
        # event, within the pinned video, to only the region actually
        # covering that timestamp - mirrors _synthetic_fixed_regions'
        # own trigger condition exactly (same "defer to a genuine
        # fixed_region_id pin" carve-out below), so both functions agree on
        # what a given constraint means. fixed_region_id itself is not
        # checked here since fix_frame() always stamps one too, often a
        # synthetic placeholder that names no real TemporalRegion. Without
        # this, once a real region already existed at the fixed timestamp
        # (so _synthetic_fixed_regions had nothing to add), every other
        # region for this event in the same video remained equally
        # eligible and a higher-scoring one could still win pooling over
        # the frame the user explicitly picked.
        exact_timestamp_pin = (
            event_constraint.fixed_timestamp_seconds is not None
            and not (
                event_constraint.fixed_frame_id is None
                and event_constraint.fixed_region_id is not None
            )
        )
        if exact_timestamp_pin and not (
            region.start_seconds
            <= event_constraint.fixed_timestamp_seconds
            <= region.end_seconds
        ):
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
