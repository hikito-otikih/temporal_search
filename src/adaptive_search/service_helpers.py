"""Pure helpers backing `AdaptiveSearchService`'s state transitions."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Sequence

from .algorithms import atomic_regions, prioritize_videos, select_refinement_frontier
from .exceptions import AdaptiveInputError, RevisionConflictError
from .proposal_profiles import generate_profiled_proposals
from .schemas import EventDefinition, SearchConstraints, TemporalRegion
from .session import SearchRun, SessionBundle, utc_now


def _clear_fixed_proposal_status(bundle: SessionBundle, event_id: str) -> None:
    bundle.artifacts.proposals = [
        item.model_copy(update={"user_status": "active"})
        if item.event_id == event_id and item.user_status == "fixed"
        else item
        for item in bundle.artifacts.proposals
    ]


def _event_index(bundle: SessionBundle, event_id: str) -> int:
    for index, event in enumerate(bundle.session.events):
        if event.event_id == event_id:
            return index
    raise AdaptiveInputError(f"unknown event_id: {event_id}")


def _require_revision(bundle: SessionBundle, expected_revision: int) -> None:
    actual = bundle.session.revision
    if actual != expected_revision:
        raise RevisionConflictError(expected_revision, actual)


def _validate_constraints_for_events(
    events: list[EventDefinition],
    constraints: SearchConstraints,
) -> None:
    event_order = [event.event_id for event in events]
    known = set(event_order)
    unknown_events = set(constraints.event_constraints) - known
    if unknown_events:
        raise AdaptiveInputError(
            f"constraints reference unknown events: {sorted(unknown_events)}"
        )
    event_index = {event_id: index for index, event_id in enumerate(event_order)}
    for gap in constraints.adjacent_gap_constraints:
        unknown_gap_events = {gap.before_event_id, gap.after_event_id} - known
        if unknown_gap_events:
            raise AdaptiveInputError(
                f"gap constraint references unknown events: {sorted(unknown_gap_events)}"
            )
        if event_index[gap.after_event_id] != event_index[gap.before_event_id] + 1:
            raise AdaptiveInputError(
                "gap constraints must reference adjacent events in forward order"
            )


def _validate_event_subset(bundle: SessionBundle, event_ids: set[str]) -> None:
    if not event_ids:
        raise AdaptiveInputError("event_ids must not be empty")
    known = {event.event_id for event in bundle.session.events}
    missing = event_ids - known
    if missing:
        raise AdaptiveInputError(f"unknown event_ids: {sorted(missing)}")


def _validate_retrieval_variants(
    events: list[EventDefinition],
    variants: dict[str, list[str]],
) -> None:
    known = {event.event_id for event in events}
    unknown_events = set(variants) - known
    if unknown_events:
        raise AdaptiveInputError(
            f"retrieval_variants reference unknown events: {sorted(unknown_events)}"
        )
    for event_id, texts in variants.items():
        if not texts:
            raise AdaptiveInputError(
                f"retrieval_variants for {event_id!r} must not be empty"
            )
        if any(not text.strip() for text in texts):
            raise AdaptiveInputError(
                f"retrieval_variants for {event_id!r} must not contain empty text"
            )


def _ensure_unique_ids(ids: Iterable[str], label: str) -> None:
    values = list(ids)
    if len(values) != len(set(values)):
        raise AdaptiveInputError(f"{label} ids must be unique")


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if key not in target:
            # Pydantic will provide the public 422/error detail for unknown keys.
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _invalidate_artifacts(
    bundle: SessionBundle,
    stages: list[str],
    event_ids: set[str] | None = None,
) -> None:
    def keep_event(item: Any) -> bool:
        return event_ids is not None and item.event_id not in event_ids

    if "retrieval" in stages:
        bundle.artifacts.candidates = [
            item for item in bundle.artifacts.candidates if keep_event(item)
        ]
    if "region" in stages:
        removed_region_ids = {
            item.id for item in bundle.artifacts.regions if not keep_event(item)
        }
        bundle.artifacts.regions = [
            item for item in bundle.artifacts.regions if keep_event(item)
        ]
        bundle.artifacts.frontier_region_ids = [
            region_id
            for region_id in bundle.artifacts.frontier_region_ids
            if region_id not in removed_region_ids
        ]
    if "frontier" in stages:
        bundle.artifacts.frontier_region_ids = []
    if "refinement" in stages:
        bundle.artifacts.frame_scores = [
            item for item in bundle.artifacts.frame_scores if keep_event(item)
        ]
    if "proposal" in stages:
        bundle.artifacts.proposals = [
            item for item in bundle.artifacts.proposals if keep_event(item)
        ]


def _rebuild_frontier(bundle: SessionBundle) -> None:
    event_order = [event.event_id for event in bundle.session.events]
    priorities = prioritize_videos(
        bundle.artifacts.regions,
        event_order,
        bundle.session.hyperparameters.refinement,
        bundle.session.constraints,
    )
    frontier = select_refinement_frontier(
        bundle.artifacts.regions,
        priorities,
        event_order,
        bundle.session.hyperparameters.refinement,
        bundle.session.constraints,
    )
    bundle.artifacts.frontier_region_ids = [item.id for item in frontier]


def _rebuild_reusable_artifacts(
    bundle: SessionBundle,
    invalidated: list[str],
) -> None:
    # "region" is a pure, local function of bundle.artifacts.candidates - no
    # upstream call needed, unlike "retrieval" - so it can and should be
    # rebuilt right here instead of left empty until a fresh
    # commands/retrieve. Without this, a clustering.* patch (mapped to
    # "region") invalidated regions and never rebuilt them: 200 OK, but
    # every subsequent /video-priorities, /regions, and .../frame-scores
    # call broke until the caller manually re-ran retrieve from scratch.
    if "region" in invalidated:
        bundle.artifacts.regions = atomic_regions(bundle.artifacts.candidates)
    if "proposal" in invalidated and "refinement" not in invalidated:
        bundle.artifacts.proposals = generate_profiled_proposals(
            bundle.session.events,
            bundle.artifacts.frame_scores,
            bundle.session.hyperparameters.boundary,
        )


def _frame_score_acceptance_window(
    region: TemporalRegion,
    all_regions: Sequence[TemporalRegion],
    margin_seconds: float,
) -> tuple[float, float]:
    """The timestamp window a `replace_frame_scores` sample must fall within
    to count as evidence for `region`'s event.

    Anchored on the region's own representative timestamp (span midpoint -
    exact for a zero-width region, the production default; a reasonable
    center for a wider one) and padded by `margin_seconds` on each side, but
    never past the midpoint to the nearest region belonging to a *different*
    event in the same video. Without that cap, a wide margin could accept a
    frame that legitimately belongs to a neighboring event as if it were
    evidence for this one - not a hypothetical: 25% of top-ranked videos on
    the real n=60 corpus have two different events landing on the same or
    near-same timestamp (region_tuple_ranking_results.md). This is the only
    place region width/margin is computed in the pipeline - ranking and
    refinement never read it (see `algorithms/regions.py::atomic_regions`)."""

    anchor = (region.start_seconds + region.end_seconds) / 2.0
    effective_margin = margin_seconds
    for other in all_regions:
        if other.video_id != region.video_id or other.event_id == region.event_id:
            continue
        other_anchor = (other.start_seconds + other.end_seconds) / 2.0
        effective_margin = min(effective_margin, abs(other_anchor - anchor) / 2.0)
    return (anchor - effective_margin, anchor + effective_margin)


def _complete_run(
    run: SearchRun,
    started: float,
    **metrics: int | float | str | bool | None,
) -> SearchRun:
    return run.model_copy(
        update={
            "status": "completed",
            "metrics": {
                **metrics,
                "elapsed_ms": (perf_counter() - started) * 1000.0,
            },
            "completed_at": utc_now(),
        }
    )
