"""Pure helpers backing `AdaptiveSearchService`'s state transitions."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable

from .algorithms import atomic_regions
from .exceptions import AdaptiveInputError, RevisionConflictError
from .schemas import EventConstraint, EventDefinition, SearchConstraints
from .session import SearchRun, SessionBundle, utc_now


def _event_index(bundle: SessionBundle, event_id: str) -> int:
    for index, event in enumerate(bundle.session.events):
        if event.event_id == event_id:
            return index
    raise AdaptiveInputError(f"unknown event_id: {event_id}")


def _apply_fix_frame(
    bundle: SessionBundle,
    *,
    session_id: str,
    event_id: str,
    video_id: str,
    frame_id: int,
    timestamp_seconds: float,
    region_id: str | None,
) -> None:
    """Pins one event+video's exact frame - the mutation core of
    AdaptiveSearchService.fix_frame()."""

    _event_index(bundle, event_id)
    constraints_payload = bundle.session.constraints.model_dump()
    event_constraints = dict(constraints_payload["event_constraints"])
    current = event_constraints.get(event_id, EventConstraint().model_dump())
    current.update(
        {
            "fixed_video_id": video_id,
            "fixed_region_id": region_id or f"user:{event_id}:{video_id}",
            "fixed_frame_id": frame_id,
            "fixed_timestamp_seconds": timestamp_seconds,
        }
    )
    event_constraints[event_id] = current
    constraints_payload["event_constraints"] = event_constraints
    bundle.session.constraints = SearchConstraints.model_validate(constraints_payload)


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


def _validate_event_relation_graph(events: list[EventDefinition]) -> None:
    """Reject a session whose events' `temporal_relation`/`reference_event_id`
    edges contain a dangling reference or a cycle.

    `build_order_constraints` (tuple_ranking.py) builds the exact same edge
    set from these two fields and silently drops a dangling reference (no
    edge, no error) - left unvalidated, a typo'd `reference_event_id` just
    quietly loses its ordering constraint instead of failing loudly at
    session creation, the only place it can still be corrected. A cycle is
    worse: `build_order_constraints`'s transitive closure turns e.g. a
    two-event mutual "before" cycle into self-edges `(0,0)` and `(1,1)`,
    and `_order_score` can never satisfy `timestamps[i] > timestamps[i]` -
    every video's order score for that event permanently loses points with
    no way to fix it short of deleting the session, since nothing else in
    the pipeline can express or repair a self-precedence constraint."""

    known = {event.event_id for event in events}
    edges: list[tuple[str, str]] = []
    for event in events:
        reference = event.reference_event_id
        if reference is None:
            continue
        if reference not in known:
            raise AdaptiveInputError(
                f"event {event.event_id!r} references unknown "
                f"reference_event_id {reference!r}"
            )
        if event.temporal_relation == "after":
            edges.append((reference, event.event_id))
        elif event.temporal_relation == "before":
            edges.append((event.event_id, reference))
        # "during" / "simultaneous": no edge - matches build_order_constraints.

    adjacency: dict[str, list[str]] = {event_id: [] for event_id in known}
    for predecessor, successor in edges:
        adjacency[predecessor].append(successor)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {event_id: WHITE for event_id in known}

    def visit(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for neighbor in adjacency[node]:
            if color[neighbor] == GRAY:
                cycle = " -> ".join([*path, neighbor])
                raise AdaptiveInputError(
                    f"event temporal relations contain a cycle: {cycle}"
                )
            if color[neighbor] == WHITE:
                visit(neighbor, [*path, neighbor])
        color[node] = BLACK

    for event_id in known:
        if color[event_id] == WHITE:
            visit(event_id, [event_id])


def _validate_event_subset(bundle: SessionBundle, event_ids: set[str]) -> None:
    if not event_ids:
        raise AdaptiveInputError("event_ids must not be empty")
    known = {event.event_id for event in bundle.session.events}
    missing = event_ids - known
    if missing:
        raise AdaptiveInputError(f"unknown event_ids: {sorted(missing)}")


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
        bundle.artifacts.regions = [
            item for item in bundle.artifacts.regions if keep_event(item)
        ]


def _rebuild_reusable_artifacts(
    bundle: SessionBundle,
    invalidated: list[str],
) -> None:
    # "region" is a pure, local function of bundle.artifacts.candidates - no
    # upstream call needed, unlike "retrieval" - so it can and should be
    # rebuilt right here instead of left empty until a fresh
    # commands/retrieve. Without this, a patch that invalidated regions
    # never rebuilt them: 200 OK, but every subsequent /video-priorities and
    # /regions call broke until the caller manually re-ran retrieve from
    # scratch.
    if "region" in invalidated:
        bundle.artifacts.regions = atomic_regions(bundle.artifacts.candidates)


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
