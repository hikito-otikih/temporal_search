"""Application service for the synchronous adaptive-search vertical slice."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable

from .algorithms import (
    assemble_ordered_tuples,
    calibrate_frame_scores,
    cluster_temporal_regions,
    prioritize_videos,
    select_refinement_frontier,
)
from .invalidation import (
    changed_paths,
    invalidated_for_constraint,
    invalidated_for_event,
    invalidated_for_parameters,
)
from .models import (
    EventConstraint,
    EventDefinition,
    EventProposal,
    FrameScoreSample,
    SearchConstraints,
    SearchHyperparameters,
    SparseCandidate,
)
from .providers import FrameProvider, UnavailableFrameProvider
from .proposal_profiles import generate_profiled_proposals
from .retrieval import fuse_candidates_rrf
from .session import (
    InMemorySessionRepository,
    SearchRun,
    SearcherType,
    SessionBundle,
    new_run,
    utc_now,
)


class AdaptiveInputError(ValueError):
    pass


class AdaptiveSearchService:
    """Coordinate state transitions while algorithms remain pure and testable."""

    def __init__(
        self,
        repository: InMemorySessionRepository | None = None,
        frame_provider: FrameProvider | None = None,
    ) -> None:
        self.repository = repository or InMemorySessionRepository()
        self.frame_provider = frame_provider or UnavailableFrameProvider()

    def create_session(
        self,
        *,
        events: list[EventDefinition],
        common_query: str | None = None,
        searcher_type: SearcherType = "adaptive_temporal",
        hyperparameters: SearchHyperparameters | None = None,
        constraints: SearchConstraints | None = None,
        retrieval_variants: dict[str, list[str]] | None = None,
    ) -> SessionBundle:
        selected_constraints = constraints or SearchConstraints()
        _validate_constraints_for_events(events, selected_constraints)
        selected_variants = dict(retrieval_variants or {})
        _validate_retrieval_variants(events, selected_variants)
        return self.repository.create(
            events=events,
            common_query=common_query,
            searcher_type=searcher_type,
            hyperparameters=hyperparameters or SearchHyperparameters(),
            constraints=selected_constraints,
            retrieval_variants=selected_variants,
        )

    def get_session(self, session_id: str) -> SessionBundle:
        return self.repository.get(session_id)

    def delete_session(self, session_id: str, expected_revision: int) -> None:
        self.repository.delete(session_id, expected_revision)

    def patch_event(
        self,
        session_id: str,
        event_id: str,
        *,
        expected_revision: int,
        patch: dict[str, Any],
    ) -> tuple[SessionBundle, list[str]]:
        snapshot = self.repository.get(session_id)
        _require_revision(snapshot, expected_revision)
        current = snapshot.session.events[_event_index(snapshot, event_id)]
        payload = current.model_dump()
        payload.update(patch)
        payload["event_id"] = event_id
        replacement = EventDefinition.model_validate(payload)
        if not changed_paths(current, replacement):
            return snapshot, []

        invalidated: list[str] = []

        def mutate(bundle: SessionBundle) -> None:
            nonlocal invalidated
            index = _event_index(bundle, event_id)
            current = bundle.session.events[index]
            payload = current.model_dump()
            payload.update(patch)
            payload["event_id"] = event_id
            replacement = EventDefinition.model_validate(payload)
            paths = changed_paths(current, replacement)
            invalidated = invalidated_for_event(paths)
            if not paths:
                return
            bundle.session.events[index] = replacement
            bundle.session.last_invalidated_stages = invalidated
            _invalidate_artifacts(bundle, invalidated, event_ids={event_id})
            _rebuild_reusable_artifacts(bundle, invalidated)

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )

    def patch_hyperparameters(
        self,
        session_id: str,
        *,
        expected_revision: int,
        patch: dict[str, Any],
    ) -> tuple[SessionBundle, list[str], set[str]]:
        snapshot = self.repository.get(session_id)
        _require_revision(snapshot, expected_revision)
        preview_payload = snapshot.session.hyperparameters.model_dump()
        _deep_merge(preview_payload, patch)
        preview = SearchHyperparameters.model_validate(preview_payload)
        if not changed_paths(snapshot.session.hyperparameters, preview):
            return snapshot, [], set()

        invalidated: list[str] = []
        parameter_paths: set[str] = set()

        def mutate(bundle: SessionBundle) -> None:
            nonlocal invalidated, parameter_paths
            current = bundle.session.hyperparameters
            payload = current.model_dump()
            _deep_merge(payload, patch)
            replacement = SearchHyperparameters.model_validate(payload)
            parameter_paths = changed_paths(current, replacement)
            invalidated = invalidated_for_parameters(parameter_paths)
            if not parameter_paths:
                return
            bundle.session.hyperparameters = replacement
            bundle.session.last_invalidated_stages = invalidated
            _invalidate_artifacts(bundle, invalidated)
            _rebuild_reusable_artifacts(bundle, invalidated)

        bundle = self.repository.mutate(session_id, expected_revision, mutate)
        return bundle, invalidated, parameter_paths

    def replace_constraints(
        self,
        session_id: str,
        *,
        expected_revision: int,
        constraints: SearchConstraints,
    ) -> tuple[SessionBundle, list[str]]:
        snapshot = self.repository.get(session_id)
        _require_revision(snapshot, expected_revision)
        _validate_constraints_for_events(snapshot.session.events, constraints)
        paths = changed_paths(snapshot.session.constraints, constraints)
        if not paths:
            return snapshot, []

        root_paths = {path.split(".", 1)[0] for path in paths}
        video_scope_changed = bool(
            root_paths & {"allowed_video_ids", "rejected_video_ids"}
        )
        event_scope_changed = "event_constraints" in root_paths
        if video_scope_changed:
            invalidated = invalidated_for_constraint("mark_videos")
        elif event_scope_changed:
            invalidated = ["proposal", "tuple"]
        else:
            invalidated = ["tuple"]

        def mutate(bundle: SessionBundle) -> None:
            _validate_constraints_for_events(bundle.session.events, constraints)
            bundle.session.constraints = constraints.model_copy(deep=True)
            bundle.session.last_invalidated_stages = invalidated
            if video_scope_changed:
                _rebuild_frontier(bundle)
            _rebuild_tuples(bundle)

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )

    def replace_candidates(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_ids: Iterable[str],
        candidates: list[SparseCandidate],
    ) -> tuple[SessionBundle, SearchRun]:
        snapshot = self.repository.get(session_id)
        _require_revision(snapshot, expected_revision)
        replaced_events = set(event_ids)
        _validate_event_subset(snapshot, replaced_events)
        for candidate in candidates:
            if candidate.session_id != session_id:
                raise AdaptiveInputError("candidate session_id does not match URL")
            if candidate.event_id not in replaced_events:
                raise AdaptiveInputError(
                    "candidate event_id must be listed in replaced event_ids"
                )
        _ensure_unique_ids((candidate.id for candidate in candidates), "candidate")
        fused_candidates = fuse_candidates_rrf(
            candidates,
            snapshot.session.hyperparameters.retrieval,
        )

        run = new_run(snapshot, ["retrieval", "region", "frontier"])
        started = perf_counter()

        def commit(bundle: SessionBundle) -> None:
            retained = [
                item
                for item in bundle.artifacts.candidates
                if item.event_id not in replaced_events
            ]
            bundle.artifacts.candidates = [*retained, *fused_candidates]
            bundle.artifacts.regions = cluster_temporal_regions(
                bundle.artifacts.candidates,
                bundle.session.hyperparameters.clustering,
            )
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
            bundle.artifacts.frame_scores = [
                item
                for item in bundle.artifacts.frame_scores
                if item.event_id not in replaced_events
            ]
            bundle.artifacts.proposals = [
                item
                for item in bundle.artifacts.proposals
                if item.event_id not in replaced_events
            ]
            bundle.artifacts.tuples = []
            bundle.session.status = "ready"
            bundle.session.last_invalidated_stages = [
                "retrieval",
                "region",
                "frontier",
                "refinement",
                "proposal",
                "tuple",
            ]
            completed = _complete_run(
                run,
                started,
                input_candidates=len(candidates),
                fused_candidates=len(bundle.artifacts.candidates),
                regions=len(bundle.artifacts.regions),
                frontier_regions=len(frontier),
            )
            bundle.runs.append(completed)

        result = self.repository.commit_run(session_id, expected_revision, commit)
        return result, result.runs[-1]

    def replace_frame_scores(
        self,
        session_id: str,
        *,
        expected_revision: int,
        region_ids: Iterable[str],
        samples: list[FrameScoreSample],
        run_metrics: dict[str, int | float | str | bool | None] | None = None,
    ) -> tuple[SessionBundle, SearchRun]:
        snapshot = self.repository.get(session_id)
        _require_revision(snapshot, expected_revision)
        replaced_regions = set(region_ids)
        if not replaced_regions:
            raise AdaptiveInputError("region_ids must not be empty")
        known_regions = {region.id: region for region in snapshot.artifacts.regions}
        missing = replaced_regions - set(known_regions)
        if missing:
            raise AdaptiveInputError(f"unknown region_ids: {sorted(missing)}")
        sampled_regions: set[str] = set()
        for sample in samples:
            if sample.session_id != session_id:
                raise AdaptiveInputError("frame score session_id does not match URL")
            if sample.region_id not in replaced_regions:
                raise AdaptiveInputError(
                    "frame score region_id must be listed in replaced region_ids"
                )
            region = known_regions[sample.region_id]
            if (
                sample.event_id != region.event_id
                or sample.video_id != region.video_id
            ):
                raise AdaptiveInputError(
                    "frame score event/video must match its temporal region"
                )
            if not (region.start_seconds <= sample.timestamp_seconds <= region.end_seconds):
                raise AdaptiveInputError("frame score timestamp lies outside its region")
            sampled_regions.add(sample.region_id)
        regions_without_samples = replaced_regions - sampled_regions
        if regions_without_samples:
            raise AdaptiveInputError(
                "each replaced region must have at least one frame score sample: "
                f"{sorted(regions_without_samples)}"
            )

        run = new_run(snapshot, ["refinement", "proposal", "tuple"])
        started = perf_counter()
        supplied_run_metrics = dict(run_metrics or {})

        def commit(bundle: SessionBundle) -> None:
            retained = [
                item
                for item in bundle.artifacts.frame_scores
                if item.region_id not in replaced_regions
            ]
            calibrated = calibrate_frame_scores(
                samples,
                bundle.session.hyperparameters.boundary,
            )
            bundle.artifacts.frame_scores = [*retained, *calibrated]
            bundle.artifacts.regions = [
                region.model_copy(update={"refinement_status": "completed"})
                if region.id in replaced_regions
                else region
                for region in bundle.artifacts.regions
            ]
            bundle.artifacts.proposals = generate_profiled_proposals(
                bundle.session.events,
                bundle.artifacts.frame_scores,
                bundle.session.hyperparameters.boundary,
            )
            _rebuild_tuples(bundle)
            bundle.session.status = "completed"
            bundle.session.last_invalidated_stages = [
                "refinement",
                "proposal",
                "tuple",
            ]
            completed_metrics = {
                **supplied_run_metrics,
                "frame_scores": len(bundle.artifacts.frame_scores),
                "proposals": len(bundle.artifacts.proposals),
                "tuples": len(bundle.artifacts.tuples),
            }
            completed = _complete_run(run, started, **completed_metrics)
            bundle.runs.append(completed)

        result = self.repository.commit_run(session_id, expected_revision, commit)
        return result, result.runs[-1]

    def set_allowed_videos(
        self,
        session_id: str,
        *,
        expected_revision: int,
        video_ids: Iterable[str],
    ) -> tuple[SessionBundle, list[str]]:
        allowed = frozenset(video_ids)
        if any(not video_id.strip() for video_id in allowed):
            raise AdaptiveInputError("video_ids must be non-empty")
        invalidated = invalidated_for_constraint("mark_videos")

        def mutate(bundle: SessionBundle) -> None:
            payload = bundle.session.constraints.model_dump()
            payload["allowed_video_ids"] = allowed
            bundle.session.constraints = SearchConstraints.model_validate(payload)
            bundle.session.last_invalidated_stages = invalidated
            # Existing proposals are reusable.  A live provider may lazily refine
            # missing regions before this synchronous rebuild.
            priorities = prioritize_videos(
                bundle.artifacts.regions,
                [event.event_id for event in bundle.session.events],
                bundle.session.hyperparameters.refinement,
                bundle.session.constraints,
            )
            frontier = select_refinement_frontier(
                bundle.artifacts.regions,
                priorities,
                [event.event_id for event in bundle.session.events],
                bundle.session.hyperparameters.refinement,
                bundle.session.constraints,
            )
            bundle.artifacts.frontier_region_ids = [item.id for item in frontier]
            _rebuild_tuples(bundle)

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )

    def fix_frame(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_id: str,
        video_id: str,
        frame_id: int,
        timestamp_seconds: float,
        region_id: str | None = None,
    ) -> tuple[SessionBundle, list[str]]:
        invalidated = invalidated_for_constraint("fix_frame")

        def mutate(bundle: SessionBundle) -> None:
            _event_index(bundle, event_id)
            _clear_fixed_proposal_status(bundle, event_id)
            matching = next(
                (
                    proposal
                    for proposal in bundle.artifacts.proposals
                    if proposal.event_id == event_id
                    and proposal.video_id == video_id
                    and proposal.frame_id == frame_id
                ),
                None,
            )
            if matching is None:
                matching = EventProposal(
                    id=f"user:{event_id}:{video_id}:{frame_id}",
                    session_id=session_id,
                    event_id=event_id,
                    video_id=video_id,
                    region_id=region_id or f"user:{event_id}:{video_id}",
                    timestamp_seconds=timestamp_seconds,
                    frame_id=frame_id,
                    raw_semantic_score=1.0,
                    normalized_semantic_score=1.0,
                    raw_boundary_score=1.0,
                    normalized_boundary_score=1.0,
                    normalized_motion_score=0.5,
                    pre_consistency_score=1.0,
                    post_persistence_score=1.0,
                    final_event_score=1.0,
                    source="user",
                    user_status="fixed",
                )
                bundle.artifacts.proposals.append(matching)
            else:
                replacement = matching.model_copy(update={"user_status": "fixed"})
                bundle.artifacts.proposals = [
                    replacement if item.id == matching.id else item
                    for item in bundle.artifacts.proposals
                ]
            constraints_payload = bundle.session.constraints.model_dump()
            event_constraints = dict(constraints_payload["event_constraints"])
            current = event_constraints.get(event_id, EventConstraint().model_dump())
            current.update(
                {
                    "fixed_video_id": video_id,
                    "fixed_region_id": region_id or matching.region_id,
                    "fixed_frame_id": frame_id,
                    "fixed_timestamp_seconds": timestamp_seconds,
                }
            )
            event_constraints[event_id] = current
            constraints_payload["event_constraints"] = event_constraints
            bundle.session.constraints = SearchConstraints.model_validate(
                constraints_payload
            )
            bundle.session.last_invalidated_stages = invalidated
            _rebuild_tuples(bundle)

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )

    def reject_proposal(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_id: str,
        proposal_id: str,
    ) -> tuple[SessionBundle, list[str]]:
        invalidated = invalidated_for_constraint("reject_proposal")

        def mutate(bundle: SessionBundle) -> None:
            _event_index(bundle, event_id)
            proposal = next(
                (
                    item
                    for item in bundle.artifacts.proposals
                    if item.id == proposal_id and item.event_id == event_id
                ),
                None,
            )
            if proposal is None:
                raise AdaptiveInputError("proposal does not exist for this event")
            constraints_payload = bundle.session.constraints.model_dump()
            event_constraints = dict(constraints_payload["event_constraints"])
            current = EventConstraint.model_validate(
                event_constraints.get(event_id, {})
            )
            current_payload = current.model_dump()
            current_payload["rejected_proposal_ids"] = frozenset(
                {*current.rejected_proposal_ids, proposal_id}
            )
            event_constraints[event_id] = current_payload
            constraints_payload["event_constraints"] = event_constraints
            bundle.session.constraints = SearchConstraints.model_validate(
                constraints_payload
            )
            bundle.session.last_invalidated_stages = invalidated
            _rebuild_tuples(bundle)

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )

    def clear_event_constraint(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_id: str,
    ) -> tuple[SessionBundle, list[str]]:
        invalidated = invalidated_for_constraint("clear_constraint")

        def mutate(bundle: SessionBundle) -> None:
            _event_index(bundle, event_id)
            _clear_fixed_proposal_status(bundle, event_id)
            payload = bundle.session.constraints.model_dump()
            event_constraints = dict(payload["event_constraints"])
            event_constraints.pop(event_id, None)
            payload["event_constraints"] = event_constraints
            bundle.session.constraints = SearchConstraints.model_validate(payload)
            bundle.session.last_invalidated_stages = invalidated
            _rebuild_tuples(bundle)

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )


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
        from .session import RevisionConflictError

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
    if "tuple" in stages:
        bundle.artifacts.tuples = []


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
    if "proposal" in invalidated and "refinement" not in invalidated:
        bundle.artifacts.proposals = generate_profiled_proposals(
            bundle.session.events,
            bundle.artifacts.frame_scores,
            bundle.session.hyperparameters.boundary,
        )
        _rebuild_tuples(bundle)
    elif invalidated == ["tuple"]:
        _rebuild_tuples(bundle)


def _rebuild_tuples(bundle: SessionBundle) -> None:
    bundle.artifacts.tuples = assemble_ordered_tuples(
        bundle.session.events,
        bundle.artifacts.proposals,
        bundle.session.constraints,
        bundle.session.hyperparameters.ranking,
    )


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
