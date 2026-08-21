"""Application service for the synchronous adaptive-search vertical slice."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable

from .algorithms import atomic_regions
from .exceptions import AdaptiveInputError
from .invalidation import (
    changed_paths,
    invalidated_for_constraint,
    invalidated_for_event,
    invalidated_for_parameters,
)
from .schemas import (
    EventDefinition,
    FixFrameEntry,
    SearchConstraints,
    SearchHyperparameters,
    SparseCandidate,
)
from .providers import FrameProvider, UnavailableFrameProvider
from .retrieval import fuse_candidates_rrf
from .service_helpers import (
    _apply_fix_frame,
    _clear_fixed_proposal_status,
    _complete_run,
    _deep_merge,
    _ensure_unique_ids,
    _event_index,
    _invalidate_artifacts,
    _rebuild_reusable_artifacts,
    _require_revision,
    _validate_constraints_for_events,
    _validate_event_relation_graph,
    _validate_event_subset,
    _validate_retrieval_variants,
)
from .session import (
    InMemorySessionRepository,
    SearchRun,
    SearcherType,
    SessionBundle,
    new_run,
)


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
        _validate_event_relation_graph(events)
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
            # fix_frame() marks an EventProposal user_status="fixed" and
            # relies on clear_event_constraint() to flip it back via
            # _clear_fixed_proposal_status() - but this generic replace
            # bypasses both of those command endpoints, so a caller that
            # clears or *replaces* an event's pin via a raw PUT here (rather
            # than commands/fix-frame or commands/clear-constraint) left
            # GET /proposals reporting a stale "fixed" status for a proposal
            # the constraint no longer names - not just on full removal:
            # replacing video A's pin with video B's (or the same video with
            # a different frame/timestamp) via raw PUT leaves video A's
            # proposal marked fixed too, since it compared only "is there
            # still some pin" rather than "is it still the same pin".
            # Comparing every identifying field catches both cases; this
            # module has no way to *re*-mark a new match as fixed the way
            # fix_frame() does (a raw PUT carries no separate video_id/
            # frame_id/timestamp_seconds to correlate against
            # bundle.artifacts.proposals), so it only ever clears stale
            # status, never sets new.
            previous_event_constraints = bundle.session.constraints.event_constraints
            for event_id, previous in previous_event_constraints.items():
                if previous.fixed_video_id is None:
                    continue
                updated = constraints.event_constraints.get(event_id)
                pin_unchanged = updated is not None and (
                    updated.fixed_video_id,
                    updated.fixed_region_id,
                    updated.fixed_frame_id,
                    updated.fixed_timestamp_seconds,
                ) == (
                    previous.fixed_video_id,
                    previous.fixed_region_id,
                    previous.fixed_frame_id,
                    previous.fixed_timestamp_seconds,
                )
                if not pin_unchanged:
                    _clear_fixed_proposal_status(bundle, event_id)
            bundle.session.constraints = constraints.model_copy(deep=True)
            bundle.session.last_invalidated_stages = invalidated

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

        run = new_run(snapshot, ["retrieval", "region"])
        started = perf_counter()

        def commit(bundle: SessionBundle) -> None:
            retained = [
                item
                for item in bundle.artifacts.candidates
                if item.event_id not in replaced_events
            ]
            bundle.artifacts.candidates = [*retained, *fused_candidates]
            # atomic_regions, not cluster_temporal_regions - see that
            # function's docstring (algorithms/regions.py) for why merging
            # candidates was dropped. Zero-width (no margin) - ranking and
            # live boundary refinement never read region span.
            bundle.artifacts.regions = atomic_regions(bundle.artifacts.candidates)
            bundle.artifacts.proposals = [
                item
                for item in bundle.artifacts.proposals
                if item.event_id not in replaced_events
            ]
            bundle.session.status = "ready"
            bundle.session.last_invalidated_stages = [
                "retrieval",
                "region",
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
            )
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
            _apply_fix_frame(
                bundle,
                session_id=session_id,
                event_id=event_id,
                video_id=video_id,
                frame_id=frame_id,
                timestamp_seconds=timestamp_seconds,
                region_id=region_id,
            )
            bundle.session.last_invalidated_stages = invalidated

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )

    def fix_frames(
        self,
        session_id: str,
        *,
        expected_revision: int,
        fixes: list[FixFrameEntry],
    ) -> tuple[SessionBundle, list[str]]:
        """Batch form of fix_frame() - applies every item inside one mutation,
        so N pins cost one expected_revision check and one revision bump
        instead of N round trips each racing the next against a concurrent
        caller. Items are applied in order; a later item pinning the same
        event_id as an earlier one simply overrides it, matching what N
        sequential fix_frame() calls would do."""

        invalidated = invalidated_for_constraint("fix_frame")

        def mutate(bundle: SessionBundle) -> None:
            for fix in fixes:
                _apply_fix_frame(
                    bundle,
                    session_id=session_id,
                    event_id=fix.event_id,
                    video_id=fix.video_id,
                    frame_id=fix.frame_id,
                    timestamp_seconds=fix.timestamp_seconds,
                    region_id=fix.region_id,
                )
            bundle.session.last_invalidated_stages = invalidated

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

        return (
            self.repository.mutate(session_id, expected_revision, mutate),
            invalidated,
        )


