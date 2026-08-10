"""Single-process session state for the adaptive research vertical slice."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import RevisionConflictError, SessionNotFoundError
from .schemas import (
    EventDefinition,
    EventProposal,
    FrameScoreSample,
    SearchConstraints,
    SearchHyperparameters,
    SparseCandidate,
    TemporalRegion,
    TupleResult,
)


SearcherType = Literal[
    "legacy_temporal",
    "legacy_ambiguous",
    "adaptive_temporal",
]
SessionStatus = Literal["draft", "ready", "running", "completed", "failed"]
RunStatus = Literal["accepted", "running", "completed", "failed", "stale"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchSession(SessionModel):
    id: str
    revision: int = Field(default=0, ge=0)
    searcher_type: SearcherType = "adaptive_temporal"
    common_query: str | None = Field(default=None, max_length=8000)
    events: list[EventDefinition] = Field(min_length=1, max_length=32)
    retrieval_variants: dict[str, list[str]] = Field(default_factory=dict)
    hyperparameters: SearchHyperparameters = Field(
        default_factory=SearchHyperparameters
    )
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    status: SessionStatus = "draft"
    last_invalidated_stages: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_unique_event_ids(self):
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within a session")
        return self


class ArtifactState(SessionModel):
    revision: int = Field(default=0, ge=0)
    candidates: list[SparseCandidate] = Field(default_factory=list)
    regions: list[TemporalRegion] = Field(default_factory=list)
    frontier_region_ids: list[str] = Field(default_factory=list)
    frame_scores: list[FrameScoreSample] = Field(default_factory=list)
    proposals: list[EventProposal] = Field(default_factory=list)
    tuples: list[TupleResult] = Field(default_factory=list)


class SearchRun(SessionModel):
    id: str
    session_id: str
    session_revision: int = Field(ge=0)
    searcher_type: SearcherType
    status: RunStatus
    stages: list[str] = Field(default_factory=list)
    hyperparameter_snapshot: dict
    constraint_snapshot: dict
    metrics: dict[str, int | float | str | bool | None] = Field(
        default_factory=dict
    )
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class SessionBundle(SessionModel):
    session: SearchSession
    artifacts: ArtifactState = Field(default_factory=ArtifactState)
    runs: list[SearchRun] = Field(default_factory=list)


class InMemorySessionRepository:
    """Thread-safe MVP store with compare-and-swap revision semantics.

    This class is intentionally explicit about its single-process durability
    limit.  Its interface can later be implemented with SQLite or Postgres.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._bundles: dict[str, SessionBundle] = {}

    def create(
        self,
        *,
        events: list[EventDefinition],
        common_query: str | None,
        searcher_type: SearcherType,
        hyperparameters: SearchHyperparameters,
        constraints: SearchConstraints,
        retrieval_variants: dict[str, list[str]] | None = None,
    ) -> SessionBundle:
        session_id = str(uuid4())
        session = SearchSession(
            id=session_id,
            events=events,
            common_query=common_query,
            searcher_type=searcher_type,
            retrieval_variants=dict(retrieval_variants or {}),
            hyperparameters=hyperparameters,
            constraints=constraints,
        )
        bundle = SessionBundle(
            session=session,
            artifacts=ArtifactState(revision=session.revision),
        )
        with self._lock:
            self._bundles[session_id] = bundle
        return bundle.model_copy(deep=True)

    def get(self, session_id: str) -> SessionBundle:
        with self._lock:
            bundle = self._bundles.get(session_id)
            if bundle is None:
                raise SessionNotFoundError(session_id)
            return bundle.model_copy(deep=True)

    def mutate(
        self,
        session_id: str,
        expected_revision: int,
        mutator: Callable[[SessionBundle], None],
    ) -> SessionBundle:
        """Apply a user mutation and increment the session revision once."""

        with self._lock:
            current = self._require(session_id)
            self._check_revision(current, expected_revision)
            working = current.model_copy(deep=True)
            mutator(working)
            working.session.revision += 1
            working.session.updated_at = utc_now()
            working.artifacts.revision = working.session.revision
            validated = SessionBundle.model_validate(working.model_dump())
            self._bundles[session_id] = validated
            return validated.model_copy(deep=True)

    def commit_run(
        self,
        session_id: str,
        expected_revision: int,
        mutator: Callable[[SessionBundle], None],
    ) -> SessionBundle:
        """Commit artifacts only if the originating revision is still current."""

        with self._lock:
            current = self._require(session_id)
            self._check_revision(current, expected_revision)
            working = current.model_copy(deep=True)
            mutator(working)
            working.session.updated_at = utc_now()
            working.artifacts.revision = expected_revision
            validated = SessionBundle.model_validate(working.model_dump())
            self._bundles[session_id] = validated
            return validated.model_copy(deep=True)

    def delete(self, session_id: str, expected_revision: int) -> None:
        with self._lock:
            current = self._require(session_id)
            self._check_revision(current, expected_revision)
            del self._bundles[session_id]

    def clear(self) -> None:
        """Test helper; production code should use scoped deletion/TTL."""

        with self._lock:
            self._bundles.clear()

    def _require(self, session_id: str) -> SessionBundle:
        bundle = self._bundles.get(session_id)
        if bundle is None:
            raise SessionNotFoundError(session_id)
        return bundle

    @staticmethod
    def _check_revision(bundle: SessionBundle, expected_revision: int) -> None:
        actual = bundle.session.revision
        if expected_revision != actual:
            raise RevisionConflictError(expected_revision, actual)


def new_run(bundle: SessionBundle, stages: list[str]) -> SearchRun:
    return SearchRun(
        id=str(uuid4()),
        session_id=bundle.session.id,
        session_revision=bundle.session.revision,
        searcher_type=bundle.session.searcher_type,
        status="running",
        stages=stages,
        hyperparameter_snapshot=bundle.session.hyperparameters.model_dump(mode="json"),
        constraint_snapshot=bundle.session.constraints.model_dump(mode="json"),
    )
