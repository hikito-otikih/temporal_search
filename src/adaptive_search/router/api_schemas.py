"""Pydantic request/response models for the adaptive-search API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import (
    BoundaryType,
    EventDefinition,
    FrameScoreSample,
    SearchHyperparameters,
    SearchConstraints,
    SparseCandidate,
)
from ..session import SearchSession


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CreateSessionRequest(ApiModel):
    searcher_type: Literal["adaptive_temporal"] = "adaptive_temporal"
    common_query: str | None = Field(default=None, max_length=8000)
    events: list[EventDefinition] = Field(min_length=1, max_length=32)
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    hyperparameters: SearchHyperparameters = Field(
        default_factory=SearchHyperparameters
    )


class EventPatch(ApiModel):
    original_query: str | None = Field(default=None, min_length=1)
    anchor_query: str | None = Field(default=None, min_length=1)
    pre_state: str | None = Field(default=None, min_length=1)
    post_state: str | None = Field(default=None, min_length=1)
    boundary_type: BoundaryType | None = None

    @model_validator(mode="after")
    def require_a_change(self):
        if not self.model_fields_set:
            raise ValueError("event patch must contain at least one field")
        return self


class PatchEventRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    patch: EventPatch


class PatchHyperparametersRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    patch: dict[str, Any]

    @model_validator(mode="after")
    def require_a_change(self):
        if not self.patch:
            raise ValueError("hyperparameter patch must not be empty")
        return self


class ReplaceConstraintsRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    constraints: SearchConstraints


class CandidateIngestRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    event_ids: list[str] = Field(min_length=1)
    candidates: list[SparseCandidate]


class FrameScoreIngestRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    region_ids: list[str] = Field(min_length=1)
    samples: list[FrameScoreSample] = Field(min_length=1)


class RetrieveSessionRequest(ApiModel):
    top_k: int = Field(default=20, ge=1, le=10_000)
    event_ids: list[str] | None = Field(default=None, min_length=1)


class MarkVideosRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    video_ids: list[str]


class FixFrameRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    event_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    region_id: str | None = Field(default=None, min_length=1)


class RejectProposalRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    event_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)


class ClearEventConstraintRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    event_id: str = Field(min_length=1)


class ArtifactCounts(ApiModel):
    candidates: int
    regions: int
    frontier_regions: int
    frame_scores: int
    proposals: int
    runs: int


class SessionResponse(ApiModel):
    session: SearchSession
    artifact_counts: ArtifactCounts
    live_refinement: dict[str, Any]


class MutationResponse(ApiModel):
    session_revision: int
    invalidated_stages: list[str]
    artifact_counts: ArtifactCounts


class RunResponse(ApiModel):
    session_revision: int
    run_id: str
    run_status: str
    metrics: dict[str, Any]
    artifact_counts: ArtifactCounts
