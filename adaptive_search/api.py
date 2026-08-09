"""Versioned FastAPI router for the adaptive-search research vertical slice."""

from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rewrite_queries import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
)
from rewrite_schema import RewriteRequest

from .embedding import (
    DEFAULT_SIGLIP2_MODEL,
    QUALITY_QWEN_EMBEDDING_MODEL,
    QUALITY_QWEN_RERANKER_MODEL,
)
from .models import (
    BoundaryType,
    EventDefinition,
    FrameScoreSample,
    SearchHyperparameters,
    SearchConstraints,
    SparseCandidate,
)
from .refinement import (
    LiveRefinementConfigurationError,
    LiveRefinementDataError,
    LiveRefinementOrchestrator,
    LiveRefinementUnavailableError,
)
from .rewrite_bridge import rewrite_queries_and_build_plan
from .runtime import configure_from_environment
from .service import AdaptiveInputError, AdaptiveSearchService
from .session import (
    RevisionConflictError,
    SearchSession,
    SessionBundle,
    SessionNotFoundError,
)
from .upstream import (
    QueryVariant,
    UpstreamSearchClient,
    UpstreamSearchError,
)


router = APIRouter(prefix="/v1", tags=["adaptive temporal search"])
adaptive_service = AdaptiveSearchService()
live_refinement_orchestrator = LiveRefinementOrchestrator(adaptive_service)

upstream_search_client = UpstreamSearchClient(
    os.getenv("UPSTREAM_URL")
    or os.getenv("ADAPTIVE_UPSTREAM_URL")
    or os.getenv("UPSTREAM_SEARCH_URL")
    or "http://172.26.176.1:8000"
)


@router.on_event("startup")
def configure_live_refinement_runtime() -> None:
    configure_from_environment(adaptive_service, live_refinement_orchestrator)


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


class RefineSessionRequest(ApiModel):
    expected_revision: int = Field(ge=0)
    region_ids: list[str] | None = Field(default=None, min_length=1)
    max_frames: int | None = Field(default=None, gt=0)


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
    tuples: int
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


@router.get("/searchers")
def list_searchers():
    capability = live_refinement_orchestrator.capabilities()
    return {
        "searchers": [
            {
                "searcher_type": "legacy_temporal",
                "available": True,
                "session_api": False,
                "ordered_events": True,
                "endpoint": "/temporal-search",
                "legacy_request_value": "TemporalSearcher",
            },
            {
                "searcher_type": "legacy_ambiguous",
                "available": True,
                "session_api": False,
                "ordered_events": False,
                "endpoint": "/temporal-search",
                "legacy_request_value": "AmbiguousSearcher",
            },
            {
                "searcher_type": "adaptive_temporal",
                "available": True,
                "session_api": True,
                "ordered_events": True,
                "algorithm_core_available": True,
                "live_refinement_available": capability.available,
                "frame_provider": asdict(capability.provider),
                "live_refinement": asdict(capability),
                "runtime_model": capability.model_id or DEFAULT_SIGLIP2_MODEL,
                "runtime_model_spec": capability.runtime_model_spec,
                "quality_embedding_model": QUALITY_QWEN_EMBEDDING_MODEL,
                "quality_reranker_model": QUALITY_QWEN_RERANKER_MODEL,
                "supported_relations": ["ordered_sequence", "adjacent_gap"],
                "supported_boundary_profiles": [
                    "onset",
                    "offset",
                    "transition",
                    "state",
                    "plateau_start",
                    "symmetric_peak",
                    "unknown",
                ],
            },
        ]
    }


@router.post(
    "/search-sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_search_session(request: CreateSessionRequest):
    try:
        hyperparameters = _apply_runtime_embedding_default(request.hyperparameters)
        bundle = adaptive_service.create_session(
            events=request.events,
            common_query=request.common_query,
            searcher_type=request.searcher_type,
            hyperparameters=hyperparameters,
            constraints=request.constraints,
        )
        return _session_response(bundle)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/from-queries",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session_from_queries(request: RewriteRequest):
    try:
        plan = await rewrite_queries_and_build_plan(
            queries=request.query,
            common_query=request.common_query,
        )
        bundle = adaptive_service.create_session(
            events=list(plan.events),
            common_query=plan.common_query,
            retrieval_variants={
                event_id: list(texts)
                for event_id, texts in plan.retrieval_variants.items()
            },
            hyperparameters=_apply_runtime_embedding_default(
                SearchHyperparameters()
            ),
        )
        return _session_response(bundle)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/search-sessions/{session_id}", response_model=SessionResponse)
def get_search_session(session_id: str):
    try:
        return _session_response(adaptive_service.get_session(session_id))
    except Exception as exc:
        _raise_api_error(exc)


@router.delete(
    "/search-sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_search_session(
    session_id: str,
    expected_revision: int = Query(ge=0),
):
    try:
        adaptive_service.delete_session(session_id, expected_revision)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as exc:
        _raise_api_error(exc)


@router.patch(
    "/search-sessions/{session_id}/events/{event_id}",
    response_model=MutationResponse,
)
def patch_event(session_id: str, event_id: str, request: PatchEventRequest):
    try:
        bundle, invalidated = adaptive_service.patch_event(
            session_id,
            event_id,
            expected_revision=request.expected_revision,
            patch=request.patch.model_dump(exclude_unset=True),
        )
        return _mutation_response(bundle, invalidated)
    except Exception as exc:
        _raise_api_error(exc)


@router.patch(
    "/search-sessions/{session_id}/hyperparameters",
    response_model=MutationResponse,
)
def patch_hyperparameters(
    session_id: str,
    request: PatchHyperparametersRequest,
):
    try:
        bundle, invalidated, _ = adaptive_service.patch_hyperparameters(
            session_id,
            expected_revision=request.expected_revision,
            patch=request.patch,
        )
        return _mutation_response(bundle, invalidated)
    except Exception as exc:
        _raise_api_error(exc)


@router.put(
    "/search-sessions/{session_id}/constraints",
    response_model=MutationResponse,
)
def replace_constraints(session_id: str, request: ReplaceConstraintsRequest):
    try:
        bundle, invalidated = adaptive_service.replace_constraints(
            session_id,
            expected_revision=request.expected_revision,
            constraints=request.constraints,
        )
        return _mutation_response(bundle, invalidated)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/artifacts/candidates",
    response_model=RunResponse,
)
def ingest_candidates(session_id: str, request: CandidateIngestRequest):
    try:
        bundle, run = adaptive_service.replace_candidates(
            session_id,
            expected_revision=request.expected_revision,
            event_ids=request.event_ids,
            candidates=request.candidates,
        )
        return _run_response(bundle, run)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/artifacts/frame-scores",
    response_model=RunResponse,
)
def ingest_frame_scores(session_id: str, request: FrameScoreIngestRequest):
    try:
        bundle, run = adaptive_service.replace_frame_scores(
            session_id,
            expected_revision=request.expected_revision,
            region_ids=request.region_ids,
            samples=request.samples,
        )
        return _run_response(bundle, run)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/commands/refine",
    response_model=RunResponse,
)
def refine_session(session_id: str, request: RefineSessionRequest):
    try:
        result = live_refinement_orchestrator.refine_session(
            session_id,
            expected_revision=request.expected_revision,
            region_ids=request.region_ids,
            max_frames=request.max_frames,
        )
        return _run_response(
            result.bundle,
            result.run,
            metrics=result.metrics,
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/commands/retrieve",
    response_model=RunResponse,
)
def retrieve_session(session_id: str, request: RetrieveSessionRequest):
    try:
        bundle = adaptive_service.get_session(session_id)
        selected_events = request.event_ids or [
            event.event_id for event in bundle.session.events
        ]
        variants = _build_query_variants(bundle, selected_events)
        candidates = upstream_search_client.retrieve_candidates(
            session_id=session_id,
            variants=variants,
            top_k=request.top_k,
        )
        bundle, run = adaptive_service.replace_candidates(
            session_id,
            expected_revision=bundle.session.revision,
            event_ids=selected_events,
            candidates=candidates,
        )
        return _run_response(bundle, run)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/commands/mark-videos",
    response_model=MutationResponse,
)
def mark_videos(session_id: str, request: MarkVideosRequest):
    try:
        bundle, invalidated = adaptive_service.set_allowed_videos(
            session_id,
            expected_revision=request.expected_revision,
            video_ids=request.video_ids,
        )
        return _mutation_response(bundle, invalidated)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/commands/fix-frame",
    response_model=MutationResponse,
)
def fix_frame(session_id: str, request: FixFrameRequest):
    try:
        bundle, invalidated = adaptive_service.fix_frame(
            session_id,
            expected_revision=request.expected_revision,
            event_id=request.event_id,
            video_id=request.video_id,
            frame_id=request.frame_id,
            timestamp_seconds=request.timestamp_seconds,
            region_id=request.region_id,
        )
        return _mutation_response(bundle, invalidated)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/commands/reject-proposal",
    response_model=MutationResponse,
)
def reject_proposal(session_id: str, request: RejectProposalRequest):
    try:
        bundle, invalidated = adaptive_service.reject_proposal(
            session_id,
            expected_revision=request.expected_revision,
            event_id=request.event_id,
            proposal_id=request.proposal_id,
        )
        return _mutation_response(bundle, invalidated)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/search-sessions/{session_id}/commands/clear-event-constraint",
    response_model=MutationResponse,
)
def clear_event_constraint(
    session_id: str,
    request: ClearEventConstraintRequest,
):
    try:
        bundle, invalidated = adaptive_service.clear_event_constraint(
            session_id,
            expected_revision=request.expected_revision,
            event_id=request.event_id,
        )
        return _mutation_response(bundle, invalidated)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/search-sessions/{session_id}/regions")
def get_regions(
    session_id: str,
    event_id: str | None = None,
    video_id: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        bundle = adaptive_service.get_session(session_id)
        items = [
            item
            for item in bundle.artifacts.regions
            if (event_id is None or item.event_id == event_id)
            and (video_id is None or item.video_id == video_id)
        ]
        frontier = set(bundle.artifacts.frontier_region_ids)
        page = items[offset : offset + limit]
        return {
            "items": [
                {**item.model_dump(mode="json"), "selected_for_refinement": item.id in frontier}
                for item in page
            ],
            "total": len(items),
            "offset": offset,
            "limit": limit,
        }
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/search-sessions/{session_id}/proposals")
def get_proposals(
    session_id: str,
    event_id: str | None = None,
    video_id: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        bundle = adaptive_service.get_session(session_id)
        items = [
            item
            for item in bundle.artifacts.proposals
            if (event_id is None or item.event_id == event_id)
            and (video_id is None or item.video_id == video_id)
        ]
        return _page(items, offset, limit)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/search-sessions/{session_id}/frame-scores")
def get_frame_scores(
    session_id: str,
    event_id: str | None = None,
    video_id: str | None = None,
    region_id: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
):
    try:
        bundle = adaptive_service.get_session(session_id)
        items = [
            item
            for item in bundle.artifacts.frame_scores
            if (event_id is None or item.event_id == event_id)
            and (video_id is None or item.video_id == video_id)
            and (region_id is None or item.region_id == region_id)
        ]
        return _page(items, offset, limit)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/search-sessions/{session_id}/tuples")
def get_tuples(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
):
    try:
        bundle = adaptive_service.get_session(session_id)
        return _page(bundle.artifacts.tuples, offset, limit)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/search-sessions/{session_id}/runs")
def get_runs(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
):
    try:
        bundle = adaptive_service.get_session(session_id)
        return _page(bundle.runs, offset, limit)
    except Exception as exc:
        _raise_api_error(exc)


def _session_response(bundle: SessionBundle) -> SessionResponse:
    refinement = bundle.session.hyperparameters.refinement
    runtime_spec = (
        refinement.quality_embedding_model
        if refinement.quality_profile_enabled
        else refinement.embedding_model
    )
    return SessionResponse(
        session=bundle.session,
        artifact_counts=_counts(bundle),
        live_refinement=asdict(
            live_refinement_orchestrator.capabilities(runtime_spec)
        ),
    )


def _apply_runtime_embedding_default(
    hyperparameters: SearchHyperparameters,
) -> SearchHyperparameters:
    if "embedding_model" in hyperparameters.refinement.model_fields_set:
        return hyperparameters
    runtime_spec = live_refinement_orchestrator.configured_runtime_spec()
    if runtime_spec is None:
        return hyperparameters
    return hyperparameters.model_copy(
        update={
            "refinement": hyperparameters.refinement.model_copy(
                update={"embedding_model": runtime_spec}
            )
        }
    )


def _build_query_variants(
    bundle: SessionBundle,
    event_ids: list[str],
) -> list[QueryVariant]:
    from .rewrite_bridge import variant_texts_for_events

    known = {event.event_id for event in bundle.session.events}
    if any(event_id not in known for event_id in event_ids):
        raise AdaptiveInputError(
            f"unknown event_ids: {sorted(set(event_ids) - known)}"
        )
    texts_by_event = variant_texts_for_events(
        bundle.session.retrieval_variants,
        bundle.session.events,
        event_ids,
    )
    variants: list[QueryVariant] = []
    for event_id in event_ids:
        for index, text in enumerate(texts_by_event[event_id]):
            variants.append(
                QueryVariant(
                    event_id=event_id,
                    variant_id=f"{event_id}:v{index}",
                    text=text,
                )
            )
    return variants


def _mutation_response(
    bundle: SessionBundle,
    invalidated: list[str],
) -> MutationResponse:
    return MutationResponse(
        session_revision=bundle.session.revision,
        invalidated_stages=invalidated,
        artifact_counts=_counts(bundle),
    )


def _run_response(
    bundle: SessionBundle,
    run,
    *,
    metrics: dict[str, Any] | None = None,
) -> RunResponse:
    return RunResponse(
        session_revision=bundle.session.revision,
        run_id=run.id,
        run_status=run.status,
        metrics=run.metrics if metrics is None else metrics,
        artifact_counts=_counts(bundle),
    )


def _counts(bundle: SessionBundle) -> ArtifactCounts:
    return ArtifactCounts(
        candidates=len(bundle.artifacts.candidates),
        regions=len(bundle.artifacts.regions),
        frontier_regions=len(bundle.artifacts.frontier_region_ids),
        frame_scores=len(bundle.artifacts.frame_scores),
        proposals=len(bundle.artifacts.proposals),
        tuples=len(bundle.artifacts.tuples),
        runs=len(bundle.runs),
    )


def _page(items, offset: int, limit: int):
    page = items[offset : offset + limit]
    return {
        "items": [item.model_dump(mode="json") for item in page],
        "total": len(items),
        "offset": offset,
        "limit": limit,
    }


def _raise_api_error(exc: Exception):
    if isinstance(exc, ConfigurationError):
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ollama_not_configured",
                "message": "OLLAMA_API_KEY is not configured",
            },
        ) from exc
    if isinstance(exc, OllamaTimeoutError):
        raise HTTPException(
            status_code=504,
            detail={"code": "ollama_timeout", "message": str(exc)},
        ) from exc
    if isinstance(exc, OllamaRateLimitError):
        raise HTTPException(
            status_code=503,
            detail={"code": "ollama_rate_limited", "message": str(exc)},
        ) from exc
    if isinstance(exc, OllamaServiceError):
        raise HTTPException(
            status_code=502,
            detail={"code": "ollama_service_error", "message": str(exc)},
        ) from exc
    if isinstance(exc, UpstreamSearchError):
        raise HTTPException(
            status_code=502,
            detail={"code": "upstream_search_error", "message": str(exc)},
        ) from exc
    if isinstance(exc, LiveRefinementUnavailableError):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "live_refinement_unavailable",
                "message": str(exc),
            },
        ) from exc
    if isinstance(
        exc,
        (LiveRefinementConfigurationError, LiveRefinementDataError),
    ):
        code = (
            "live_refinement_configuration_error"
            if isinstance(exc, LiveRefinementConfigurationError)
            else "live_refinement_data_error"
        )
        raise HTTPException(
            status_code=422,
            detail={"code": code, "message": str(exc)},
        ) from exc
    if isinstance(exc, SessionNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": "session_not_found", "message": "session not found"},
        ) from exc
    if isinstance(exc, RevisionConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "revision_conflict",
                "message": str(exc),
                "expected_revision": exc.expected,
                "current_revision": exc.actual,
            },
        ) from exc
    if isinstance(exc, (AdaptiveInputError, ValidationError, ValueError)):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_adaptive_input", "message": str(exc)},
        ) from exc
    raise exc
