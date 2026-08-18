"""Capability discovery and session lifecycle routes."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Query, Response, status

from rewrite.schemas import RewriteRequest

from ..constants import SUPPORTED_BOUNDARY_PROFILES, SUPPORTED_RELATIONS
from ..dependencies import adaptive_service, boundary_refinement_runtime
from ..embedding import (
    DEFAULT_SIGLIP2_MODEL,
    QUALITY_QWEN_EMBEDDING_MODEL,
    QUALITY_QWEN_RERANKER_MODEL,
)
from ..schemas import SearchHyperparameters

from .api_schemas import (
    CreateSessionRequest,
    MutationResponse,
    PatchEventRequest,
    PatchHyperparametersRequest,
    ReplaceConstraintsRequest,
    SessionResponse,
)
from .errors import _raise_api_error
from .responses import _apply_runtime_embedding_default, _mutation_response, _session_response

router = APIRouter()


@router.get("/searchers")
def list_searchers():
    capability = boundary_refinement_runtime.capabilities()
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
                "supported_relations": SUPPORTED_RELATIONS,
                "supported_boundary_profiles": SUPPORTED_BOUNDARY_PROFILES,
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
    from adaptive_search import router as _router

    try:
        plan = await _router.rewrite_queries_and_build_plan(
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
