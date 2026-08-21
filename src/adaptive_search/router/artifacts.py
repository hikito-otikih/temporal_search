"""Artifact ingestion and read routes (candidates, regions, runs).

No frame-score ingestion route exists here - `/artifacts/frame-scores` and
its `AdaptiveSearchService.replace_frame_scores()` backing were removed
along with the refinement frontier (`select_refinement_frontier`); both
existed only to support an external client precomputing frame-level
refinement scores and submitting them back into a session, a workflow this
product does not support.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..dependencies import adaptive_service

from .api_schemas import CandidateIngestRequest, RetrieveSessionRequest, RunResponse
from .errors import _raise_api_error
from .responses import _event_queries, _page, _run_response

router = APIRouter()


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
    "/search-sessions/{session_id}/commands/retrieve",
    response_model=RunResponse,
)
def retrieve_session(session_id: str, request: RetrieveSessionRequest):
    from adaptive_search import router as _router

    try:
        bundle = adaptive_service.get_session(session_id)
        selected_events = request.event_ids or [
            event.event_id for event in bundle.session.events
        ]
        queries = _event_queries(bundle, selected_events)
        candidates = _router.upstream_search_client.retrieve_candidates(
            session_id=session_id,
            events=queries,
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
        page = items[offset : offset + limit]
        return {
            "items": [item.model_dump(mode="json") for item in page],
            "total": len(items),
            "offset": offset,
            "limit": limit,
        }
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
