"""Artifact ingestion and read routes (candidates, frame scores, regions, proposals, runs)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..dependencies import adaptive_service

from .api_schemas import CandidateIngestRequest, FrameScoreIngestRequest, RetrieveSessionRequest, RunResponse
from .errors import _raise_api_error
from .responses import _build_query_variants, _page, _run_response

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
        variants = _build_query_variants(bundle, selected_events)
        candidates = _router.upstream_search_client.retrieve_candidates(
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
