"""Session commands that mutate constraints (mark videos, fix a frame, reject a proposal, clear a constraint)."""

from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import adaptive_service

from .api_schemas import (
    ClearEventConstraintRequest,
    FixFrameRequest,
    MarkVideosRequest,
    MutationResponse,
    RejectProposalRequest,
)
from .errors import _raise_api_error
from .responses import _mutation_response

router = APIRouter()


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
