"""Translate internal exceptions raised while handling a request into HTTPException."""

from __future__ import annotations

from fastapi import HTTPException
from pydantic import ValidationError

from ..constants import (
    ERROR_CODE_INVALID_ADAPTIVE_INPUT,
    ERROR_CODE_REVISION_CONFLICT,
    ERROR_CODE_SESSION_NOT_FOUND,
    ERROR_CODE_UPSTREAM_SEARCH_ERROR,
)
from ..exceptions import (
    AdaptiveInputError,
    RevisionConflictError,
    SessionNotFoundError,
    UpstreamSearchError,
)


def _raise_api_error(exc: Exception):
    if isinstance(exc, UpstreamSearchError):
        raise HTTPException(
            status_code=502,
            detail={"code": ERROR_CODE_UPSTREAM_SEARCH_ERROR, "message": str(exc)},
        ) from exc
    if isinstance(exc, SessionNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": ERROR_CODE_SESSION_NOT_FOUND, "message": "session not found"},
        ) from exc
    if isinstance(exc, RevisionConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": ERROR_CODE_REVISION_CONFLICT,
                "message": str(exc),
                "expected_revision": exc.expected,
                "current_revision": exc.actual,
            },
        ) from exc
    if isinstance(exc, (AdaptiveInputError, ValidationError, ValueError)):
        raise HTTPException(
            status_code=422,
            detail={"code": ERROR_CODE_INVALID_ADAPTIVE_INPUT, "message": str(exc)},
        ) from exc
    raise exc
