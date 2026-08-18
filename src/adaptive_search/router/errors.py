"""Translate internal exceptions raised while handling a request into HTTPException."""

from __future__ import annotations

from fastapi import HTTPException
from pydantic import ValidationError

from rewrite.exceptions import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
)

from ..constants import (
    ERROR_CODE_INVALID_ADAPTIVE_INPUT,
    ERROR_CODE_LIVE_REFINEMENT_CONFIGURATION_ERROR,
    ERROR_CODE_LIVE_REFINEMENT_DATA_ERROR,
    ERROR_CODE_LIVE_REFINEMENT_UNAVAILABLE,
    ERROR_CODE_OLLAMA_NOT_CONFIGURED,
    ERROR_CODE_OLLAMA_RATE_LIMITED,
    ERROR_CODE_OLLAMA_SERVICE_ERROR,
    ERROR_CODE_OLLAMA_TIMEOUT,
    ERROR_CODE_REVISION_CONFLICT,
    ERROR_CODE_SESSION_NOT_FOUND,
    ERROR_CODE_UPSTREAM_SEARCH_ERROR,
    ERROR_CODE_VIDEO_NOT_FOUND,
)
from ..exceptions import (
    AdaptiveInputError,
    LiveRefinementConfigurationError,
    LiveRefinementDataError,
    LiveRefinementUnavailableError,
    RevisionConflictError,
    SessionNotFoundError,
    UpstreamSearchError,
)
from ..providers import VideoAssetNotFoundError


def _raise_api_error(exc: Exception):
    if isinstance(exc, ConfigurationError):
        raise HTTPException(
            status_code=500,
            detail={
                "code": ERROR_CODE_OLLAMA_NOT_CONFIGURED,
                "message": "OLLAMA_API_KEY is not configured",
            },
        ) from exc
    if isinstance(exc, OllamaTimeoutError):
        raise HTTPException(
            status_code=504,
            detail={"code": ERROR_CODE_OLLAMA_TIMEOUT, "message": str(exc)},
        ) from exc
    if isinstance(exc, OllamaRateLimitError):
        raise HTTPException(
            status_code=503,
            detail={"code": ERROR_CODE_OLLAMA_RATE_LIMITED, "message": str(exc)},
        ) from exc
    if isinstance(exc, OllamaServiceError):
        raise HTTPException(
            status_code=502,
            detail={"code": ERROR_CODE_OLLAMA_SERVICE_ERROR, "message": str(exc)},
        ) from exc
    if isinstance(exc, UpstreamSearchError):
        raise HTTPException(
            status_code=502,
            detail={"code": ERROR_CODE_UPSTREAM_SEARCH_ERROR, "message": str(exc)},
        ) from exc
    if isinstance(exc, LiveRefinementUnavailableError):
        raise HTTPException(
            status_code=503,
            detail={
                "code": ERROR_CODE_LIVE_REFINEMENT_UNAVAILABLE,
                "message": str(exc),
            },
        ) from exc
    if isinstance(
        exc,
        (LiveRefinementConfigurationError, LiveRefinementDataError),
    ):
        code = (
            ERROR_CODE_LIVE_REFINEMENT_CONFIGURATION_ERROR
            if isinstance(exc, LiveRefinementConfigurationError)
            else ERROR_CODE_LIVE_REFINEMENT_DATA_ERROR
        )
        raise HTTPException(
            status_code=422,
            detail={"code": code, "message": str(exc)},
        ) from exc
    if isinstance(exc, VideoAssetNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": ERROR_CODE_VIDEO_NOT_FOUND, "message": str(exc)},
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
