"""Standalone /rewrite endpoint: LLM query analysis without a session."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .exceptions import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
)
from .schemas import RewriteRequest, RewriteResponse
from .service import rewrite_queries

router = APIRouter(tags=["query rewrite"])


@router.post('/rewrite', response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    try:
        analysis = await rewrite_queries(
            common_query=request.common_query,
            queries=request.query,
        )
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=500, detail='OLLAMA_API_KEY is not configured'
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail='Ollama request timed out') from exc
    except OllamaRateLimitError as exc:
        raise HTTPException(
            status_code=503, detail='Ollama rate limit exceeded'
        ) from exc
    except OllamaServiceError as exc:
        raise HTTPException(status_code=502, detail='Ollama request failed') from exc

    return analysis
