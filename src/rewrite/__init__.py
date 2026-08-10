"""Public API for the LLM query-rewrite module."""

from .config import DEFAULT_OLLAMA_MODEL, OLLAMA_API_URL, OLLAMA_TIMEOUT_SECONDS
from .constants import MAX_ANALYSIS_ATTEMPTS, SYSTEM_PROMPT
from .exceptions import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
    SemanticValidationError,
)
from .schemas import RewriteRequest, RewriteResponse
from .service import build_analysis_prompt, rewrite_queries

__all__ = [
    "DEFAULT_OLLAMA_MODEL",
    "MAX_ANALYSIS_ATTEMPTS",
    "OLLAMA_API_URL",
    "OLLAMA_TIMEOUT_SECONDS",
    "SYSTEM_PROMPT",
    "ConfigurationError",
    "OllamaRateLimitError",
    "OllamaServiceError",
    "OllamaTimeoutError",
    "SemanticValidationError",
    "RewriteRequest",
    "RewriteResponse",
    "build_analysis_prompt",
    "rewrite_queries",
]
