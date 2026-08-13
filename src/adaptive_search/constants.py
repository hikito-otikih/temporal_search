"""Static values for the adaptive-search router and runtime wiring."""

from __future__ import annotations

ROUTER_PREFIX = "/v1"
ROUTER_TAGS = ["adaptive temporal search"]

UPSTREAM_URL_ENV_VARS = ("UPSTREAM_URL", "ADAPTIVE_UPSTREAM_URL", "UPSTREAM_SEARCH_URL")
UPSTREAM_URL_FALLBACK = "http://172.26.176.1:8000"

SUPPORTED_RELATIONS = ["ordered_sequence", "adjacent_gap"]
SUPPORTED_BOUNDARY_PROFILES = [
    "onset",
    "offset",
    "transition",
    "state",
    "plateau_start",
    "symmetric_peak",
    "unknown",
]

ERROR_CODE_OLLAMA_NOT_CONFIGURED = "ollama_not_configured"
ERROR_CODE_OLLAMA_TIMEOUT = "ollama_timeout"
ERROR_CODE_OLLAMA_RATE_LIMITED = "ollama_rate_limited"
ERROR_CODE_OLLAMA_SERVICE_ERROR = "ollama_service_error"
ERROR_CODE_UPSTREAM_SEARCH_ERROR = "upstream_search_error"
ERROR_CODE_LIVE_REFINEMENT_UNAVAILABLE = "live_refinement_unavailable"
ERROR_CODE_LIVE_REFINEMENT_CONFIGURATION_ERROR = "live_refinement_configuration_error"
ERROR_CODE_LIVE_REFINEMENT_DATA_ERROR = "live_refinement_data_error"
ERROR_CODE_SESSION_NOT_FOUND = "session_not_found"
ERROR_CODE_REVISION_CONFLICT = "revision_conflict"
ERROR_CODE_INVALID_ADAPTIVE_INPUT = "invalid_adaptive_input"
ERROR_CODE_VIDEO_NOT_FOUND = "video_not_found"
