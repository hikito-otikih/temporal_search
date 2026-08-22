"""Static values for the adaptive-search router and runtime wiring."""

from __future__ import annotations

ROUTER_PREFIX = "/v1"
ROUTER_TAGS = ["adaptive temporal search"]

UPSTREAM_URL_ENV_VARS = ("UPSTREAM_URL", "ADAPTIVE_UPSTREAM_URL", "UPSTREAM_SEARCH_URL")
UPSTREAM_URL_FALLBACK = "http://172.26.176.1:8000"

# retrieve_candidates() calls the upstream search once per event,
# sequentially - each call gets this budget independently. 30s (the old
# default) is already generous for one healthy embedding+vector search; this
# is a safety net against a genuinely hung single call, not the thing that
# should ever cap a multi-event query's total latency (that's the caller's -
# aic_gateway's - own timeout to set generously).
UPSTREAM_TIMEOUT_SECONDS_ENV_VAR = "UPSTREAM_TIMEOUT_SECONDS"
UPSTREAM_TIMEOUT_SECONDS_FALLBACK = 60.0

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

ERROR_CODE_UPSTREAM_SEARCH_ERROR = "upstream_search_error"
ERROR_CODE_SESSION_NOT_FOUND = "session_not_found"
ERROR_CODE_REVISION_CONFLICT = "revision_conflict"
ERROR_CODE_INVALID_ADAPTIVE_INPUT = "invalid_adaptive_input"
