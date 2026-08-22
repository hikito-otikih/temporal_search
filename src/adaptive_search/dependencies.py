"""Process-wide singletons the router closes over.

The service holds all session state in-process (see ``session.py``), so there
is no per-request repository/service to construct — one ``AdaptiveSearchService``
instance backs every request, and tests reset it directly between cases
(``adaptive_search.dependencies.adaptive_service.repository.clear()``).
"""

from __future__ import annotations

import os

from .client import UpstreamSearchClient
from .service import AdaptiveSearchService
from .constants import (
    UPSTREAM_TIMEOUT_SECONDS_ENV_VAR,
    UPSTREAM_TIMEOUT_SECONDS_FALLBACK,
    UPSTREAM_URL_ENV_VARS,
    UPSTREAM_URL_FALLBACK,
)

adaptive_service = AdaptiveSearchService()


def _upstream_timeout_seconds() -> float:
    raw = os.getenv(UPSTREAM_TIMEOUT_SECONDS_ENV_VAR)
    if raw is None:
        return UPSTREAM_TIMEOUT_SECONDS_FALLBACK
    try:
        value = float(raw)
    except ValueError:
        return UPSTREAM_TIMEOUT_SECONDS_FALLBACK
    return value if value > 0 else UPSTREAM_TIMEOUT_SECONDS_FALLBACK


upstream_search_client = UpstreamSearchClient(
    next(
        (url for var in UPSTREAM_URL_ENV_VARS if (url := os.getenv(var))),
        UPSTREAM_URL_FALLBACK,
    ),
    timeout_seconds=_upstream_timeout_seconds(),
)
