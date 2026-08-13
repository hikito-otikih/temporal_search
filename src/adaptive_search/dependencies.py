"""Process-wide singletons the router closes over.

The service holds all session state in-process (see ``session.py``), so there
is no per-request repository/service to construct — one ``AdaptiveSearchService``
instance backs every request, and tests reset it directly between cases
(``adaptive_search.dependencies.adaptive_service.repository.clear()``).
"""

from __future__ import annotations

import os

from .config import configure_from_environment
from .refinement_runtime import BoundaryRefinementRuntime
from .client import UpstreamSearchClient
from .service import AdaptiveSearchService
from .constants import (
    UPSTREAM_URL_ENV_VARS,
    UPSTREAM_URL_FALLBACK,
)

adaptive_service = AdaptiveSearchService()
boundary_refinement_runtime = BoundaryRefinementRuntime()

upstream_search_client = UpstreamSearchClient(
    next(
        (url for var in UPSTREAM_URL_ENV_VARS if (url := os.getenv(var))),
        UPSTREAM_URL_FALLBACK,
    )
)


def configure_boundary_refinement_runtime() -> None:
    configure_from_environment(adaptive_service, boundary_refinement_runtime)
