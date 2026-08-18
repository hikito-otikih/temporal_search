"""Versioned FastAPI router for the adaptive-search research vertical slice.

Route handlers are split by resource area across this package's submodules
and composed into the single ``router`` object below, so ``main.py`` keeps
importing ``from adaptive_search.router import router`` unchanged.

``rewrite_queries_and_build_plan`` and ``upstream_search_client`` are
re-exported here (not just imported where used) because tests patch them by
dotted string path at ``adaptive_search.router.<name>``
(``unittest.mock.patch("adaptive_search.router.upstream_search_client", ...)``).
The submodules that call them (``sessions.py``, ``artifacts.py``) look them
up through this package's own namespace at call time - via
``from adaptive_search import router as _router`` then ``_router.<name>`` -
rather than binding a local copy at import time, so a patch applied here is
actually observed.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..constants import ROUTER_PREFIX, ROUTER_TAGS
from ..dependencies import (
    adaptive_service,
    boundary_refinement_runtime,
    configure_boundary_refinement_runtime,
    upstream_search_client,
)
from ..rewrite_bridge import rewrite_queries_and_build_plan

from .errors import _raise_api_error
from .api_schemas import (
    ApiModel,
    ArtifactCounts,
    CandidateIngestRequest,
    ClearEventConstraintRequest,
    CreateSessionRequest,
    EventPatch,
    FixFrameRequest,
    FrameScoreIngestRequest,
    MarkVideosRequest,
    MutationResponse,
    PatchEventRequest,
    PatchHyperparametersRequest,
    RejectProposalRequest,
    ReplaceConstraintsRequest,
    RetrieveSessionRequest,
    RunResponse,
    SessionResponse,
)

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


@router.on_event("startup")
def _configure_boundary_refinement_runtime() -> None:
    configure_boundary_refinement_runtime()


from .sessions import router as _sessions_router
from .artifacts import router as _artifacts_router
from .commands import router as _commands_router
from .video_priorities import router as _video_priorities_router
from .media import router as _media_router

router.include_router(_sessions_router)
router.include_router(_artifacts_router)
router.include_router(_commands_router)
router.include_router(_video_priorities_router)
router.include_router(_media_router)
