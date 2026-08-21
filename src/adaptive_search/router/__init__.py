"""Versioned FastAPI router for the adaptive-search research vertical slice.

Route handlers are split by resource area across this package's submodules
and composed into the single ``router`` object below, so ``main.py`` keeps
importing ``from adaptive_search.router import router`` unchanged.

``upstream_search_client`` is re-exported here (not just imported where
used) because tests patch it by dotted string path at
``adaptive_search.router.upstream_search_client``
(``unittest.mock.patch("adaptive_search.router.upstream_search_client", ...)``).
The submodule that calls it (``artifacts.py``) looks it up through this
package's own namespace at call time - via
``from adaptive_search import router as _router`` then
``_router.upstream_search_client`` - rather than binding a local copy at
import time, so a patch applied here is actually observed.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..constants import ROUTER_PREFIX, ROUTER_TAGS
from ..dependencies import (
    adaptive_service,
    upstream_search_client,
)

from .errors import _raise_api_error
from .api_schemas import (
    ApiModel,
    ArtifactCounts,
    CandidateIngestRequest,
    ClearEventConstraintRequest,
    CreateSessionRequest,
    EventPatch,
    FixFrameRequest,
    MarkVideosRequest,
    MutationResponse,
    PatchEventRequest,
    PatchHyperparametersRequest,
    ReplaceConstraintsRequest,
    RetrieveSessionRequest,
    RunResponse,
    SessionResponse,
)

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


from .sessions import router as _sessions_router
from .artifacts import router as _artifacts_router
from .commands import router as _commands_router
from .video_priorities import router as _video_priorities_router

router.include_router(_sessions_router)
router.include_router(_artifacts_router)
router.include_router(_commands_router)
router.include_router(_video_priorities_router)
