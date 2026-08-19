"""Standalone /temporal-search endpoint (pre-adaptive pipeline)."""

from __future__ import annotations

from fastapi import APIRouter

from adaptive_search.router import _raise_api_error

from .schemas import TemporalSearchRequest, TemporalSearchResponse
from .service import temporal_search

router = APIRouter(tags=["legacy temporal search"])


@router.post("/temporal-search", response_model=TemporalSearchResponse)
def search(request: TemporalSearchRequest):
    try:
        results, boundary_refinement_capability, search_truncated = temporal_search(
            request.query,
            request.top_k_tuple,
            request.top_k_each_query,
            request.gamma,
            request.searcher_type,
            request.objectFilterMode,
            request.object_name_list,
            request.objectThreshold,
            request.apply_boundary_refinement,
        )
        return {
            "query": request.query,
            "results": results,
            "boundary_refinement_capability": boundary_refinement_capability,
            # True if at least one video's backtracking search hit its node
            # budget (searchers/*.MAX_TRAVERSAL_NODES) before exploring
            # exhaustively - results are still the best found so far, but
            # are not guaranteed complete for that video.
            "search_truncated": search_truncated,
        }
    except Exception as exc:
        _raise_api_error(exc)
