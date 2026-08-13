"""Standalone /temporal-search endpoint (pre-adaptive pipeline)."""

from __future__ import annotations

from fastapi import APIRouter

from adaptive_search.router import _raise_api_error

from .schemas import TemporalSearchRequest
from .service import temporal_search

router = APIRouter(tags=["legacy temporal search"])


@router.post("/temporal-search")
def search(request: TemporalSearchRequest):
    try:
        results, boundary_refinement_capability = temporal_search(
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
        }
    except Exception as exc:
        _raise_api_error(exc)
