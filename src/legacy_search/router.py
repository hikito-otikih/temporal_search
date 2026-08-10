"""Standalone /temporal-search endpoint (pre-adaptive pipeline)."""

from __future__ import annotations

from fastapi import APIRouter

from .schemas import TemporalSearchRequest
from .service import temporal_search

router = APIRouter(tags=["legacy temporal search"])


@router.post("/temporal-search")
def search(request: TemporalSearchRequest):
    results = temporal_search(
        request.query,
        request.top_k_tuple,
        request.top_k_each_query,
        request.gamma,
        request.searcher_type,
        request.objectFilterMode,
        request.object_name_list,
        request.objectThreshold,
    )
    return {"query": request.query, "results": results}
