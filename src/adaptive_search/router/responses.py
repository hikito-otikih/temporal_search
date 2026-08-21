"""Shared response-shaping helpers used across the adaptive-search route modules."""

from __future__ import annotations

from typing import Any

from ..client import EventQuery
from ..exceptions import AdaptiveInputError
from ..session import SessionBundle

from .api_schemas import ArtifactCounts, MutationResponse, RunResponse, SessionResponse


def _session_response(bundle: SessionBundle) -> SessionResponse:
    return SessionResponse(
        session=bundle.session,
        artifact_counts=_counts(bundle),
    )


def _event_queries(
    bundle: SessionBundle,
    event_ids: list[str],
) -> list[EventQuery]:
    known = {event.event_id for event in bundle.session.events}
    if any(event_id not in known for event_id in event_ids):
        raise AdaptiveInputError(
            f"unknown event_ids: {sorted(set(event_ids) - known)}"
        )
    by_id = {event.event_id: event for event in bundle.session.events}
    return [EventQuery(event_id, by_id[event_id].anchor_query) for event_id in event_ids]


def _mutation_response(
    bundle: SessionBundle,
    invalidated: list[str],
) -> MutationResponse:
    return MutationResponse(
        session_revision=bundle.session.revision,
        invalidated_stages=invalidated,
        artifact_counts=_counts(bundle),
    )


def _run_response(
    bundle: SessionBundle,
    run,
    *,
    metrics: dict[str, Any] | None = None,
) -> RunResponse:
    return RunResponse(
        session_revision=bundle.session.revision,
        run_id=run.id,
        run_status=run.status,
        metrics=run.metrics if metrics is None else metrics,
        artifact_counts=_counts(bundle),
    )


def _counts(bundle: SessionBundle) -> ArtifactCounts:
    return ArtifactCounts(
        candidates=len(bundle.artifacts.candidates),
        regions=len(bundle.artifacts.regions),
        runs=len(bundle.runs),
    )


def _page(items, offset: int, limit: int):
    page = items[offset : offset + limit]
    return {
        "items": [item.model_dump(mode="json") for item in page],
        "total": len(items),
        "offset": offset,
        "limit": limit,
    }
