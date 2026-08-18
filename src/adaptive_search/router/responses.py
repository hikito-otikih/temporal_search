"""Shared response-shaping helpers used across the adaptive-search route modules."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..client import QueryVariant
from ..dependencies import boundary_refinement_runtime
from ..exceptions import AdaptiveInputError
from ..schemas import SearchHyperparameters
from ..session import SessionBundle

from .api_schemas import ArtifactCounts, MutationResponse, RunResponse, SessionResponse


def _session_response(bundle: SessionBundle) -> SessionResponse:
    refinement = bundle.session.hyperparameters.refinement
    runtime_spec = (
        refinement.quality_embedding_model
        if refinement.quality_profile_enabled
        else refinement.embedding_model
    )
    return SessionResponse(
        session=bundle.session,
        artifact_counts=_counts(bundle),
        live_refinement=asdict(
            boundary_refinement_runtime.capabilities(runtime_spec)
        ),
    )


def _apply_runtime_embedding_default(
    hyperparameters: SearchHyperparameters,
) -> SearchHyperparameters:
    if "embedding_model" in hyperparameters.refinement.model_fields_set:
        return hyperparameters
    runtime_spec = boundary_refinement_runtime.configured_runtime_spec()
    if runtime_spec is None:
        return hyperparameters
    return hyperparameters.model_copy(
        update={
            "refinement": hyperparameters.refinement.model_copy(
                update={"embedding_model": runtime_spec}
            )
        }
    )


def _build_query_variants(
    bundle: SessionBundle,
    event_ids: list[str],
) -> list[QueryVariant]:
    from ..rewrite_bridge import variant_texts_for_events

    known = {event.event_id for event in bundle.session.events}
    if any(event_id not in known for event_id in event_ids):
        raise AdaptiveInputError(
            f"unknown event_ids: {sorted(set(event_ids) - known)}"
        )
    texts_by_event = variant_texts_for_events(
        bundle.session.retrieval_variants,
        bundle.session.events,
        event_ids,
    )
    variants: list[QueryVariant] = []
    for event_id in event_ids:
        for index, text in enumerate(texts_by_event[event_id]):
            variants.append(
                QueryVariant(
                    event_id=event_id,
                    variant_id=f"{event_id}:v{index}",
                    text=text,
                )
            )
    return variants


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
        proposals=len(bundle.artifacts.proposals),
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
