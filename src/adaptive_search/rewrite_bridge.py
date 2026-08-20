"""Bridge from the LLM query rewrite to adaptive-search session artifacts.

The rewrite endpoint produces structured Vietnamese analysis (``RewriteResponse``).
This module maps that analysis into the domain inputs the adaptive pipeline
consumes: ordered ``EventDefinition`` objects plus per-event retrieval variants
that the ``commands/retrieve`` endpoint will send to the sparse upstream service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rewrite.service import rewrite_queries
from rewrite.schemas import RewriteResponse

from .schemas import BoundaryType, EventDefinition

BOUNDARY_MAPPING: dict[str, BoundaryType] = {
    "start": "onset",
    "end": "offset",
    "transition": "transition",
    "state": "state",
    "interval": "state",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class SessionPlan:
    """Inputs needed to create an adaptive session from a rewrite analysis."""

    common_query: str | None
    events: tuple[EventDefinition, ...]
    retrieval_variants: dict[str, tuple[str, ...]]

    def variant_texts_for(self, event_id: str) -> tuple[str, ...]:
        return self.retrieval_variants.get(event_id, ())


def build_session_plan(
    analysis: RewriteResponse,
    common_query: str | None = None,
) -> SessionPlan:
    events: list[EventDefinition] = []
    retrieval_variants: dict[str, tuple[str, ...]] = {}
    for index, event in enumerate(analysis.events):
        event_id = f"evt{index}"
        relation = event.temporal_relation
        events.append(
            EventDefinition(
                event_id=event_id,
                original_query=event.original_query,
                anchor_query=event.anchor_query,
                pre_state=event.pre_state,
                post_state=event.post_state,
                boundary_type=_boundary_type(event.boundary),
                temporal_relation=relation.relation,
                reference_event_id=(
                    f"evt{relation.reference_event_id}"
                    if relation.reference_event_id is not None
                    else None
                ),
            )
        )
        # original_query (the caller's own unmodified text) is included as
        # a variant, replacing one of the two retrieval_queries_vi
        # paraphrases rather than adding a 5th (query_variants_per_event
        # defaults to 4 - retrieval.py rejects more). Measured on a 60-video
        # live YouCook2 benchmark: retrieval using only the LLM's own
        # elaborated variants (previous behavior) landed below a baseline
        # that searched with this exact original_query text directly -
        # recall@1 0.267 vs 0.300, MRR 0.352 vs 0.378 - meaning the RRF
        # fusion pool never contained the one signal that outperformed
        # everything the rewrite stage produced on its own. Including it
        # here lets fusion combine that strong signal with the LLM's
        # variants (multi-language coverage, common-query-distilled
        # context) instead of choosing between them.
        retrieval_variants[event_id] = tuple(
            dict.fromkeys(
                [
                    event.original_query,
                    *event.retrieval_queries_vi[1:],
                    *event.retrieval_queries_en,
                ]
            )
        )
    return SessionPlan(
        common_query=common_query,
        events=tuple(events),
        retrieval_variants=retrieval_variants,
    )


def _boundary_type(literal: str) -> BoundaryType:
    try:
        return BOUNDARY_MAPPING[literal]
    except KeyError as exc:
        raise ValueError(f"unsupported rewrite boundary literal: {literal!r}") from exc


async def rewrite_queries_and_build_plan(
    *,
    queries: Sequence[str],
    common_query: str | None = None,
) -> SessionPlan:
    """Run the LLM rewrite and map its analysis into an adaptive SessionPlan."""
    analysis = await rewrite_queries(queries=queries, common_query=common_query)
    return build_session_plan(analysis, common_query)


def variant_texts_for_events(
    retrieval_variants: Mapping[str, tuple[str, ...]],
    events: Sequence[EventDefinition],
    event_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Resolve retrieval variant texts for a selected event subset.

    Events without stored variants fall back to their anchor query so the
    sparse retrieval command always has at least one query per event.
    """
    by_id = {event.event_id: event for event in events}
    texts_by_event: dict[str, tuple[str, ...]] = {}
    for event_id in event_ids:
        if event_id not in by_id:
            raise ValueError(f"unknown event_id: {event_id}")
        texts = retrieval_variants.get(event_id)
        texts_by_event[event_id] = tuple(texts) if texts else (by_id[event_id].anchor_query,)
    return texts_by_event
