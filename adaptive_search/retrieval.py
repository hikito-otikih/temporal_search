"""Per-event query-variant fusion for sparse retrieval candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256

from .models import RetrievalHyperparameters, SparseCandidate


def fuse_candidates_rrf(
    candidates: Sequence[SparseCandidate],
    parameters: RetrievalHyperparameters | None = None,
) -> list[SparseCandidate]:
    """Fuse query variants with Reciprocal Rank Fusion and frame deduplication.

    Upstream scores are only used to rank results inside one query variant.
    Their raw scales are never averaged across Vietnamese/English variants.
    The output remains grouped by semantic event, so four rewrite variants do
    not accidentally become four temporal events.
    """

    parameters = parameters or RetrievalHyperparameters()
    grouped: dict[tuple[str, str], list[SparseCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.session_id, candidate.event_id)].append(candidate)

    fused: list[SparseCandidate] = []
    for (session_id, event_id), event_candidates in sorted(grouped.items()):
        by_variant: dict[str, list[SparseCandidate]] = defaultdict(list)
        for candidate in event_candidates:
            by_variant[candidate.query_variant].append(candidate)
        if len(by_variant) > parameters.query_variants_per_event:
            raise ValueError(
                f"event {event_id!r} has {len(by_variant)} query variants; "
                f"maximum is {parameters.query_variants_per_event}"
            )

        aggregate: dict[tuple[str, int], dict] = {}
        for variant in sorted(by_variant):
            # A variant can contain duplicate rows. Keep its strongest row for
            # each physical frame before assigning ranks.
            best_by_frame: dict[tuple[str, int], SparseCandidate] = {}
            for candidate in by_variant[variant]:
                key = (candidate.video_id, candidate.frame_id)
                current = best_by_frame.get(key)
                if current is None or (
                    candidate.raw_relevance_score,
                    -candidate.timestamp_seconds,
                    candidate.id,
                ) > (
                    current.raw_relevance_score,
                    -current.timestamp_seconds,
                    current.id,
                ):
                    best_by_frame[key] = candidate

            ranked = sorted(
                best_by_frame.values(),
                key=lambda item: (
                    -item.raw_relevance_score,
                    item.video_id,
                    item.frame_id,
                    item.timestamp_seconds,
                    item.id,
                ),
            )[: parameters.top_n_per_variant]
            for rank, candidate in enumerate(ranked, start=1):
                key = (candidate.video_id, candidate.frame_id)
                bucket = aggregate.setdefault(
                    key,
                    {
                        "representative": candidate,
                        "rrf_score": 0.0,
                        "variants": set(),
                    },
                )
                bucket["rrf_score"] += 1.0 / (parameters.rrf_k + rank)
                bucket["variants"].add(variant)
                representative = bucket["representative"]
                if candidate.raw_relevance_score > representative.raw_relevance_score:
                    bucket["representative"] = candidate

        ranked_fused = sorted(
            aggregate.items(),
            key=lambda item: (
                -item[1]["rrf_score"],
                -item[1]["representative"].raw_relevance_score,
                item[0][0],
                item[0][1],
            ),
        )[: parameters.top_n_fused]
        for (video_id, frame_id), bucket in ranked_fused:
            representative = bucket["representative"]
            variants = sorted(bucket["variants"])
            fused.append(
                SparseCandidate(
                    id=_stable_id(session_id, event_id, video_id, frame_id),
                    session_id=session_id,
                    event_id=event_id,
                    video_id=video_id,
                    frame_id=frame_id,
                    timestamp_seconds=representative.timestamp_seconds,
                    raw_relevance_score=bucket["rrf_score"],
                    normalized_relevance_score=None,
                    query_variant="rrf:" + ",".join(variants),
                )
            )

    return fused


def _stable_id(session_id: str, event_id: str, video_id: str, frame_id: int) -> str:
    payload = "\x1f".join(
        (session_id, event_id, video_id, str(frame_id))
    ).encode("utf-8")
    return f"candidate_rrf_{sha256(payload).hexdigest()[:20]}"
