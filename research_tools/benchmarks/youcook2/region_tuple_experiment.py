"""Ablation sweep: does multi-region pooling + order-aware tuple scoring
(`adaptive_search.tuple_ranking`, the shipped production module) beat
today's production `prioritize_videos()` (independent per-event argmax) on
real YouCook2 queries?

This driver imports the production algorithm directly (not a benchmark-tree
duplicate) so results describe the exact code that ships, including both
production correctness fixes: order constraints come from real
`EventDefinition.temporal_relation`/`.reference_event_id` with transitive
closure (`build_order_constraints`), not adjacent list position.

For each of n sample videos, fetch real upstream candidates ONCE, fuse ONCE,
then build TWO region sets from that same fused pool - clustered
(`cluster_temporal_regions`, the production default) and atomic (one
`TemporalRegion` per candidate, no clustering at all - the "treat every
keyframe as its own region" ablation condition) - and compute, against
each:

  - rank_baseline(_atomic): prioritize_videos()'s rank for the ground-truth
    video.
  - hits_baseline(_atomic): how many of the GT video's own events land
    inside their ground-truth interval, using today's independent per-event
    best region (select_event_seeds()'s own seed[0]).
  - for every swept TupleRankingHyperparameters config: rank_tuple and
    hits_tuple, same two questions, answered by tuple ranking instead.

All metrics are computed from the SAME fetched candidate set for a paired,
noise-free comparison - no repeated network calls once data is fetched.
Metrics reported with the repo's existing recall@k_new / final_query_score
definitions (boundary_metrics.py) for direct comparability with prior
benchmark documents.

Several configs use `temporal_relation` from `build_temporal_relations_cache.py`'s
cached real LLM classification (--temporal-relations-cache): cached per-event
relation/reference data is used to build real `EventDefinition` objects and
pass them through the production `build_order_constraints()` - the same
function `router.py` calls for a live session - so this benchmark is not
just testing the *algorithm* end to end, but this specific piece of
production wiring as well.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .boundary_metrics import DEFAULT_KS, frame_hits
from .coarse_anchor import WINNER_REFINEMENT_PARAMS, WINNER_RETRIEVAL_PARAMS, WINNER_UPSTREAM_TOP_K
from .core import VideoQueryGroup, canonical_video_id, load_query_directory_grouped
from .legacy_clustering import cluster_temporal_regions
from adaptive_search.algorithms import atomic_regions, prioritize_videos
from adaptive_search.boundary_seeds import select_event_seeds
from adaptive_search.client import QueryVariant
from adaptive_search.dependencies import upstream_search_client
from adaptive_search.retrieval import fuse_candidates_rrf
from adaptive_search.schemas import EventDefinition, SparseCandidate, TemporalRegion, TupleRankingHyperparameters
from adaptive_search.tuple_ranking import build_order_constraints, rank_videos_by_region_tuples

DEFAULT_PARAMS = TupleRankingHyperparameters()


def _with(**overrides: Any) -> TupleRankingHyperparameters:
    """`TupleRankingHyperparameters` is a Pydantic model, not a dataclass -
    `dataclasses.replace()` doesn't apply; `.model_copy(update=...)` does."""
    return DEFAULT_PARAMS.model_copy(update=overrides)


# The N^event_count blowup only bites once pools are large; N<=20's default
# cap of 20000 already truncates some real videos (20^4=160000 > 20000).
# Raised generously for the N-scaling ablation so a larger N is measured on
# its own merits, not on how much of the truncated search space it explored.
_LARGE_N_MAX_COMBINATIONS = 500_000


@dataclass(frozen=True)
class SweepConfig:
    label: str
    params: TupleRankingHyperparameters
    use_temporal_relation: bool = False
    use_atomic_regions: bool = False


SWEEP_CONFIGS: list[SweepConfig] = [
    SweepConfig("baseline_equivalent (order_weight=0)", _with(order_weight=0.0)),
    SweepConfig("default", DEFAULT_PARAMS),
    SweepConfig("delta=0.05", _with(relative_delta=0.05)),
    SweepConfig("delta=0.10", _with(relative_delta=0.10)),
    SweepConfig("delta=0.25", _with(relative_delta=0.25)),
    SweepConfig("delta=0.40", _with(relative_delta=0.40)),
    SweepConfig("N=1 (degenerate, no pooling)", _with(max_regions_per_event=1)),
    SweepConfig("N=2", _with(max_regions_per_event=2)),
    SweepConfig("N=3", _with(max_regions_per_event=3)),
    SweepConfig("N=5", _with(max_regions_per_event=5)),
    SweepConfig("N=10", _with(max_regions_per_event=10)),
    SweepConfig("N=20 (proposal's own cap)", _with(max_regions_per_event=20)),
    SweepConfig("N=30", _with(max_regions_per_event=30, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("N=50", _with(max_regions_per_event=50, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("N=100", _with(max_regions_per_event=100, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("N=200", _with(max_regions_per_event=200, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("order_weight=0.05", _with(order_weight=0.05)),
    SweepConfig("order_weight=0.2", _with(order_weight=0.2)),
    SweepConfig("order_weight=0.4", _with(order_weight=0.4)),
    SweepConfig("order_weight=0.8", _with(order_weight=0.8)),
    SweepConfig("pooling=mean", _with(pooling="mean")),
    # --- temporal_relation gating (needs --temporal-relations-cache; now via
    # the real production build_order_constraints(), transitively closed) ---
    SweepConfig("temporal_relation-gated (default)", DEFAULT_PARAMS, use_temporal_relation=True),
    SweepConfig("temporal_relation-gated, order_weight=0.4", _with(order_weight=0.4), use_temporal_relation=True),
    SweepConfig("temporal_relation-gated, order_weight=0.8", _with(order_weight=0.8), use_temporal_relation=True),
    SweepConfig("temporal_relation-gated, pooling=mean", _with(pooling="mean"), use_temporal_relation=True),
    # --- confidence gating (DEFAULT_PARAMS already gates at threshold@0.5 -
    # this "none" row is the Round-1-style ungated reference point, needed to
    # show what gating is actually correcting, not assumed from n=30 alone) ---
    SweepConfig("confidence_gate=none, order_weight=0.8", _with(order_weight=0.8, confidence_gate="none")),
    SweepConfig("confidence_gate=linear, order_weight=0.8", _with(order_weight=0.8, confidence_gate="linear")),
    SweepConfig("confidence_gate=threshold@0.3, order_weight=0.8", _with(order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.3)),
    SweepConfig("confidence_gate=threshold@0.5, order_weight=0.8", _with(order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.5)),
    SweepConfig("confidence_gate=threshold@0.7, order_weight=0.8", _with(order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.7)),
    # --- combined: temporal_relation + confidence gating together ---
    SweepConfig(
        "temporal_relation + confidence_gate=linear, order_weight=0.8",
        _with(order_weight=0.8, confidence_gate="linear"),
        use_temporal_relation=True,
    ),
    SweepConfig(
        "temporal_relation + pooling=mean + confidence_gate=linear, order_weight=0.8",
        _with(order_weight=0.8, pooling="mean", confidence_gate="linear"),
        use_temporal_relation=True,
    ),
    # --- production shipped default (threshold@0.5, order_weight=0.8),
    # with temporal_relation now wired in too - the actual live configuration ---
    SweepConfig(
        "production (threshold@0.5, order_weight=0.8) + temporal_relation",
        _with(order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.5),
        use_temporal_relation=True,
    ),
    # --- no-clustering ablation: does Stage 3 (cluster_temporal_regions)
    # matter for tuple ranking specifically, unlike the already-proven
    # invariance of the independent-argmax baseline? ---
    SweepConfig("no-clustering (atomic regions), default", DEFAULT_PARAMS, use_atomic_regions=True),
    SweepConfig("no-clustering (atomic regions), N=1", _with(max_regions_per_event=1), use_atomic_regions=True),
    SweepConfig("no-clustering (atomic regions), N=5", _with(max_regions_per_event=5), use_atomic_regions=True),
    SweepConfig("no-clustering (atomic regions), N=20", _with(max_regions_per_event=20), use_atomic_regions=True),
    SweepConfig(
        "no-clustering (atomic regions), production (threshold@0.5, ow=0.8)",
        _with(order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.5),
        use_atomic_regions=True,
    ),
    SweepConfig("no-clustering (atomic regions), pooling=mean", _with(pooling="mean"), use_atomic_regions=True),
    # --- the one combo not yet tested: does the atomic-region win at
    # production hyperparameters (Table above) hold up once temporal_relation
    # gating - the actual shipped default - is layered on top too? ---
    SweepConfig(
        "no-clustering (atomic regions), production (threshold@0.5, ow=0.8) + temporal_relation",
        _with(order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.5),
        use_atomic_regions=True,
        use_temporal_relation=True,
    ),
]

# --- fine-grained tau sweep, both confidence signals (mean region score via
# "threshold", vs. chosen-region-vs-runner-up "margin" - see tuple_ranking.py
# _region_margin) - 19-point grid (step 0.05) each, at order_weight=0.8,
# generated rather than hand-listed to keep the two signals' grids identical
# and to make it cheap to widen later. ---
_TAU_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95

_TAU_SWEEP_CONFIGS: list[SweepConfig] = [
    SweepConfig(
        f"confidence_gate=threshold@{tau:.2f}, order_weight=0.8 (fine sweep)",
        _with(order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=tau),
    )
    for tau in _TAU_GRID
] + [
    SweepConfig(
        f"confidence_gate=margin@{tau:.2f}, order_weight=0.8 (fine sweep)",
        _with(order_weight=0.8, confidence_gate="margin", confidence_gate_threshold=tau),
    )
    for tau in _TAU_GRID
]

# --- full atomic-region grid: the same delta/N/order_weight points Rounds
# 1-2 swept under clustering, now also run with clustering skipped entirely,
# to characterize the clustering interaction at every setting, not just the
# default/N=1/N=5/N=20/production/pooling=mean spot checks above. Points
# already covered by those spot checks (delta=0.15 default, N in {1,5,20},
# order_weight=0.8) are not repeated. ---
_ATOMIC_GRID_CONFIGS: list[SweepConfig] = (
    [
        SweepConfig(f"no-clustering (atomic), delta={d}", _with(relative_delta=d), use_atomic_regions=True)
        for d in (0.05, 0.10, 0.25, 0.40)
    ]
    + [
        SweepConfig(
            f"no-clustering (atomic), N={n}",
            _with(
                max_regions_per_event=n,
                max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS if n > 20 else DEFAULT_PARAMS.max_combinations_per_video,
            ),
            use_atomic_regions=True,
        )
        for n in (2, 3, 10, 30, 50, 100, 200)
    ]
    + [
        SweepConfig(f"no-clustering (atomic), order_weight={ow}", _with(order_weight=ow), use_atomic_regions=True)
        for ow in (0.05, 0.2, 0.4)
    ]
)

SWEEP_CONFIGS = SWEEP_CONFIGS + _TAU_SWEEP_CONFIGS + _ATOMIC_GRID_CONFIGS


def load_temporal_relations_cache(path: Path) -> dict[str, list[dict[str, Any]]]:
    """video_id -> list of {event_id (int, 0-indexed, query order), relation,
    reference_event_id}, skipping any row that recorded an error."""

    cache: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            continue
        cache[row["video_id"]] = row["events"]
    return cache


def events_with_relations_from_cache(
    cached_events: list[dict[str, Any]] | None, event_ids: list[str]
) -> list[EventDefinition] | None:
    """Build real `EventDefinition`s (with `temporal_relation`/
    `reference_event_id`) from one video's cached rewrite classification, so
    the benchmark can call the exact same `build_order_constraints()`
    `router.py` calls for a live session - not a reimplementation.

    None (no cached classification for this video, e.g. the rewrite call
    failed) propagates through to `build_order_constraints`'s own None
    handling, which falls back to the adjacent-list-position chain."""

    if cached_events is None or len(cached_events) != len(event_ids):
        return None
    by_index = {event["event_id"]: event for event in cached_events}
    events: list[EventDefinition] = []
    for index, event_id in enumerate(event_ids):
        cached = by_index.get(index)
        if cached is None:
            return None
        reference = cached["reference_event_id"]
        events.append(
            EventDefinition(
                event_id=event_id,
                original_query=event_id,
                anchor_query=event_id,
                temporal_relation=cached["relation"],
                reference_event_id=event_ids[reference] if reference is not None else None,
            )
        )
    return events


def _baseline_rank_and_hits(
    group: VideoQueryGroup, regions, candidates_by_id, event_ids: list[str]
) -> tuple[int | None, int]:
    priorities = prioritize_videos(regions, event_ids, WINNER_REFINEMENT_PARAMS)
    rank = next(
        (pos for pos, p in enumerate(priorities, 1)
         if canonical_video_id(p.video_id) == group.video_id),
        None,
    )
    anchors: dict[str, float] = {}
    for event_id in event_ids:
        seeds = select_event_seeds(
            regions=regions, candidates_by_id=candidates_by_id,
            event_id=event_id, video_id=group.video_id,
        )
        if seeds:
            anchors[event_id] = seeds[0]
    return rank, frame_hits(group, anchors)


def _tuple_gt_hits(
    group: VideoQueryGroup, tuples_by_video, event_ids: list[str]
) -> int:
    tuples = tuples_by_video.get(group.video_id)
    if not tuples:
        return 0
    winner = tuples[0]
    anchors = {
        event_id: timestamp
        for event_id, timestamp in zip(event_ids, winner.timestamps)
        if timestamp is not None
    }
    return frame_hits(group, anchors)


def run_sweep(
    *, query_dir: Path, output_path: Path, video_limit: int,
    temporal_relations_cache_path: Path | None,
    progress_every: int = 5,
) -> None:
    groups = load_query_directory_grouped(query_dir)[:video_limit]
    print(f"Loaded {len(groups)} video query groups from {query_dir}")

    relations_cache: dict[str, list[dict[str, Any]]] = {}
    if temporal_relations_cache_path is not None:
        relations_cache = load_temporal_relations_cache(temporal_relations_cache_path)
        print(f"Loaded temporal_relation cache for {len(relations_cache)} videos "
              f"from {temporal_relations_cache_path}")
        missing = [g.video_id for g in groups if g.video_id not in relations_cache]
        if missing:
            print(f"  (missing/failed cache entries for {len(missing)} videos - "
                  f"those fall back to 'check every pair' for temporal_relation configs)")

    upstream = upstream_search_client
    rows: list[dict[str, Any]] = []
    started = time.monotonic()

    for index, group in enumerate(groups, 1):
        event_ids = [event_id for event_id, _ in group.events]
        variants = [
            QueryVariant(event_id=event_id, variant_id=f"{event_id}:v0", text=text)
            for event_id, text in group.events
        ]
        session_id = f"tuple_exp_{group.video_id}"
        candidates = upstream.retrieve_candidates(
            session_id=session_id, variants=variants, top_k=WINNER_UPSTREAM_TOP_K
        )
        fused = fuse_candidates_rrf(candidates, WINNER_RETRIEVAL_PARAMS)
        candidates_by_id = {c.id: c for c in fused}

        clustered = cluster_temporal_regions(fused)
        atomic = atomic_regions(fused)

        rank_baseline, hits_baseline = _baseline_rank_and_hits(group, clustered, candidates_by_id, event_ids)
        rank_baseline_atomic, hits_baseline_atomic = _baseline_rank_and_hits(group, atomic, candidates_by_id, event_ids)

        events = events_with_relations_from_cache(relations_cache.get(group.video_id), event_ids)
        constraints = build_order_constraints(events) if events is not None else None

        row: dict[str, Any] = {
            "video_id": group.video_id,
            "event_count": len(event_ids),
            "unique_video_count": len({r.video_id for r in clustered}),
            "rank_baseline": rank_baseline,
            "hits_baseline": hits_baseline,
            "rank_baseline_atomic": rank_baseline_atomic,
            "hits_baseline_atomic": hits_baseline_atomic,
            "order_constraints": constraints,
            "variants": {},
        }
        for config in SWEEP_CONFIGS:
            regions = atomic if config.use_atomic_regions else clustered
            pairs = constraints if config.use_temporal_relation else None
            ranking, tuples_by_video = rank_videos_by_region_tuples(
                regions, candidates_by_id, event_ids, config.params, pairs
            )
            rank_tuple = next(
                (pos for pos, (vid, _) in enumerate(ranking, 1)
                 if canonical_video_id(vid) == group.video_id),
                None,
            )
            hits_tuple = _tuple_gt_hits(group, tuples_by_video, event_ids)
            row["variants"][config.label] = {"rank": rank_tuple, "hits": hits_tuple}

        rows.append(row)
        if index % progress_every == 0 or index == len(groups):
            elapsed = time.monotonic() - started
            print(f"  {index}/{len(groups)} videos done ({elapsed:.0f}s elapsed)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/query")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-limit", type=int, default=30)
    parser.add_argument("--temporal-relations-cache", default=None)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    run_sweep(
        query_dir=Path(args.query_dir),
        output_path=Path(args.output),
        video_limit=args.video_limit,
        temporal_relations_cache_path=(
            Path(args.temporal_relations_cache) if args.temporal_relations_cache else None
        ),
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
