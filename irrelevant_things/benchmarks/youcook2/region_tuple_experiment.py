"""Ablation sweep: does multi-region pooling + order-aware tuple scoring
(region_tuple_ranking.py) beat today's production `prioritize_videos()`
(independent per-event argmax) on real YouCook2 queries?

For each of n sample videos, fetch real upstream candidates ONCE, fuse and
cluster ONCE (clustering hyperparameters are proven invariant to this
question - see clustering_impact_benchmark.py in scratch/ - so they're held
at defaults throughout), then compute:

  - rank_baseline: prioritize_videos()'s rank for the ground-truth video.
  - hits_baseline: how many of the GT video's own events land inside their
    ground-truth interval, using today's independent per-event best region
    (select_event_seeds()'s own seed[0], the production seed).
  - for every swept RegionTupleParams config: rank_tuple and hits_tuple,
    same two questions, answered by the new algorithm instead.

Both metrics are computed from the SAME fused/clustered candidate set for a
paired, noise-free before/after comparison - no repeated network calls once
data is fetched. Metrics reported with the repo's existing recall@k_new /
final_query_score definitions (boundary_metrics.py) for direct comparability
with prior benchmark docs.

Several configs use `temporal_relation` from `build_temporal_relations_cache.py`'s
cached real LLM classification (--temporal-relations-cache) to build real
directional order constraints (see region_tuple_ranking.py's
`order_constraints`): for each event with relation "after"/"before" and a
`reference_event_id`, a (predecessor, successor) pair is derived from that
relation's actual direction - NOT from adjacent list position. This matters:
an event listed earlier in the query array is not guaranteed to be the
expected *predecessor* in time (a query can describe events out of
chronological order, or - as happened on this specific corpus - always
describe them in order; the constraint-building code must not assume either
case). Events with relation "independent"/"simultaneous" (or no reference)
contribute no constraint at all, matching an opt-out from today's blanket
adjacent-pair check, not an opt-in - it only ever removes or redirects
penalties, it doesn't invent stricter ones a query didn't ask for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .boundary_metrics import DEFAULT_KS, frame_hits
from .coarse_anchor import WINNER_REFINEMENT_PARAMS, WINNER_RETRIEVAL_PARAMS, WINNER_UPSTREAM_TOP_K
from .core import VideoQueryGroup, canonical_video_id, load_query_directory_grouped
from .region_tuple_ranking import RegionTupleParams, rank_videos_by_region_tuples
from adaptive_search.algorithms import cluster_temporal_regions, prioritize_videos
from adaptive_search.boundary_seeds import select_event_seeds
from adaptive_search.client import QueryVariant
from adaptive_search.dependencies import upstream_search_client
from adaptive_search.retrieval import fuse_candidates_rrf

DEFAULT_PARAMS = RegionTupleParams()
# The N^event_count blowup only bites once pools are large; N<=20's default
# cap of 20000 already truncates some real videos (20^4=160000 > 20000).
# Raised generously for the N-scaling ablation so a larger N is measured on
# its own merits, not on how much of the truncated search space it explored.
_LARGE_N_MAX_COMBINATIONS = 500_000


@dataclass(frozen=True)
class SweepConfig:
    label: str
    params: RegionTupleParams
    use_temporal_relation: bool = False


SWEEP_CONFIGS: list[SweepConfig] = [
    SweepConfig("baseline_equivalent (order_weight=0)", replace(DEFAULT_PARAMS, order_weight=0.0)),
    SweepConfig("default", DEFAULT_PARAMS),
    SweepConfig("delta=0.05", replace(DEFAULT_PARAMS, relative_delta=0.05)),
    SweepConfig("delta=0.10", replace(DEFAULT_PARAMS, relative_delta=0.10)),
    SweepConfig("delta=0.25", replace(DEFAULT_PARAMS, relative_delta=0.25)),
    SweepConfig("delta=0.40", replace(DEFAULT_PARAMS, relative_delta=0.40)),
    SweepConfig("N=1 (degenerate, no pooling)", replace(DEFAULT_PARAMS, max_regions_per_event=1)),
    SweepConfig("N=2", replace(DEFAULT_PARAMS, max_regions_per_event=2)),
    SweepConfig("N=3", replace(DEFAULT_PARAMS, max_regions_per_event=3)),
    SweepConfig("N=5", replace(DEFAULT_PARAMS, max_regions_per_event=5)),
    SweepConfig("N=10", replace(DEFAULT_PARAMS, max_regions_per_event=10)),
    SweepConfig("N=20 (proposal's own cap)", replace(DEFAULT_PARAMS, max_regions_per_event=20)),
    SweepConfig("N=30", replace(DEFAULT_PARAMS, max_regions_per_event=30, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("N=50", replace(DEFAULT_PARAMS, max_regions_per_event=50, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("N=100", replace(DEFAULT_PARAMS, max_regions_per_event=100, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("N=200", replace(DEFAULT_PARAMS, max_regions_per_event=200, max_combinations_per_video=_LARGE_N_MAX_COMBINATIONS)),
    SweepConfig("order_weight=0.05", replace(DEFAULT_PARAMS, order_weight=0.05)),
    SweepConfig("order_weight=0.2", replace(DEFAULT_PARAMS, order_weight=0.2)),
    SweepConfig("order_weight=0.4", replace(DEFAULT_PARAMS, order_weight=0.4)),
    SweepConfig("order_weight=0.8", replace(DEFAULT_PARAMS, order_weight=0.8)),
    SweepConfig("pooling=mean", replace(DEFAULT_PARAMS, pooling="mean")),
    # --- temporal_relation gating (needs --temporal-relations-cache) ---
    SweepConfig("temporal_relation-gated (default)", DEFAULT_PARAMS, use_temporal_relation=True),
    SweepConfig("temporal_relation-gated, order_weight=0.4", replace(DEFAULT_PARAMS, order_weight=0.4), use_temporal_relation=True),
    SweepConfig("temporal_relation-gated, order_weight=0.8", replace(DEFAULT_PARAMS, order_weight=0.8), use_temporal_relation=True),
    SweepConfig("temporal_relation-gated, pooling=mean", replace(DEFAULT_PARAMS, pooling="mean"), use_temporal_relation=True),
    # --- confidence gating ---
    SweepConfig("confidence_gate=linear, order_weight=0.8", replace(DEFAULT_PARAMS, order_weight=0.8, confidence_gate="linear")),
    SweepConfig("confidence_gate=threshold@0.3, order_weight=0.8", replace(DEFAULT_PARAMS, order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.3)),
    SweepConfig("confidence_gate=threshold@0.5, order_weight=0.8", replace(DEFAULT_PARAMS, order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.5)),
    SweepConfig("confidence_gate=threshold@0.7, order_weight=0.8", replace(DEFAULT_PARAMS, order_weight=0.8, confidence_gate="threshold", confidence_gate_threshold=0.7)),
    # --- combined: temporal_relation + confidence gating together ---
    SweepConfig(
        "temporal_relation + confidence_gate=linear, order_weight=0.8",
        replace(DEFAULT_PARAMS, order_weight=0.8, confidence_gate="linear"),
        use_temporal_relation=True,
    ),
    SweepConfig(
        "temporal_relation + pooling=mean + confidence_gate=linear, order_weight=0.8",
        replace(DEFAULT_PARAMS, order_weight=0.8, pooling="mean", confidence_gate="linear"),
        use_temporal_relation=True,
    ),
]


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


def order_constraints_from_relations(
    events: list[dict[str, Any]] | None, event_count: int
) -> list[tuple[int, int]] | None:
    """(predecessor_index, successor_index) pairs derived from each event's
    real `relation`/`reference_event_id`, direction included - NOT from
    adjacent list position (see module docstring and region_tuple_ranking.py
    `_order_score`'s docstring for why that distinction is load-bearing).

    "after" with reference r on event i: expect t(r) < t(i) -> (r, i).
    "before" with reference r on event i: expect t(i) < t(r) -> (i, r).
    "sequence_start"/"during"/"simultaneous"/"independent"/"unknown", or a
    null reference: no constraint from this event.

    None (no cached classification for this video, e.g. the rewrite call
    failed) falls back to the adjacent-list-position chain, i.e. today's
    blanket behavior - see `_order_score`'s own None handling."""

    if events is None or len(events) != event_count:
        return None
    constraints: list[tuple[int, int]] = []
    for event in events:
        relation = event["relation"]
        reference = event["reference_event_id"]
        index = event["event_id"]
        if reference is None:
            continue
        if relation == "after":
            constraints.append((reference, index))
        elif relation == "before":
            constraints.append((index, reference))
    return constraints


def _baseline_gt_hits(
    group: VideoQueryGroup, regions, candidates_by_id, event_ids: list[str]
) -> int:
    anchors: dict[str, float] = {}
    for event_id in event_ids:
        seeds = select_event_seeds(
            regions=regions, candidates_by_id=candidates_by_id,
            event_id=event_id, video_id=group.video_id,
        )
        if seeds:
            anchors[event_id] = seeds[0]
    return frame_hits(group, anchors)


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
        regions = cluster_temporal_regions(fused)
        candidates_by_id = {c.id: c for c in fused}

        priorities = prioritize_videos(regions, event_ids, WINNER_REFINEMENT_PARAMS)
        rank_baseline = next(
            (pos for pos, p in enumerate(priorities, 1)
             if canonical_video_id(p.video_id) == group.video_id),
            None,
        )
        unique_video_count = len(priorities)
        hits_baseline = _baseline_gt_hits(group, regions, candidates_by_id, event_ids)

        constraints = order_constraints_from_relations(relations_cache.get(group.video_id), len(event_ids))

        row: dict[str, Any] = {
            "video_id": group.video_id,
            "event_count": len(event_ids),
            "unique_video_count": unique_video_count,
            "rank_baseline": rank_baseline,
            "hits_baseline": hits_baseline,
            "order_constraints": constraints,
            "variants": {},
        }
        for config in SWEEP_CONFIGS:
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
