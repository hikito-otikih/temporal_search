"""Orchestrate the moment-oriented boundary-refinement experiment.

For each of (legacy_temporal, legacy_ambiguous, adaptive_coarse) x (first, last
moment) x query: compute the pipeline's "before" state (video rank + a
per-event anchor timestamp), then run the local-sweep boundary refiner
(boundary_refinement.py) to get the "after" state, and compare recall@k_new
before vs after.

Correctness/compute-saving result relied on here: refinement only relocates a
timestamp *within* an already-identified video - it never changes video rank.
So `rank` is identical before and after refinement for a given query; only
`frame_hits` can differ. That means recall@k_new can only differ between
conditions for queries whose "before" rank is <= 100 (the largest k), so the
(GPU-bound) refinement call is skipped entirely for any query where rank is
None or > 100 - its contribution to every reported k is provably 0 either way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .augmented_query_generator import generate_augmented_queries
from .boundary_metrics import DEFAULT_KS, aggregate_boundary_metrics, frame_hits
from .boundary_refinement import build_runtime, refine_event_boundary
from .coarse_anchor import adaptive_coarse_before
from .core import (
    VideoQueryGroup,
    canonical_video_id,
    legacy_event_timestamps,
    load_query_directory_grouped,
)
from .tuple_client import TemporalSearchBackendClient
from adaptive_search.client import UpstreamSearchClient

Pipeline = Literal["legacy_temporal", "legacy_ambiguous", "adaptive_coarse"]
MomentType = Literal["first", "last"]

_LEGACY_SEARCHER_TYPES: dict[Pipeline, str] = {
    "legacy_temporal": "TemporalSearcher",
    "legacy_ambiguous": "AmbiguousSearcher",
}
_PIPELINES: tuple[Pipeline, ...] = ("legacy_temporal", "legacy_ambiguous", "adaptive_coarse")
_BOUNDARY_TYPE: dict[MomentType, str] = {"first": "onset", "last": "offset"}
_RANK_SKIP_THRESHOLD = max(DEFAULT_KS)


def _pre_post_state(action_text: str, moment_type: MomentType) -> tuple[str, str]:
    if moment_type == "first":
        return (f"{action_text} vẫn chưa bắt đầu.", f"{action_text} đang xảy ra.")
    return (f"{action_text} vẫn đang xảy ra.", f"{action_text} vừa mới kết thúc.")


# Verified directly against the live backend: at the CLI's default 100/100,
# TemporalSearcher's per-video backtracking requires ALL 4 events to have a
# candidate frame within the top-100 per-event pool (in valid temporal
# order) - real 4-event queries in this dataset routinely find zero videos
# satisfying that at 100/100 (confirmed: a 1-event query returns 100 real
# hits, but the same video's full 4-event query returns 0 results at
# top_k_each_query=100, and still doesn't surface the GT video's rank even
# at top_k_tuple=500/top_k_each_query=2000). Use a generous-but-bounded
# top_k so legacy pipelines get a fair shot without making each call
# prohibitively slow at n=30 videos.
_LEGACY_TOP_K_TUPLE = 200
_LEGACY_TOP_K_EACH_QUERY = 1000


def _legacy_before(
    backend: TemporalSearchBackendClient, group: VideoQueryGroup, pipeline: Pipeline
) -> tuple[int | None, dict[str, float]]:
    results = backend.legacy_temporal_search(
        [text for _, text in group.events],
        top_k_tuple=_LEGACY_TOP_K_TUPLE,
        top_k_each_query=_LEGACY_TOP_K_EACH_QUERY,
        gamma=0.05,
        searcher_type=_LEGACY_SEARCHER_TYPES[pipeline],
    )
    for position, item in enumerate(results, 1):
        if canonical_video_id(item.video_name) == group.video_id:
            return position, legacy_event_timestamps(group, item.frames)
    return None, {}


def _primary_anchors(event_anchors: Mapping[str, float | Sequence[float]]) -> dict[str, float]:
    """Normalize either shape (legacy's single float, coarse's seed tuple) to one anchor per
    event - the best-scoring (first) seed, matching the pre-multi-seed max-only semantics.
    Used for the "before" hit-rate baseline, so it stays comparable across pipelines and to
    the original single-anchor report."""

    return {
        event_id: (value if isinstance(value, (int, float)) else value[0])
        for event_id, value in event_anchors.items()
    }


def _refine_all_events(
    *,
    provider: Any,
    embedder: Any,
    augmented_group: VideoQueryGroup,
    source_group: VideoQueryGroup,
    moment_type: MomentType,
    event_anchors: Mapping[str, float | Sequence[float]],
) -> tuple[dict[str, float], int, int]:
    """Returns (refined_anchors, refined_count, fallback_count)."""

    source_text_by_event = dict(source_group.events)
    boundary_type = _BOUNDARY_TYPE[moment_type]
    metadata = provider.catalog.metadata(augmented_group.video_id)
    if metadata is None or metadata.fps is None:
        # No metadata for this video -> nothing to refine against; leave anchors as-is.
        return _primary_anchors(event_anchors), 0, len(event_anchors)

    # refine_event_boundary() takes a single anchor - it fine-tunes within one
    # neighborhood, it doesn't adjudicate between several candidate ones (see
    # boundary_refinement.py's docstring for why not). Multi-seed inputs are
    # collapsed to the best-ranked seed here, same as the "before" baseline.
    refined: dict[str, float] = {}
    fallback_count = 0
    for event_id, anchor_seconds in _primary_anchors(event_anchors).items():
        action_text = source_text_by_event[event_id]
        pre_state, post_state = _pre_post_state(action_text, moment_type)
        outcome = refine_event_boundary(
            provider=provider,
            embedder=embedder,
            video_id=augmented_group.video_id,
            event_id=event_id,
            anchor_query=action_text,
            pre_state=pre_state,
            post_state=post_state,
            boundary_type=boundary_type,
            anchor_seconds=anchor_seconds,
            fps=metadata.fps,
            duration_ms=metadata.duration_ms,
        )
        refined[event_id] = outcome.refined_seconds
        if outcome.used_fallback:
            fallback_count += 1
    return refined, len(event_anchors), fallback_count


def run_experiment(
    *,
    source_query_dir: Path,
    augmented_dir: Path,
    output_dir: Path,
    video_limit: int | None,
    backend_base_url: str,
    upstream_base_url: str,
    progress_every: int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "rows.jsonl"

    print(f"Generating augmented queries from {source_query_dir} -> {augmented_dir}")
    summary = generate_augmented_queries(source_query_dir, augmented_dir)
    print(f"  {summary['video_count']} source videos -> augmented")

    source_groups = load_query_directory_grouped(source_query_dir)
    if video_limit is not None:
        source_groups = source_groups[:video_limit]
    video_ids = {group.video_id: group for group in source_groups}
    print(f"Running experiment on {len(video_ids)} source videos "
          f"({2 * len(video_ids)} augmented queries)")

    augmented_by_moment: dict[MomentType, dict[str, VideoQueryGroup]] = {}
    for moment_type in ("first", "last"):
        groups = load_query_directory_grouped(augmented_dir / moment_type)
        augmented_by_moment[moment_type] = {
            group.video_id: group for group in groups if group.video_id in video_ids
        }

    backend = TemporalSearchBackendClient(
        backend_base_url, timeout_seconds=60.0, retries=2, retry_backoff_seconds=1.0
    )
    upstream = UpstreamSearchClient(upstream_base_url, timeout_seconds=60.0)
    provider, embedder = build_runtime()

    done: set[tuple[str, str, str]] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["pipeline"], row["moment_type"], row["video_id"]))
        print(f"Resuming: {len(done)} (pipeline, moment_type, video_id) rows already done")

    handle = checkpoint_path.open("a", encoding="utf-8")
    completed = 0
    skipped_refinement = 0
    total = len(_PIPELINES) * 2 * len(video_ids)
    started = time.monotonic()

    try:
        for moment_type in ("first", "last"):
            for pipeline in _PIPELINES:
                for video_id, source_group in video_ids.items():
                    key = (pipeline, moment_type, video_id)
                    if key in done:
                        continue
                    augmented_group = augmented_by_moment[moment_type][video_id]

                    # Retrieval uses source_group (base action text), not augmented_group
                    # (the "khoảnh khắc đầu tiên/cuối cùng"-wrapped text) - verified directly
                    # against the live upstream service that the wrapper measurably distorts
                    # candidate ranking (a near-tied correct candidate dropped from #1 to #2).
                    # source_group and augmented_group share the same video_id/event_ids/answers,
                    # differing only in event text, so this is a safe substitution.
                    if pipeline == "adaptive_coarse":
                        result = adaptive_coarse_before(source_group, upstream=upstream)
                        rank, anchors_before = result.rank, result.event_anchors
                    else:
                        rank, anchors_before = _legacy_before(backend, source_group, pipeline)

                    # frame_hits() needs one float per event - normalize adaptive_coarse's
                    # multi-seed tuples down to the top seed for a fair, before/after- and
                    # legacy-comparable baseline (legacy structurally only ever has one).
                    hits_before = frame_hits(augmented_group, _primary_anchors(anchors_before))

                    if rank is None or rank > _RANK_SKIP_THRESHOLD:
                        # Cannot affect any recall@k_new(k<=100) value either way - skip refinement.
                        hits_after = hits_before
                        refined_count = 0
                        fallback_count = 0
                        skipped_refinement += 1
                    else:
                        anchors_after, refined_count, fallback_count = _refine_all_events(
                            provider=provider,
                            embedder=embedder,
                            augmented_group=augmented_group,
                            source_group=source_group,
                            moment_type=moment_type,
                            event_anchors=anchors_before,
                        )
                        hits_after = frame_hits(augmented_group, anchors_after)

                    row = {
                        "pipeline": pipeline,
                        "moment_type": moment_type,
                        "video_id": video_id,
                        "rank": rank,
                        "hits_before": hits_before,
                        "hits_after": hits_after,
                        "event_count": len(source_group.events),
                        "refined_event_count": refined_count,
                        "fallback_count": fallback_count,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed += 1
                    if completed % progress_every == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"  {completed}/{total - len(done)} new rows done "
                            f"({elapsed:.0f}s elapsed, {skipped_refinement} refinement-skipped)"
                        )
    finally:
        handle.close()

    print(f"Done. {completed} new rows written, {skipped_refinement} skipped refinement "
          f"(rank None or > {_RANK_SKIP_THRESHOLD}).")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-query-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/query")
    parser.add_argument("--augmented-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/augmented_query")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-limit", type=int, default=None)
    parser.add_argument("--backend-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--upstream-base-url", default="http://172.26.176.1:8000")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    run_experiment(
        source_query_dir=Path(args.source_query_dir),
        augmented_dir=Path(args.augmented_dir),
        output_dir=Path(args.output_dir),
        video_limit=args.video_limit,
        backend_base_url=args.backend_base_url,
        upstream_base_url=args.upstream_base_url,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
