"""Checkpointed execution for the tuple-level (whole-video) ablation benchmark.

Unlike `runner.py` (one independent retrieval request per event, ranking that
event's frame hits), this evaluates the thing the product actually returns:
one ordered tuple per video, ranked, and checked against the ground-truth
`video_id`. Reuses `runner.py`'s checkpoint/resume/atomic-write machinery and
`core.py`'s `compute_metrics` recall/MRR aggregation as-is - both are already
generic over "rows with a rank", so nothing needed duplicating there.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .core import (
    VideoQueryGroup,
    canonical_video_id,
    compute_metrics,
    event_timestamp_accuracy,
    legacy_event_timestamps,
)
from .runner import (
    _append_checkpoint,
    _atomic_json,
    _load_checkpoint,
    fingerprint_files,
    utc_now,
)
from .tuple_client import TemporalSearchBackendClient


SCHEMA_VERSION = "youcook2-tuple-recall/v1"

TuplePipeline = Literal["legacy_temporal", "legacy_ambiguous", "adaptive_coarse"]

_LEGACY_SEARCHER_TYPES: dict[str, str] = {
    "legacy_temporal": "TemporalSearcher",
    "legacy_ambiguous": "AmbiguousSearcher",
}


@dataclass(frozen=True)
class TupleRunConfig:
    backend_base_url: str
    pipeline: TuplePipeline
    top_k_tuple: int
    top_k_each_query: int
    gamma: float
    adaptive_top_k: int
    recall_ks: tuple[int, ...]
    timeout_seconds: float
    retries: int
    retry_backoff_seconds: float
    # Caps the size of the ranked list GET .../video-priorities returns -
    # independent of --adaptive-top-k (the raw per-variant retrieval cap).
    # Raise this if you want e.g. Recall@50 to mean anything. None keeps
    # the server's own default (100, max 1000).
    adaptive_ranking_top_k: int | None = None
    # hyperparameters.retrieval overrides (None keeps the server's own
    # default for that field). top_n_per_variant/top_n_fused/rrf_k feed
    # fuse_candidates_rrf().
    adaptive_top_n_per_variant: int | None = None
    adaptive_top_n_fused: int | None = None
    adaptive_rrf_k: int | None = None
    # hyperparameters.refinement overrides - the three weights that blend
    # into GET .../video-priorities' priority_score alongside
    # video_distinctness_weight (router/video_priorities.py).
    adaptive_video_coverage_weight: float | None = None
    adaptive_video_mean_weight: float | None = None
    adaptive_video_min_weight: float | None = None


def source_fingerprint(source_path: Path) -> tuple[str, int]:
    files = sorted(source_path.glob("*.txt")) if source_path.is_dir() else [source_path]
    return fingerprint_files(files), len(files)


def evaluation_fingerprint(
    file_digest: str,
    source_description: Mapping[str, Any],
    groups: Sequence[VideoQueryGroup],
) -> str:
    """Bind resume compatibility to the exact video groups and source options."""

    payload = {
        "file_digest": file_digest,
        "source": dict(source_description),
        "groups": [
            {
                "video_id": group.video_id,
                "events": list(group.events),
                "answers": {key: list(value) for key, value in group.answers.items()},
            }
            for group in groups
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_compatible(existing: Mapping[str, Any], config: TupleRunConfig, fingerprint: str) -> bool:
    existing_config = json.dumps(existing.get("config"), sort_keys=True, separators=(",", ":"))
    requested_config = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return (
        existing.get("schema_version") == SCHEMA_VERSION
        and existing_config == requested_config
        and existing.get("source_fingerprint") == fingerprint
    )


def _build_event_payload(group: VideoQueryGroup) -> list[dict[str, Any]]:
    return [
        {"event_id": event_id, "original_query": text, "anchor_query": text}
        for event_id, text in group.events
    ]


def _adaptive_hyperparameters(config: TupleRunConfig) -> dict[str, Any] | None:
    """Build a session hyperparameters override, or None to send nothing."""

    retrieval = {
        key: value
        for key, value in (
            ("top_n_per_variant", config.adaptive_top_n_per_variant),
            ("top_n_fused", config.adaptive_top_n_fused),
            ("rrf_k", config.adaptive_rrf_k),
        )
        if value is not None
    }
    refinement = {
        key: value
        for key, value in (
            ("video_coverage_weight", config.adaptive_video_coverage_weight),
            ("video_mean_weight", config.adaptive_video_mean_weight),
            ("video_min_weight", config.adaptive_video_min_weight),
        )
        if value is not None
    }
    hyperparameters: dict[str, Any] = {}
    if retrieval:
        hyperparameters["retrieval"] = retrieval
    if refinement:
        hyperparameters["refinement"] = refinement
    return hyperparameters or None


def _rank_of_video(items: Sequence[Mapping[str, Any]], video_key: str, target: str) -> int | None:
    for position, item in enumerate(items, 1):
        if canonical_video_id(str(item[video_key])) == target:
            return position
    return None


def _run_legacy(
    backend: TemporalSearchBackendClient, group: VideoQueryGroup, config: TupleRunConfig
) -> dict[str, Any]:
    outcome = backend.legacy_temporal_search(
        [text for _, text in group.events],
        top_k_tuple=config.top_k_tuple,
        top_k_each_query=config.top_k_each_query,
        gamma=config.gamma,
        searcher_type=_LEGACY_SEARCHER_TYPES[config.pipeline],
    )
    results = outcome.results
    rank = None
    matched_frames: tuple[Mapping[str, Any], ...] | None = None
    for position, item in enumerate(results, 1):
        if canonical_video_id(item.video_name) == group.video_id:
            rank = position
            matched_frames = item.frames
            break
    row: dict[str, Any] = {
        "status": "ok",
        "rank": rank,
        "unique_video_count": len({canonical_video_id(item.video_name) for item in results}),
        "top_score": results[0].score if results else None,
        # Surfaced (not silently dropped) so a run summary can flag it -
        # results are the best found so far, not guaranteed complete, and
        # a clean-looking recall/rank summary would otherwise hide that.
        "search_truncated": outcome.search_truncated,
    }
    if matched_frames is not None:
        row["event_timestamp_accuracy"] = event_timestamp_accuracy(
            group, legacy_event_timestamps(group, matched_frames)
        )
    return row


def _run_adaptive_coarse(
    backend: TemporalSearchBackendClient, group: VideoQueryGroup, config: TupleRunConfig
) -> dict[str, Any]:
    session_id = backend.create_adaptive_session(
        _build_event_payload(group),
        hyperparameters=_adaptive_hyperparameters(config),
    )
    backend.retrieve(session_id, top_k=config.adaptive_top_k)
    priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)
    rank = _rank_of_video(priorities, "video_id", group.video_id)
    return {
        "status": "ok",
        "rank": rank,
        "unique_video_count": len(priorities),
        "session_id": session_id,
        "top_score": priorities[0]["priority_score"] if priorities else None,
    }


_PIPELINE_RUNNERS = {
    "legacy_temporal": _run_legacy,
    "legacy_ambiguous": _run_legacy,
    "adaptive_coarse": _run_adaptive_coarse,
}


def _result_csv(path: Path, rows: Sequence[Mapping[str, Any]], recall_ks: Sequence[int]) -> None:
    columns = [
        "video_id",
        "pipeline",
        "event_count",
        "rank",
        "reciprocal_rank",
        *[f"hit_at_{k}" for k in recall_ks],
        "unique_video_count",
        "latency_ms",
        "status",
        "error",
        "source",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            rank = int(row["rank"]) if row.get("rank") is not None else None
            writer.writerow(
                {
                    "video_id": row.get("video_id"),
                    "pipeline": row.get("pipeline"),
                    "event_count": row.get("event_count"),
                    "rank": rank,
                    "reciprocal_rank": 1.0 / rank if rank else 0.0,
                    **{f"hit_at_{k}": int(rank is not None and rank <= k) for k in recall_ks},
                    "unique_video_count": row.get("unique_video_count"),
                    "latency_ms": row.get("latency_ms"),
                    "status": row.get("status"),
                    "error": row.get("error"),
                    "source": row.get("source"),
                }
            )
    temporary.replace(path)


def run_tuple_benchmark(
    groups: Sequence[VideoQueryGroup],
    output_dir: str | Path,
    config: TupleRunConfig,
    source_description: Mapping[str, Any],
    source_digest: str,
    resume: bool = False,
    force_resume: bool = False,
    client: TemporalSearchBackendClient | None = None,
    progress_every: int = 10,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if not groups:
        raise ValueError("no video query groups to evaluate")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "query_results.jsonl"
    manifest_path = directory / "run_manifest.json"
    metrics_path = directory / "metrics.json"
    csv_path = directory / "query_results.csv"

    if force_resume:
        raise ValueError(
            "--force-resume is disabled because it can mix incompatible runs; "
            "choose a new output directory"
        )

    manifest_exists = manifest_path.exists()
    checkpoint_exists = checkpoint_path.exists()
    if manifest_exists != checkpoint_exists:
        raise ValueError(
            "output directory has only one of run_manifest.json and "
            "query_results.jsonl; use a new output directory"
        )
    if resume and not (manifest_exists and checkpoint_exists):
        raise ValueError("--resume requires both run_manifest.json and query_results.jsonl")
    if checkpoint_exists and not resume:
        raise FileExistsError(
            f"{checkpoint_path} exists; use --resume or choose a new output directory"
        )

    backend = client or TemporalSearchBackendClient(
        base_url=config.backend_base_url,
        timeout_seconds=config.timeout_seconds,
        retries=config.retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )

    existing_manifest: Mapping[str, Any] | None = None
    if manifest_exists:
        decoded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(decoded_manifest, Mapping):
            raise ValueError(f"run manifest must be a JSON object: {manifest_path}")
        existing_manifest = decoded_manifest
    if resume and existing_manifest is not None and not _manifest_compatible(
        existing_manifest, config, source_digest
    ):
        raise ValueError(
            "resume schema/configuration/source differs from run_manifest.json; "
            "choose a new output directory"
        )

    latest = _load_checkpoint(checkpoint_path) if resume else {}
    started_at = existing_manifest.get("started_at") if existing_manifest else utc_now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": started_at,
        "updated_at": utc_now(),
        "source": dict(source_description),
        "source_fingerprint": source_digest,
        "group_count": len(groups),
        "config": asdict(config),
        "ground_truth_sent_to_backend": False,
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    _atomic_json(manifest_path, manifest)

    # A group is "done" once it has a final status; "unavailable" (e.g. live
    # refinement not configured) is a deterministic environment fact, not a
    # transient failure, so resume must not keep retrying it either.
    completed = {
        video_id for video_id, row in latest.items() if row.get("status") in ("ok", "unavailable")
    }
    pending = [group for group in groups if group.video_id not in completed]
    print(
        f"YouCook2 tuple eval ({config.pipeline}): {len(groups)} videos, "
        f"{len(completed)} resumed, {len(pending)} pending"
    )

    runner = _PIPELINE_RUNNERS[config.pipeline]
    with checkpoint_path.open("a", encoding="utf-8", buffering=1) as checkpoint:
        for index, group in enumerate(pending, 1):
            common: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                # _load_checkpoint() keys rows by "query_id" - reused as-is,
                # so every row needs one even though it holds a video id here.
                "query_id": group.video_id,
                "video_id": group.video_id,
                "pipeline": config.pipeline,
                "event_count": len(group.events),
                "source": group.source,
                "started_at": utc_now(),
            }
            started = time.perf_counter()
            try:
                outcome = runner(backend, group, config)
                row = {**common, "error": None, **outcome}
            except Exception as exc:  # keep the long benchmark resumable
                row = {
                    **common,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "rank": None,
                    "unique_video_count": 0,
                }
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["completed_at"] = utc_now()
            _append_checkpoint(checkpoint, row)
            latest[group.video_id] = row
            if progress_every > 0 and (index % progress_every == 0 or index == len(pending)):
                print(f"completed {index}/{len(pending)} pending videos")

    ordered_results = [latest[group.video_id] for group in groups]
    metric_rows = [row for row in ordered_results if row.get("status") != "unavailable"]
    unavailable_count = len(ordered_results) - len(metric_rows)
    metrics = compute_metrics(metric_rows, config.recall_ks)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline": config.pipeline,
            "recall_ks": list(config.recall_ks),
            "unavailable_count": unavailable_count,
            "unavailable_note": (
                "groups with status=unavailable are excluded from recall/MRR "
                "above; check unavailable_count before comparing this pipeline "
                "against others"
            ),
        }
    )
    _atomic_json(metrics_path, metrics)
    _result_csv(csv_path, ordered_results, config.recall_ks)
    manifest.update(
        {
            "status": "completed" if metrics["error_count"] == 0 else "completed_with_errors",
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "metrics_file": metrics_path.name,
            "results_jsonl_file": checkpoint_path.name,
            "results_csv_file": csv_path.name,
            "metrics": metrics,
        }
    )
    _atomic_json(manifest_path, manifest)
    return metrics, ordered_results
