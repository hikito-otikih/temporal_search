"""Checkpointed execution and reproducible benchmark artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .client import SearchClient
from .core import QueryRecord, aggregate_video_hits, compute_metrics, rank_of


SCHEMA_VERSION = "youcook2-video-recall/v2"

# Health endpoints often include liveness fields that change on every request.
# Only fields capable of changing retrieval output belong in the resume key.
_BACKEND_IDENTITY_TOKENS = (
    "model",
    "revision",
    "preprocess",
    "processor",
    "index",
    "corpus",
    "dataset",
    "vector",
    "store",
    "dimension",
    "dtype",
    "device",
    "translator",
    "translation",
)


@dataclass(frozen=True)
class RunConfig:
    base_url: str
    frame_top_k: int
    recall_ks: tuple[int, ...]
    aggregation: str
    top_m: int
    temperature: float
    timeout_seconds: float
    retries: int
    retry_backoff_seconds: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def source_fingerprint(source_path: Path) -> tuple[str, int]:
    files = sorted(source_path.glob("*.txt")) if source_path.is_dir() else [source_path]
    return fingerprint_files(files), len(files)


def evaluation_fingerprint(
    file_digest: str,
    source_description: Mapping[str, Any],
    records: Sequence[QueryRecord],
) -> str:
    """Bind resume compatibility to the exact records and source options."""

    payload = {
        "file_digest": file_digest,
        "source": dict(source_description),
        "queries": [record.as_dict() for record in records],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def backend_identity(health: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a stable resume fingerprint from retrieval-relevant health fields.

    A backend should expose an immutable model revision and index/content hash for
    paper runs. The current API exposes a smaller identity (model, dtype, store,
    and vector count), which is still safer than binding only to the base URL.
    """

    selected = {
        str(key): value
        for key, value in sorted(health.items(), key=lambda item: str(item[0]))
        if any(token in str(key).lower() for token in _BACKEND_IDENTITY_TOKENS)
    }
    has_model_identity = any(
        "model" in key.lower() or "revision" in key.lower() for key in selected
    )
    has_index_identity = any(
        any(token in key.lower() for token in ("index", "corpus", "dataset", "vector", "store"))
        for key in selected
    )
    if not has_model_identity or not has_index_identity:
        raise ValueError(
            "GET /health must expose stable model and index identity fields "
            "(for example model plus index_hash or n_vectors)"
        )
    encoded = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), selected


def _load_checkpoint(path: Path) -> dict[str, Mapping[str, Any]]:
    """Load a checkpoint and repair only an interrupted final write."""

    latest: dict[str, Mapping[str, Any]] = {}
    if not path.exists():
        return latest

    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    valid_end = 0
    repair_end = len(payload)
    for line_number, raw_line in enumerate(lines, 1):
        line_end = valid_end + len(raw_line)
        if not raw_line.strip():
            valid_end = line_end
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            remaining = b"".join(lines[line_number:])
            if not remaining.strip():
                # A killed process can leave one partial tail record.  All
                # earlier records remain a valid checkpoint.
                repair_end = valid_end
                break
            raise ValueError(f"invalid checkpoint JSON at line {line_number}: {path}")
        if not isinstance(row, Mapping) or not row.get("query_id"):
            raise ValueError(
                f"invalid checkpoint record at line {line_number}: {path}"
            )
        latest[str(row["query_id"])] = row
        valid_end = line_end

    if repair_end < len(payload):
        with path.open("r+b") as handle:
            handle.truncate(repair_end)
            handle.flush()
            os.fsync(handle.fileno())
        payload = payload[:repair_end]
    if payload and payload[-1:] not in {b"\n", b"\r"}:
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return latest


def _append_checkpoint(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _result_csv(path: Path, rows: Sequence[Mapping[str, Any]], recall_ks: Sequence[int]) -> None:
    columns = [
        "query_id",
        "event_id",
        "query_text",
        "ground_truth_video",
        "rank",
        "reciprocal_rank",
        *[f"hit_at_{k}" for k in recall_ks],
        "raw_hit_count",
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
                    "query_id": row.get("query_id"),
                    "event_id": row.get("event_id"),
                    "query_text": row.get("query_text"),
                    "ground_truth_video": row.get("ground_truth_video"),
                    "rank": rank,
                    "reciprocal_rank": 1.0 / rank if rank else 0.0,
                    **{f"hit_at_{k}": int(rank is not None and rank <= k) for k in recall_ks},
                    "raw_hit_count": row.get("raw_hit_count"),
                    "unique_video_count": row.get("unique_video_count"),
                    "latency_ms": row.get("latency_ms"),
                    "status": row.get("status"),
                    "error": row.get("error"),
                    "source": row.get("source"),
                }
            )
    temporary.replace(path)


def _manifest_compatible(
    existing: Mapping[str, Any],
    config: RunConfig,
    fingerprint: str,
    backend_fingerprint: str,
) -> bool:
    existing_config = json.dumps(existing.get("config"), sort_keys=True, separators=(",", ":"))
    requested_config = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return (
        existing.get("schema_version") == SCHEMA_VERSION
        and existing_config == requested_config
        and existing.get("source_fingerprint") == fingerprint
        and existing.get("backend_fingerprint") == backend_fingerprint
    )


def run_benchmark(
    records: Sequence[QueryRecord],
    output_dir: str | Path,
    config: RunConfig,
    source_description: Mapping[str, Any],
    source_digest: str,
    resume: bool = False,
    force_resume: bool = False,
    client: SearchClient | None = None,
    progress_every: int = 10,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if not records:
        raise ValueError("no query records to evaluate")
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
        raise ValueError(
            "--resume requires both run_manifest.json and query_results.jsonl"
        )
    if checkpoint_exists and not resume:
        raise FileExistsError(f"{checkpoint_path} exists; use --resume or choose a new output directory")

    search_client = client or SearchClient(
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        retries=config.retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )
    health = search_client.health()
    backend_fingerprint, backend_identity_fields = backend_identity(health)

    existing_manifest: Mapping[str, Any] | None = None
    if manifest_exists:
        decoded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(decoded_manifest, Mapping):
            raise ValueError(f"run manifest must be a JSON object: {manifest_path}")
        existing_manifest = decoded_manifest
    if resume and existing_manifest is not None and not _manifest_compatible(
        existing_manifest,
        config,
        source_digest,
        backend_fingerprint,
    ):
        raise ValueError(
            "resume schema/configuration/source/backend identity differs from "
            "run_manifest.json; choose a new output directory"
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
        "query_count": len(records),
        "config": asdict(config),
        "request_payload_fields": ["query", "top_k"],
        "ground_truth_sent_to_backend": False,
        "api_health": dict(health),
        "backend_identity": backend_identity_fields,
        "backend_fingerprint": backend_fingerprint,
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    _atomic_json(manifest_path, manifest)

    completed_ok = {query_id for query_id, row in latest.items() if row.get("status") == "ok"}
    attempts = {query_id: int(row.get("attempt", 0)) for query_id, row in latest.items()}
    pending = [record for record in records if record.query_id not in completed_ok]
    print(
        f"YouCook2 Video Recall: {len(records)} queries, {len(completed_ok)} resumed, "
        f"{len(pending)} pending"
    )

    with checkpoint_path.open("a", encoding="utf-8", buffering=1) as checkpoint:
        for index, record in enumerate(pending, 1):
            attempt = attempts.get(record.query_id, 0) + 1
            common: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "query_id": record.query_id,
                "event_id": record.event_id,
                "query_text": record.query_text,
                "ground_truth_video": record.ground_truth_video,
                "ground_truth_start_seconds": record.start_seconds,
                "ground_truth_end_seconds": record.end_seconds,
                "source": record.source,
                "attempt": attempt,
                "started_at": utc_now(),
            }
            started = time.perf_counter()
            try:
                response = search_client.search(record.query_text, config.frame_top_k)
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                ranking = aggregate_video_hits(
                    response.hits,
                    method=config.aggregation,
                    top_m=config.top_m,
                    temperature=config.temperature,
                )
                rank = rank_of(ranking, record.ground_truth_video)
                row = {
                    **common,
                    "status": "ok",
                    "error": None,
                    "rank": rank,
                    "raw_hit_count": len(response.hits),
                    "unique_video_count": len(ranking),
                    "latency_ms": latency_ms,
                    "backend_query": response.response_query,
                    "backend_english_query": response.english_query,
                    "backend_top_k": response.response_top_k,
                    "ranked_videos": [video.as_dict() for video in ranking],
                }
            except Exception as exc:  # keep the long benchmark resumable
                row = {
                    **common,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "rank": None,
                    "raw_hit_count": 0,
                    "unique_video_count": 0,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "ranked_videos": [],
                }
            row["completed_at"] = utc_now()
            _append_checkpoint(checkpoint, row)
            latest[record.query_id] = row
            if progress_every > 0 and (index % progress_every == 0 or index == len(pending)):
                print(f"completed {index}/{len(pending)} pending queries")

    ordered_results = [latest[record.query_id] for record in records]
    metrics = compute_metrics(ordered_results, config.recall_ks)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "aggregation": config.aggregation,
            "frame_top_k": config.frame_top_k,
            "recall_ks": list(config.recall_ks),
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
