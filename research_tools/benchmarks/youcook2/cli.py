"""Command-line interface for YouCook2 corpus-level Video Recall@K."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .client import SearchApiError, SearchClient
from .core import (
    DatasetFormatError,
    QueryRecord,
    VideoQueryGroup,
    load_official_annotations,
    load_query_directory,
    load_query_directory_grouped,
    load_query_manifest,
)
from .runner import RunConfig, evaluation_fingerprint, run_benchmark, source_fingerprint
from .tuple_client import BackendApiError
from .tuple_runner import (
    TupleRunConfig,
    evaluation_fingerprint as tuple_evaluation_fingerprint,
    run_tuple_benchmark,
    source_fingerprint as tuple_source_fingerprint,
)
from .video_manifest import load_video_manifest


def _configure_utf8_console() -> None:
    """Make Vietnamese output safe on legacy Windows console code pages."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (AttributeError, OSError, ValueError):
                # StringIO, redirected streams, and some test doubles cannot be
                # reconfigured. They already accept Unicode strings directly.
                pass


def _csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parsed = tuple(int(item) for item in value)
    else:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or min(parsed) < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return tuple(sorted(set(parsed)))


def _defaults_from_config(argv: Sequence[str]) -> dict[str, Any]:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    known, _ = preliminary.parse_known_args(argv)
    if not known.config:
        return {}
    value = json.loads(Path(known.config).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("config JSON must be an object")
    return {str(key).replace("-", "_"): item for key, item in value.items()}


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.youcook2",
        description="Evaluate corpus-level YouCook2 Video Recall@K through POST /search.",
    )
    parser.add_argument("--config", help="JSON defaults; explicit CLI arguments take precedence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health", help="validate GET /health")
    health.add_argument("--base-url", default=defaults.get("base_url", "http://127.0.0.1:8000"))
    health.add_argument("--timeout", type=float, default=defaults.get("timeout", 10.0))
    health.add_argument("--retries", type=int, default=defaults.get("retries", 0))

    run = subparsers.add_parser("run", help="run or resume Video Recall@K evaluation")
    run.add_argument("--query-dir", default=defaults.get("query_dir"))
    run.add_argument("--query-manifest", default=defaults.get("query_manifest"))
    run.add_argument("--annotations-json", default=defaults.get("annotations_json"))
    run.add_argument(
        "--annotation-subset",
        action="append",
        default=defaults.get("annotation_subset", []),
        help="official subset to include; repeat for multiple subsets",
    )
    run.add_argument(
        "--query-mode",
        choices=("event", "event_with_context", "file"),
        default=defaults.get("query_mode", "event"),
        help="retrieval unit for local query TXT files",
    )
    run.add_argument("--video-manifest", default=defaults.get("video_manifest"))
    run.add_argument(
        "--missing-ground-truth",
        choices=("error", "skip", "keep"),
        default=defaults.get("missing_ground_truth", "error"),
        help="policy when GT is absent from --video-manifest",
    )
    run.add_argument("--output-dir", default=defaults.get("output_dir"))
    run.add_argument("--base-url", default=defaults.get("base_url", "http://127.0.0.1:8000"))
    run.add_argument("--frame-top-k", type=int, default=defaults.get("frame_top_k", 200))
    run.add_argument("--recall-k", type=_csv_ints, default=_csv_ints(defaults.get("recall_k", "1,5,10,20,50")))
    run.add_argument(
        "--aggregation",
        choices=("max", "top_m_mean", "logsumexp"),
        default=defaults.get("aggregation", "max"),
    )
    run.add_argument("--top-m", type=int, default=defaults.get("top_m", 3))
    run.add_argument("--temperature", type=float, default=defaults.get("temperature", 1.0))
    run.add_argument("--timeout", type=float, default=defaults.get("timeout", 30.0))
    run.add_argument("--retries", type=int, default=defaults.get("retries", 2))
    run.add_argument("--retry-backoff", type=float, default=defaults.get("retry_backoff", 0.5))
    run.add_argument("--limit", type=int, default=defaults.get("limit"), help="deterministic first-N smoke run")
    run.add_argument("--resume", action="store_true", default=defaults.get("resume", False))
    run.add_argument(
        "--force-resume",
        action="store_true",
        default=defaults.get("force_resume", False),
        help=(
            "deprecated safety flag: incompatible runs are never mixed; "
            "use a new output directory"
        ),
    )
    run.add_argument("--dry-run", action="store_true", default=defaults.get("dry_run", False))
    run.add_argument("--progress-every", type=int, default=defaults.get("progress_every", 10))

    tuple_run = subparsers.add_parser(
        "tuple-run",
        help="run or resume the tuple-level (whole-video) ablation benchmark",
    )
    tuple_run.add_argument(
        "--query-dir",
        default=defaults.get("query_dir"),
        help="directory of YouCook2 query TXT files (the only supported source)",
    )
    tuple_run.add_argument(
        "--pipeline",
        choices=("legacy_temporal", "legacy_ambiguous", "adaptive_coarse", "adaptive_full"),
        default=defaults.get("pipeline"),
    )
    tuple_run.add_argument(
        "--backend-base-url",
        default=defaults.get("backend_base_url", "http://127.0.0.1:8001"),
        help="this repo's own FastAPI app (src/main.py), not the sparse-search service",
    )
    tuple_run.add_argument("--top-k-tuple", type=int, default=defaults.get("top_k_tuple", 100))
    tuple_run.add_argument("--top-k-each-query", type=int, default=defaults.get("top_k_each_query", 100))
    tuple_run.add_argument("--gamma", type=float, default=defaults.get("gamma", 0.05))
    tuple_run.add_argument("--adaptive-top-k", type=int, default=defaults.get("adaptive_top_k", 20))
    tuple_run.add_argument(
        "--adaptive-max-frames",
        type=int,
        default=defaults.get("adaptive_max_frames"),
        help=(
            "cap commands/refine's frame budget for adaptive_full (server default is "
            "2000 across the whole frontier - real video decode + GPU embedding at that "
            "scale can take minutes per session; try 100-400 for benchmark turnaround)"
        ),
    )
    tuple_run.add_argument(
        "--adaptive-ranking-top-k",
        type=int,
        default=defaults.get("adaptive_ranking_top_k"),
        help=(
            "cap the size of whichever ranked list the chosen --pipeline produces - "
            "independent of --adaptive-top-k (the raw per-variant retrieval cap). "
            "adaptive_full: hyperparameters.ranking.top_k (server default 20). "
            "adaptive_coarse: the video-priorities page limit (server default 100, "
            "max 1000). No effect on legacy_temporal/legacy_ambiguous."
        ),
    )
    tuple_run.add_argument(
        "--adaptive-top-n-per-variant",
        type=int,
        default=defaults.get("adaptive_top_n_per_variant"),
        help=(
            "hyperparameters.retrieval.top_n_per_variant - truncates each query "
            "variant's fused-candidate input after --adaptive-top-k's raw upstream "
            "fetch (server default 200). Only binds if --adaptive-top-k is larger."
        ),
    )
    tuple_run.add_argument(
        "--adaptive-top-n-fused",
        type=int,
        default=defaults.get("adaptive_top_n_fused"),
        help="hyperparameters.retrieval.top_n_fused - cap on RRF-fused candidates per event, across every candidate video combined (server default 500).",
    )
    tuple_run.add_argument(
        "--adaptive-rrf-k",
        type=int,
        default=defaults.get("adaptive_rrf_k"),
        help="hyperparameters.retrieval.rrf_k - the RRF fusion constant (server default 60).",
    )
    tuple_run.add_argument(
        "--adaptive-video-coverage-weight",
        type=float,
        default=defaults.get("adaptive_video_coverage_weight"),
        help="hyperparameters.refinement.video_coverage_weight for prioritize_videos() (server default 0.5).",
    )
    tuple_run.add_argument(
        "--adaptive-video-mean-weight",
        type=float,
        default=defaults.get("adaptive_video_mean_weight"),
        help="hyperparameters.refinement.video_mean_weight for prioritize_videos() (server default 0.3).",
    )
    tuple_run.add_argument(
        "--adaptive-video-min-weight",
        type=float,
        default=defaults.get("adaptive_video_min_weight"),
        help="hyperparameters.refinement.video_min_weight for prioritize_videos() (server default 0.2).",
    )
    tuple_run.add_argument(
        "--adaptive-max-initial-videos",
        type=int,
        default=defaults.get("adaptive_max_initial_videos"),
        help=(
            "hyperparameters.refinement.max_initial_videos - how many coarse-ranked "
            "videos get full multi-region frontier coverage in adaptive_full "
            "(server default 100). No effect on adaptive_coarse."
        ),
    )
    tuple_run.add_argument(
        "--adaptive-max-total-regions",
        type=int,
        default=defaults.get("adaptive_max_total_regions"),
        help=(
            "hyperparameters.refinement.max_total_regions - the actual budget "
            "bottleneck behind --adaptive-max-initial-videos in adaptive_full; "
            "raising videos without raising this does little. No effect on "
            "adaptive_coarse."
        ),
    )
    tuple_run.add_argument(
        "--adaptive-max-regions-per-event",
        type=int,
        default=defaults.get("adaptive_max_regions_per_event_per_video"),
        help="hyperparameters.refinement.max_regions_per_event_per_video for adaptive_full's frontier. No effect on adaptive_coarse.",
    )
    tuple_run.add_argument(
        "--adaptive-max-frames-per-run",
        type=int,
        default=defaults.get("adaptive_max_frames_per_run"),
        help=(
            "hyperparameters.refinement.max_frames_per_run - the session's own "
            "hard ceiling that --adaptive-max-frames (commands/refine's request) "
            "is not allowed to exceed. Raise this alongside --adaptive-max-frames, "
            "not instead of it - they are different knobs."
        ),
    )
    tuple_run.add_argument(
        "--recall-k", type=_csv_ints, default=_csv_ints(defaults.get("recall_k", "1,5,10,20,50"))
    )
    tuple_run.add_argument("--output-dir", default=defaults.get("output_dir"))
    tuple_run.add_argument("--limit", type=int, default=defaults.get("limit"), help="deterministic first-N smoke run")
    tuple_run.add_argument("--resume", action="store_true", default=defaults.get("resume", False))
    tuple_run.add_argument(
        "--force-resume",
        action="store_true",
        default=defaults.get("force_resume", False),
        help=(
            "deprecated safety flag: incompatible runs are never mixed; "
            "use a new output directory"
        ),
    )
    tuple_run.add_argument("--dry-run", action="store_true", default=defaults.get("dry_run", False))
    tuple_run.add_argument("--timeout", type=float, default=defaults.get("timeout", 60.0))
    tuple_run.add_argument("--retries", type=int, default=defaults.get("retries", 2))
    tuple_run.add_argument("--retry-backoff", type=float, default=defaults.get("retry_backoff", 0.5))
    tuple_run.add_argument("--progress-every", type=int, default=defaults.get("progress_every", 10))
    return parser


def _load_records(args: argparse.Namespace) -> tuple[list[QueryRecord], Path, dict[str, Any]]:
    specified = [value for value in (args.query_dir, args.query_manifest, args.annotations_json) if value]
    if len(specified) != 1:
        raise DatasetFormatError(
            "select exactly one source: --query-dir, --query-manifest, or --annotations-json"
        )
    if args.query_dir:
        source_path = Path(args.query_dir)
        records = load_query_directory(source_path, query_mode=args.query_mode)
        description = {"type": "youcook2_query_txt", "path": str(source_path), "query_mode": args.query_mode}
    elif args.query_manifest:
        source_path = Path(args.query_manifest)
        records = load_query_manifest(source_path)
        description = {"type": "query_manifest", "path": str(source_path)}
    else:
        source_path = Path(args.annotations_json)
        records = load_official_annotations(source_path, subsets=args.annotation_subset)
        description = {
            "type": "official_annotations",
            "path": str(source_path),
            "subsets": list(args.annotation_subset),
        }

    if args.video_manifest:
        available = load_video_manifest(args.video_manifest)
        missing = [record for record in records if record.ground_truth_video not in available]
        description["video_manifest"] = str(args.video_manifest)
        description["video_manifest_count"] = len(available)
        description["ground_truth_missing_count"] = len(missing)
        if missing and args.missing_ground_truth == "error":
            sample = ", ".join(record.ground_truth_video for record in missing[:5])
            raise DatasetFormatError(
                f"{len(missing)} query GT videos are absent from video manifest; examples: {sample}"
            )
        if args.missing_ground_truth == "skip":
            missing_ids = {record.query_id for record in missing}
            records = [record for record in records if record.query_id not in missing_ids]
            description["ground_truth_skipped_count"] = len(missing)
    if args.limit is not None:
        if args.limit < 1:
            raise DatasetFormatError("--limit must be positive")
        records = records[: args.limit]
        description["limit"] = args.limit
    return records, source_path, description


def _load_groups(args: argparse.Namespace) -> tuple[list[VideoQueryGroup], Path, dict[str, Any]]:
    if not args.query_dir:
        raise DatasetFormatError("tuple-run requires --query-dir (the only supported source)")
    source_path = Path(args.query_dir)
    groups = load_query_directory_grouped(source_path)
    description: dict[str, Any] = {
        "type": "youcook2_query_txt_grouped",
        "path": str(source_path),
    }
    if args.limit is not None:
        if args.limit < 1:
            raise DatasetFormatError("--limit must be positive")
        groups = groups[: args.limit]
        description["limit"] = args.limit
    return groups, source_path, description


def _run_tuple_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.pipeline:
        parser.error("tuple-run requires --pipeline")

    groups, source_path, description = _load_groups(args)
    file_digest, source_file_count = tuple_source_fingerprint(source_path)
    description["source_file_count"] = source_file_count
    description["loaded_group_count"] = len(groups)
    digest = tuple_evaluation_fingerprint(file_digest, description, groups)
    plan = {
        "source": description,
        "source_file_fingerprint": file_digest,
        "source_fingerprint": digest,
        "group_count": len(groups),
        "sample_groups": [
            {"video_id": group.video_id, "events": list(group.events)} for group in groups[:3]
        ],
        "backend_base_url": args.backend_base_url,
        "pipeline": args.pipeline,
        "ground_truth_sent_to_backend": False,
        "recall_ks": list(args.recall_k),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not args.output_dir:
        parser.error("tuple-run requires --output-dir unless --dry-run is used")

    config = TupleRunConfig(
        backend_base_url=args.backend_base_url.rstrip("/."),
        pipeline=args.pipeline,
        top_k_tuple=args.top_k_tuple,
        top_k_each_query=args.top_k_each_query,
        gamma=args.gamma,
        adaptive_top_k=args.adaptive_top_k,
        recall_ks=tuple(args.recall_k),
        timeout_seconds=args.timeout,
        retries=args.retries,
        retry_backoff_seconds=args.retry_backoff,
        adaptive_max_frames=args.adaptive_max_frames,
        adaptive_ranking_top_k=args.adaptive_ranking_top_k,
        adaptive_top_n_per_variant=args.adaptive_top_n_per_variant,
        adaptive_top_n_fused=args.adaptive_top_n_fused,
        adaptive_rrf_k=args.adaptive_rrf_k,
        adaptive_video_coverage_weight=args.adaptive_video_coverage_weight,
        adaptive_video_mean_weight=args.adaptive_video_mean_weight,
        adaptive_video_min_weight=args.adaptive_video_min_weight,
        adaptive_max_initial_videos=args.adaptive_max_initial_videos,
        adaptive_max_total_regions=args.adaptive_max_total_regions,
        adaptive_max_regions_per_event_per_video=args.adaptive_max_regions_per_event,
        adaptive_max_frames_per_run=args.adaptive_max_frames_per_run,
    )
    metrics, _ = run_tuple_benchmark(
        groups=groups,
        output_dir=args.output_dir,
        config=config,
        source_description=description,
        source_digest=digest,
        resume=args.resume,
        force_resume=args.force_resume,
        progress_every=args.progress_every,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["error_count"] == 0 else 2


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_console()
    argv = list(argv if argv is not None else sys.argv[1:])
    try:
        defaults = _defaults_from_config(argv)
        parser = build_parser(defaults)
        args = parser.parse_args(argv)
        if args.command == "health":
            client = SearchClient(args.base_url, timeout_seconds=args.timeout, retries=args.retries)
            print(json.dumps(client.health(), ensure_ascii=False, indent=2))
            return 0

        if args.command == "tuple-run":
            return _run_tuple_command(args, parser)

        records, source_path, description = _load_records(args)
        file_digest, source_file_count = source_fingerprint(source_path)
        description["source_file_count"] = source_file_count
        description["loaded_query_count"] = len(records)
        digest = evaluation_fingerprint(file_digest, description, records)
        plan = {
            "source": description,
            "source_file_fingerprint": file_digest,
            "source_fingerprint": digest,
            "query_count": len(records),
            "sample_queries": [record.as_dict() for record in records[:3]],
            "backend": args.base_url,
            "search_request_schema": {"query": "<query_text_only>", "top_k": args.frame_top_k},
            "ground_truth_sent_to_backend": False,
            "aggregation": args.aggregation,
            "recall_ks": list(args.recall_k),
        }
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if not args.output_dir:
            parser.error("run requires --output-dir unless --dry-run is used")
        if args.frame_top_k < 1 or args.top_m < 1 or args.temperature <= 0:
            parser.error("frame-top-k/top-m must be positive and temperature must be greater than zero")
        if max(args.recall_k) > args.frame_top_k:
            print(
                "warning: max Recall@K exceeds frame-top-k; duplicate frame hits may leave fewer unique videos",
                file=sys.stderr,
            )
        config = RunConfig(
            base_url=args.base_url.rstrip("/."),
            frame_top_k=args.frame_top_k,
            recall_ks=tuple(args.recall_k),
            aggregation=args.aggregation,
            top_m=args.top_m,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff,
        )
        metrics, _ = run_benchmark(
            records=records,
            output_dir=args.output_dir,
            config=config,
            source_description=description,
            source_digest=digest,
            resume=args.resume,
            force_resume=args.force_resume,
            progress_every=args.progress_every,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0 if metrics["error_count"] == 0 else 2
    except (
        DatasetFormatError,
        FileNotFoundError,
        FileExistsError,
        SearchApiError,
        BackendApiError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
