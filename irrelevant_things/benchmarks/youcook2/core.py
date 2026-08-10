"""Dataset parsing, video aggregation, and retrieval metrics.

This module deliberately has no dependency on the demo code that produced the
local YouCook2 assets.  It only understands exported query/annotation files and
the public JSON contract of the frame-search API.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v")
EVENT_RE = re.compile(r"^\s*(E\d+)\s*:\s*(.*?)\s*$", re.IGNORECASE)
ANSWER_RE = re.compile(r"^\s*\*\*\s*Answer\s*$", re.IGNORECASE)
VIDEO_PATH_RE = re.compile(r'^\s*video_path\s*:\s*["\']?(.*?)["\']?\s*$', re.IGNORECASE)
TIME_RANGE_RE = re.compile(
    r"^\s*(E\d+)\s*:\s*(\d+(?::\d+){1,2}(?:\.\d+)?)\s*-\s*"
    r"(\d+(?::\d+){1,2}(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


class DatasetFormatError(ValueError):
    """Raised when benchmark input would make evaluation ambiguous."""


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_text: str
    ground_truth_video: str
    source: str
    event_id: str | None = None
    context: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("query_id", "query_text", "ground_truth_video", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DatasetFormatError(f"{name} must be a non-empty string")
        if self.event_id is not None and (
            not isinstance(self.event_id, str) or not self.event_id.strip()
        ):
            raise DatasetFormatError("event_id must be a non-empty string when set")
        for name in ("start_seconds", "end_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise DatasetFormatError(
                    f"{name} must be finite and non-negative when set"
                )
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise DatasetFormatError("end_seconds must not precede start_seconds")

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "ground_truth_video": self.ground_truth_video,
            "source": self.source,
            "event_id": self.event_id,
            "context": self.context,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RankedVideo:
    video_id: str
    score: float
    hit_count: int
    best_frame_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "score": self.score,
            "hit_count": self.hit_count,
            "best_frame_score": self.best_frame_score,
        }


def canonical_video_id(value: str) -> str:
    """Return a comparable video id without changing case-sensitive IDs."""

    cleaned = str(value).strip().strip('"\'').replace("\\", "/").rstrip("/")
    if not cleaned:
        raise DatasetFormatError("video id/path is empty")
    name = cleaned.rsplit("/", 1)[-1]
    lower = name.lower()
    for suffix in VIDEO_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise DatasetFormatError(f"invalid timestamp: {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise DatasetFormatError(f"invalid timestamp: {value!r}") from exc
    if any(number < 0 for number in numbers) or numbers[-1] >= 60:
        raise DatasetFormatError(f"invalid timestamp: {value!r}")
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if numbers[1] >= 60:
        raise DatasetFormatError(f"invalid timestamp: {value!r}")
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def _parse_query_file(path: Path, query_mode: str) -> list[QueryRecord]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    before_answer = True
    context_lines: list[str] = []
    events: list[tuple[str, str]] = []
    answers: dict[str, tuple[float, float]] = {}
    video_path: str | None = None

    for line in lines:
        if before_answer and ANSWER_RE.match(line):
            before_answer = False
            continue
        if before_answer:
            match = EVENT_RE.match(line)
            if match:
                events.append((match.group(1).upper(), match.group(2).strip()))
            elif line.strip():
                context_lines.append(line.strip())
            continue

        match = VIDEO_PATH_RE.match(line)
        if match:
            video_path = match.group(1).strip()
            continue
        match = TIME_RANGE_RE.match(line)
        if match:
            start = parse_timestamp(match.group(2))
            end = parse_timestamp(match.group(3))
            if end < start:
                raise DatasetFormatError(f"{path}: end precedes start for {match.group(1)}")
            answers[match.group(1).upper()] = (start, end)

    if not events:
        raise DatasetFormatError(f"{path}: no query event found before **Answer")
    if not video_path:
        raise DatasetFormatError(f"{path}: missing video_path in answer section")
    if len({event_id for event_id, _ in events}) != len(events):
        raise DatasetFormatError(f"{path}: duplicate event id")

    video_id = canonical_video_id(video_path)
    context = " ".join(context_lines).strip() or None
    missing_answers = [event_id for event_id, _ in events if event_id not in answers]
    if missing_answers:
        raise DatasetFormatError(f"{path}: missing answers for {', '.join(missing_answers)}")

    if query_mode == "file":
        parts = ([context] if context else []) + [text for _, text in events]
        return [
            QueryRecord(
                query_id=f"{video_id}:all",
                query_text=" ".join(parts),
                ground_truth_video=video_id,
                source=str(path),
                context=context,
                metadata={"event_count": len(events)},
            )
        ]
    if query_mode not in {"event", "event_with_context"}:
        raise ValueError(f"unknown query_mode: {query_mode}")

    records: list[QueryRecord] = []
    for event_id, event_text in events:
        start, end = answers[event_id]
        retrieval_text = event_text
        if query_mode == "event_with_context" and context:
            retrieval_text = f"{context} {event_text}"
        records.append(
            QueryRecord(
                query_id=f"{video_id}:{event_id}",
                query_text=retrieval_text,
                ground_truth_video=video_id,
                source=str(path),
                event_id=event_id,
                context=context,
                start_seconds=start,
                end_seconds=end,
            )
        )
    return records


def _ensure_unique_query_ids(records: Sequence[QueryRecord]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.query_id in seen:
            duplicates.add(record.query_id)
        seen.add(record.query_id)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise DatasetFormatError(f"duplicate query_id values: {sample}")


def load_query_directory(path: str | Path, query_mode: str = "event") -> list[QueryRecord]:
    directory = Path(path)
    if not directory.is_dir():
        raise DatasetFormatError(f"query directory does not exist: {directory}")
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise DatasetFormatError(f"no .txt query files found in {directory}")
    records = [record for file_path in files for record in _parse_query_file(file_path, query_mode)]
    _ensure_unique_query_ids(records)
    return records


def _record_from_mapping(row: Mapping[str, Any], source: str, row_number: int) -> QueryRecord:
    query_id = row.get("query_id") or row.get("id")
    query_text = row.get("query_text") or row.get("query") or row.get("text")
    gt_video = (
        row.get("ground_truth_video")
        or row.get("video_id")
        or row.get("video_name")
        or row.get("video_path")
    )
    if not query_id or not query_text or not gt_video:
        raise DatasetFormatError(
            f"{source} row {row_number}: query_id, query text, and ground-truth video are required"
        )
    start = row.get("start_seconds")
    end = row.get("end_seconds")
    return QueryRecord(
        query_id=str(query_id),
        query_text=str(query_text).strip(),
        ground_truth_video=canonical_video_id(str(gt_video)),
        source=source,
        event_id=str(row["event_id"]) if row.get("event_id") is not None else None,
        context=str(row["context"]) if row.get("context") else None,
        start_seconds=float(start) if start not in (None, "") else None,
        end_seconds=float(end) if end not in (None, "") else None,
    )


def load_query_manifest(path: str | Path) -> list[QueryRecord]:
    manifest = Path(path)
    if not manifest.is_file():
        raise DatasetFormatError(f"query manifest does not exist: {manifest}")
    suffix = manifest.suffix.lower()
    rows: list[Mapping[str, Any]]
    if suffix == ".jsonl":
        rows = []
        for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise DatasetFormatError(f"{manifest} row {number}: expected JSON object")
                rows.append(value)
    elif suffix == ".json":
        value = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if isinstance(value, Mapping):
            value = value.get("queries")
        if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
            raise DatasetFormatError(f"{manifest}: expected a list or an object with a queries list")
        rows = value
    elif suffix == ".csv":
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise DatasetFormatError("query manifest must be .jsonl, .json, or .csv")
    records = [_record_from_mapping(row, str(manifest), number) for number, row in enumerate(rows, 1)]
    if not records:
        raise DatasetFormatError(f"query manifest is empty: {manifest}")
    _ensure_unique_query_ids(records)
    return records


def load_official_annotations(
    path: str | Path,
    subsets: Iterable[str] | None = None,
) -> list[QueryRecord]:
    """Create English event queries from the official YouCook2 annotation JSON."""

    annotation_path = Path(path)
    value = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    database = value.get("database") if isinstance(value, Mapping) else None
    if not isinstance(database, Mapping):
        raise DatasetFormatError(f"{annotation_path}: missing database object")
    subset_filter = {item.lower() for item in subsets or []}
    records: list[QueryRecord] = []
    for video_id, video in sorted(database.items()):
        if not isinstance(video, Mapping):
            continue
        subset = str(video.get("subset", "")).lower()
        if subset_filter and subset not in subset_filter:
            continue
        annotations = video.get("annotations")
        if not isinstance(annotations, list):
            continue
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, Mapping) or not annotation.get("sentence"):
                continue
            event_id = str(annotation.get("id", index))
            segment = annotation.get("segment")
            start = end = None
            if isinstance(segment, list) and len(segment) == 2:
                start, end = float(segment[0]), float(segment[1])
            records.append(
                QueryRecord(
                    query_id=f"{video_id}:{event_id}",
                    query_text=str(annotation["sentence"]).strip(),
                    ground_truth_video=canonical_video_id(str(video_id)),
                    source=str(annotation_path),
                    event_id=event_id,
                    start_seconds=start,
                    end_seconds=end,
                    metadata={"subset": subset, "recipe_type": video.get("recipe_type")},
                )
            )
    if not records:
        raise DatasetFormatError(f"{annotation_path}: no matching annotations")
    _ensure_unique_query_ids(records)
    return records


def aggregate_video_hits(
    hits: Sequence[Mapping[str, Any]],
    method: str = "max",
    top_m: int = 3,
    temperature: float = 1.0,
) -> list[RankedVideo]:
    """Collapse frame hits into a deterministic unique-video ranking."""

    if top_m < 1:
        raise ValueError("top_m must be at least 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    grouped: dict[str, list[float]] = {}
    for index, hit in enumerate(hits):
        if not isinstance(hit, Mapping):
            raise ValueError(f"search result {index} is not an object")
        raw_video = hit.get("video_name") or hit.get("video_id") or hit.get("video") or hit.get("path")
        if not raw_video:
            raise ValueError(f"search result {index} has no video_name/video_id")
        try:
            score = float(hit["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"search result {index} has an invalid score") from exc
        if not math.isfinite(score):
            raise ValueError(f"search result {index} has a non-finite score")
        grouped.setdefault(canonical_video_id(str(raw_video)), []).append(score)

    ranked: list[RankedVideo] = []
    for video_id, scores in grouped.items():
        descending = sorted(scores, reverse=True)
        if method == "max":
            aggregate = descending[0]
        elif method == "top_m_mean":
            selected = descending[:top_m]
            aggregate = sum(selected) / len(selected)
        elif method == "logsumexp":
            maximum = descending[0]
            aggregate = maximum + temperature * math.log(
                sum(math.exp((score - maximum) / temperature) for score in descending)
            )
        else:
            raise ValueError(f"unknown aggregation method: {method}")
        ranked.append(
            RankedVideo(
                video_id=video_id,
                score=aggregate,
                hit_count=len(scores),
                best_frame_score=descending[0],
            )
        )
    return sorted(ranked, key=lambda item: (-item.score, item.video_id))


def rank_of(ranked_videos: Sequence[RankedVideo], ground_truth_video: str) -> int | None:
    target = canonical_video_id(ground_truth_video)
    for index, video in enumerate(ranked_videos, 1):
        if video.video_id == target:
            return index
    return None


def compute_metrics(results: Sequence[Mapping[str, Any]], recall_ks: Sequence[int]) -> dict[str, Any]:
    ks = sorted(set(int(k) for k in recall_ks))
    if not ks or ks[0] < 1:
        raise ValueError("recall_ks must contain positive integers")
    successful = [row for row in results if row.get("status") == "ok"]
    errors = [row for row in results if row.get("status") != "ok"]
    ranks = [int(row["rank"]) if row.get("rank") is not None else None for row in successful]
    all_ranks = [
        int(row["rank"])
        if row.get("status") == "ok" and row.get("rank") is not None
        else None
        for row in results
    ]
    recalls = {
        f"recall_at_{k}": (
            sum(rank is not None and rank <= k for rank in all_ranks)
            / len(all_ranks)
            if all_ranks
            else None
        )
        for k in ks
    }
    successful_recalls = {
        f"recall_at_{k}_successful_requests": (
            sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
            if ranks
            else None
        )
        for k in ks
    }
    reciprocal_ranks = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
    all_reciprocal_ranks = [
        1.0 / rank if rank is not None else 0.0 for rank in all_ranks
    ]
    found_ranks = [rank for rank in ranks if rank is not None]
    unique_video_counts = [int(row.get("unique_video_count", 0)) for row in successful]
    # A missing item is only known to rank after the returned unique candidates.
    # This is an explicit optimistic imputation for a truncated list, not a
    # statistical censored median; report it alongside the found-only median.
    imputed_ranks = [
        rank if rank is not None else int(row.get("unique_video_count", 0)) + 1
        for row, rank in zip(successful, ranks)
    ]
    latencies = [float(row["latency_ms"]) for row in successful if row.get("latency_ms") is not None]
    return {
        "query_count": len(results),
        "evaluated_count": len(successful),
        "error_count": len(errors),
        "completion_rate": len(successful) / len(results) if results else None,
        **recalls,
        **successful_recalls,
        "mrr": (
            sum(all_reciprocal_ranks) / len(all_reciprocal_ranks)
            if all_reciprocal_ranks
            else None
        ),
        "mrr_successful_requests": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else None
        ),
        "median_rank": statistics.median(imputed_ranks) if imputed_ranks else None,
        "median_rank_definition": (
            "optimistic truncated-list imputation: missing ground truth is "
            "assigned unique_video_count + 1"
        ),
        "median_rank_found": statistics.median(found_ranks) if found_ranks else None,
        "ground_truth_found_count": len(found_ranks),
        "ground_truth_miss_count": len(successful) - len(found_ranks),
        "unique_video_count_min": min(unique_video_counts) if unique_video_counts else None,
        "unique_video_count_median": statistics.median(unique_video_counts) if unique_video_counts else None,
        "unique_video_count_mean": (
            sum(unique_video_counts) / len(unique_video_counts) if unique_video_counts else None
        ),
        "unique_video_count_max": max(unique_video_counts) if unique_video_counts else None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }
