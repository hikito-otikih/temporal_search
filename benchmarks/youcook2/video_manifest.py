"""Optional audit manifest for videos expected to exist in the retrieval index."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .core import DatasetFormatError, canonical_video_id


def load_video_manifest(path: str | Path) -> set[str]:
    """Load indexed video ids from TXT, CSV, JSONL, or JSON."""

    manifest = Path(path)
    if not manifest.is_file():
        raise DatasetFormatError(f"video manifest does not exist: {manifest}")
    suffix = manifest.suffix.lower()
    values: list[Any]
    if suffix == ".txt":
        values = [line.strip() for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".csv":
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            values = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        values = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".json":
        parsed = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, Mapping):
            parsed = parsed.get("videos") or parsed.get("video_ids")
        if not isinstance(parsed, list):
            raise DatasetFormatError(f"{manifest}: expected a list or an object with videos/video_ids")
        values = parsed
    else:
        raise DatasetFormatError("video manifest must be .txt, .csv, .jsonl, or .json")

    video_ids: set[str] = set()
    for index, value in enumerate(values, 1):
        if isinstance(value, Mapping):
            value = (
                value.get("video_id")
                or value.get("video_name")
                or value.get("video_path")
                or value.get("path")
            )
        if value is None or not str(value).strip():
            raise DatasetFormatError(f"{manifest} row {index}: missing video id/path")
        video_ids.add(canonical_video_id(str(value)))
    if not video_ids:
        raise DatasetFormatError(f"video manifest is empty: {manifest}")
    return video_ids
