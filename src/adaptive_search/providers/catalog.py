"""Resolve opaque YouCook2 video IDs to on-disk paths and optional keyframe
metadata, under a recursively-scanned video root."""

from __future__ import annotations

import json
import math
import os
from numbers import Integral
from pathlib import Path
from threading import RLock
from typing import Iterator, Sequence

from dataclasses import dataclass

from .base import (
    DEFAULT_VIDEO_EXTENSIONS,
    YOUCOOK2_DATA_ROOT_ENV,
    FrameProviderConfigurationError,
    VideoAssetNotFoundError,
    _validate_video_id,
)


@dataclass(frozen=True)
class YouCook2VideoMetadata:
    """Validated subset of an optional YouCook2 keyframe manifest."""

    video_id: str
    duration_ms: int | None = None
    fps: float | None = None
    frame_count: int | None = None


class YouCook2AssetCatalog:
    """Resolve opaque video IDs under the recursive YouCook2 video tree."""

    def __init__(
        self,
        video_root: str | os.PathLike[str] | None,
        *,
        metadata_root: str | os.PathLike[str] | None = None,
        video_extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS,
    ) -> None:
        self._configured_video_root = _optional_path(video_root)
        self._configured_metadata_root = _optional_path(metadata_root)
        normalized_extensions: set[str] = set()
        for extension in video_extensions:
            if not isinstance(extension, str) or not extension.strip():
                raise FrameProviderConfigurationError(
                    "video extensions must be non-empty strings"
                )
            value = extension.strip().lower()
            normalized_extensions.add(value if value.startswith(".") else f".{value}")
        if not normalized_extensions:
            raise FrameProviderConfigurationError(
                "at least one video extension must be configured"
            )
        self._extensions = frozenset(normalized_extensions)
        self._lock = RLock()
        self._video_root: Path | None = None
        self._assets: dict[str, Path] = {}
        self._metadata_paths: dict[str, Path] = {}
        self._metadata_cache: dict[str, YouCook2VideoMetadata | None] = {}
        self._reason: str | None = None
        self.refresh()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def asset_count(self) -> int:
        with self._lock:
            return len(self._assets)

    @property
    def video_root(self) -> Path | None:
        with self._lock:
            return self._video_root

    def refresh(self) -> None:
        """Atomically rebuild the catalog after assets are added or removed."""

        try:
            video_root = self._resolve_video_root()
            assets: dict[str, Path] = {}
            for candidate in _walk_files(video_root):
                if candidate.suffix.lower() not in self._extensions:
                    continue
                resolved = _require_within_root(candidate, video_root)
                _insert_unique_path(
                    assets,
                    candidate.stem,
                    resolved,
                    kind="video",
                )
            if not assets:
                raise FrameProviderConfigurationError(
                    f"no supported videos found under {video_root}"
                )
            metadata_paths = self._scan_metadata_paths()
        except (OSError, FrameProviderConfigurationError) as exc:
            with self._lock:
                self._video_root = None
                self._assets = {}
                self._metadata_paths = {}
                self._metadata_cache = {}
                self._reason = str(exc)
            return

        with self._lock:
            self._video_root = video_root
            self._assets = assets
            self._metadata_paths = metadata_paths
            self._metadata_cache = {}
            self._reason = None

    def resolve(self, video_id: str) -> Path:
        normalized_id = _validate_video_id(video_id)
        with self._lock:
            reason = self._reason
            path = self._assets.get(normalized_id)
        if reason is not None:
            raise FrameProviderConfigurationError(reason)
        if path is None:
            raise VideoAssetNotFoundError(
                f"video_id {normalized_id!r} is not present in the YouCook2 catalog"
            )
        return path

    def metadata(self, video_id: str) -> YouCook2VideoMetadata | None:
        normalized_id = _validate_video_id(video_id)
        with self._lock:
            if normalized_id in self._metadata_cache:
                return self._metadata_cache[normalized_id]
            path = self._metadata_paths.get(normalized_id)
        if path is None:
            return None
        metadata = _read_youcook2_metadata(path, expected_video_id=normalized_id)
        with self._lock:
            self._metadata_cache[normalized_id] = metadata
        return metadata

    def _resolve_video_root(self) -> Path:
        configured = self._configured_video_root
        if configured is None:
            raise FrameProviderConfigurationError(
                f"video root is not configured; set {YOUCOOK2_DATA_ROOT_ENV}"
            )
        if not configured.exists():
            raise FrameProviderConfigurationError(
                f"configured video root does not exist: {configured}"
            )
        if not configured.is_dir():
            raise FrameProviderConfigurationError(
                f"configured video root is not a directory: {configured}"
            )
        nested_videos = configured / "videos"
        selected = nested_videos if nested_videos.is_dir() else configured
        return selected.resolve(strict=True)

    def _scan_metadata_paths(self) -> dict[str, Path]:
        configured = self._configured_metadata_root
        if configured is None:
            return {}
        if not configured.exists() or not configured.is_dir():
            raise FrameProviderConfigurationError(
                f"configured metadata root is not a directory: {configured}"
            )
        nested_metadata = configured / "metadata"
        selected = nested_metadata if nested_metadata.is_dir() else configured
        root = selected.resolve(strict=True)
        paths: dict[str, Path] = {}
        suffix = "_keyframes.json"
        for candidate in _walk_files(root, name_suffix=suffix):
            resolved = _require_within_root(candidate, root)
            video_id = candidate.name[: -len(suffix)]
            _insert_unique_path(paths, video_id, resolved, kind="metadata")
        return paths


def _walk_files(
    root: Path,
    *,
    name_suffix: str | None = None,
) -> Iterator[Path]:
    """Yield files deterministically without following directory symlinks."""

    for directory, directory_names, file_names in os.walk(
        root,
        followlinks=False,
    ):
        directory_names.sort()
        for file_name in sorted(file_names):
            if name_suffix is not None and not file_name.endswith(name_suffix):
                continue
            yield Path(directory) / file_name


def _optional_path(
    value: str | os.PathLike[str] | None,
) -> Path | None:
    if value is None:
        return None
    try:
        raw_value = os.fspath(value)
    except TypeError as exc:
        raise FrameProviderConfigurationError(
            "configured root must be a string or path-like value"
        ) from exc
    if not isinstance(raw_value, str):
        raise FrameProviderConfigurationError(
            "configured root must resolve to a text path"
        )
    if not raw_value.strip():
        return None
    return Path(raw_value).expanduser()


def _require_within_root(candidate: Path, root: Path) -> Path:
    if candidate.is_symlink():
        raise FrameProviderConfigurationError(
            f"catalog entry must not be a symlink: {candidate}"
        )
    absolute = Path(os.path.abspath(candidate))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise FrameProviderConfigurationError(
            f"catalog entry escapes configured root: {candidate}"
        ) from exc
    return absolute


def _insert_unique_path(
    paths: dict[str, Path],
    item_id: str,
    path: Path,
    *,
    kind: str,
) -> None:
    existing = paths.get(item_id)
    if existing is not None and existing != path:
        raise FrameProviderConfigurationError(
            f"duplicate YouCook2 {kind} ID {item_id!r}: "
            f"{existing} and {path}"
        )
    paths[item_id] = path


def _read_youcook2_metadata(
    path: Path,
    *,
    expected_video_id: str,
) -> YouCook2VideoMetadata:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrameProviderConfigurationError(
            f"could not read YouCook2 metadata {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FrameProviderConfigurationError(
            f"YouCook2 metadata must be a JSON object: {path}"
        )
    if payload.get("video_id") != expected_video_id:
        raise FrameProviderConfigurationError(
            f"metadata video_id mismatch in {path}: "
            f"expected {expected_video_id!r}"
        )

    duration_ms: int | None = None
    duration = payload.get("duration")
    if duration is not None:
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            raise FrameProviderConfigurationError(
                f"metadata duration must be finite and non-negative in {path}"
            )
        duration_ms = int(round(float(duration) * 1000.0))

    normalized_fps: float | None = None
    fps = payload.get("fps")
    if fps is not None:
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or fps <= 0
        ):
            raise FrameProviderConfigurationError(
                f"metadata fps must be finite and positive in {path}"
            )
        normalized_fps = float(fps)

    normalized_frame_count: int | None = None
    frame_count = payload.get("frame_count")
    if frame_count is not None:
        if (
            not isinstance(frame_count, Integral)
            or isinstance(frame_count, bool)
            or frame_count < 0
        ):
            raise FrameProviderConfigurationError(
                f"metadata frame_count must be a non-negative integer in {path}"
            )
        normalized_frame_count = int(frame_count)

    return YouCook2VideoMetadata(
        video_id=expected_video_id,
        duration_ms=duration_ms,
        fps=normalized_fps,
        frame_count=normalized_frame_count,
    )
