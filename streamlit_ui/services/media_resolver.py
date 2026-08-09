"""Resolve local video/keyframe media for preview in metadata-only mode.

The UI must remain fully functional when media is missing.  Every resolver
returns ``None`` (not an exception) when the file does not exist so pages can
render scores/metadata and mark the preview as unavailable.

Supported layouts (YouCook2 and generic):

* videos: ``<root>/videos/<split>/<category>/<video_id>.mp4`` or any file
  named ``<video_id>`` under ``<root>/videos``.
* keyframes: ``<root>/keyframes/<video_id>/<video_id>_f<frame_index>.webp``
  or ``<root>/keyframes/<video_id>/<frame_index>.jpg``.

When running inside WSL and the configured root is a Windows path
(``C:\\...``), it is transparently translated to ``/mnt/<drive>/...``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")
_IMAGE_EXTS = (".webp", ".jpg", ".jpeg", ".png")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _normalize_root(media_root: str | Path | None) -> Path | None:
    if not media_root:
        return None
    root = Path(str(media_root).strip().strip('"'))
    if os.name == "posix" and _WINDOWS_PATH.match(str(root)):
        drive = str(root)[0].lower()
        remainder = str(root)[2:].lstrip("\\/").replace("\\", "/")
        root = Path(f"/mnt/{drive}") / remainder
    return root.expanduser()


class MediaResolver:
    def __init__(self, media_root: str | Path | None = None) -> None:
        self.media_root = _normalize_root(media_root)

    def available(self) -> bool:
        return self.media_root is not None and self.media_root.is_dir()

    def resolve_video(self, video_id: str) -> Path | None:
        """Return a local video path for ``video_id`` if it exists."""
        if not self.available() or not video_id:
            return None
        candidates = [
            self.media_root / "videos" / f"{video_id}.mp4",
            self.media_root / f"{video_id}.mp4",
            self.media_root / video_id,
        ]
        if not _has_glob_chars(video_id):
            candidates.extend(
                sorted(self.media_root.glob(f"videos/*/*/{video_id}.*"))
            )
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in _VIDEO_EXTS:
                return candidate
        return None

    def resolve_keyframe(
        self, video_id: str, frame_id: int | None = None
    ) -> Path | None:
        """Resolve a keyframe image.  ``frame_id=None`` returns the first match."""
        base = self._keyframe_base(video_id)
        if base is None:
            return None
        if frame_id is not None:
            for candidate in _keyframe_candidates(base, video_id, frame_id):
                if candidate.is_file():
                    return candidate
        matches = sorted(
            item
            for item in base.iterdir()
            if item.is_file()
            and item.suffix.lower() in _IMAGE_EXTS
            and (frame_id is None or _keyframe_index(item, video_id) == frame_id)
        )
        if matches:
            return matches[0]
        return None

    def list_keyframes(self, video_id: str, *, limit: int = 50) -> list[tuple[int, Path]]:
        """List up to ``limit`` keyframes ordered by frame index."""
        base = self._keyframe_base(video_id)
        if base is None:
            return []
        matches: list[tuple[int, Path]] = []
        for item in base.iterdir():
            if item.is_file() and item.suffix.lower() in _IMAGE_EXTS:
                index = _keyframe_index(item, video_id)
                if index is not None:
                    matches.append((index, item))
        matches.sort(key=lambda pair: pair[0])
        return matches[:limit]

    def _keyframe_base(self, video_id: str) -> Path | None:
        if not self.available() or not video_id:
            return None
        base = self.media_root / "keyframes" / video_id
        if not base.is_dir():
            base = self.media_root / "keyframes"
        return base if base.is_dir() else None


def _has_glob_chars(value: str) -> bool:
    return any(ch in value for ch in "*?[") or "{" in value


def _keyframe_candidates(base: Path, video_id: str, frame_id: int) -> list[Path]:
    names = []
    for ext in _IMAGE_EXTS:
        names.append(f"{video_id}_f{frame_id:06d}{ext}")
        names.append(f"{video_id}_f{frame_id}{ext}")
        names.append(f"{frame_id:05d}{ext}")
        names.append(f"{frame_id}{ext}")
    return [base / name for name in names]


def _keyframe_index(path: Path, video_id: str) -> int | None:
    """Extract the frame index from a keyframe filename.

    Handles ``<video_id>_f000000.webp`` (digits after ``_f``) and plain
    numeric names like ``00000.jpg``.  Digits inside the video id are ignored.
    """
    name = path.name
    match = re.search(r"_f(\d+)(?:\.\w+)?$", name)
    if match:
        return int(match.group(1))
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    return None


def make_resolver(media_root: str | None) -> MediaResolver:
    return MediaResolver(media_root)
