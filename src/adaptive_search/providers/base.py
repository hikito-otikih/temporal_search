"""Shared errors, small value types, protocols, and validation helpers for
the frame-provider boundary. No video decoding or filesystem walking lives
here - see `catalog.py` (YouCook2 asset/metadata resolution),
`pyav_decoder.py` (PyAV-backed decoding), and `youcook2_provider.py` (the
concrete `FrameProvider` implementation that composes the two)."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, Sequence, runtime_checkable


YOUCOOK2_DATA_ROOT_ENV = "YOUCOOK2_DATA_ROOT"
YOUCOOK2_METADATA_ROOT_ENV = "YOUCOOK2_METADATA_ROOT"
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm")
DEFAULT_MAX_BATCH_FRAMES = 4096
_MAX_PTS_MS = (1 << 63) - 1


class FrameProviderError(RuntimeError):
    """Base class for local frame-provider failures."""


class FrameProviderConfigurationError(FrameProviderError):
    """Raised when a provider root/catalog is unavailable or malformed."""


class VideoAssetNotFoundError(FrameProviderError):
    """Raised when a video ID is not present in the configured catalog."""


class FrameRequestError(FrameProviderError, ValueError):
    """Raised when PTS or budget values violate the provider contract."""


class FrameDecodeError(FrameProviderError):
    """Raised when a decoder cannot return every requested frame."""


@dataclass(frozen=True)
class FrameReference:
    """Canonical reference to one decoded frame.

    ``pts_ms`` is the decoded stream PTS represented at integer-millisecond
    precision. ``frame_index`` is retained for UI/legacy compatibility and
    must not be used as seconds unless a fixed frame rate is explicitly known.
    """

    video_id: str
    pts_ms: int
    frame_index: int | None = None
    image: Any | None = None
    thumbnail_url: str | None = None

    @property
    def timestamp_seconds(self) -> float:
        return self.pts_ms / 1000.0


@dataclass(frozen=True)
class FrameProviderCapabilities:
    available: bool
    supports_batch: bool
    supports_dense_sampling: bool
    supports_thumbnails: bool
    reason: str | None = None


@runtime_checkable
class FrameProvider(Protocol):
    """Fetch decoded frames at canonical presentation timestamps."""

    def capabilities(self) -> FrameProviderCapabilities: ...

    def get_frames(
        self,
        video_id: str,
        pts_ms: Sequence[int],
    ) -> list[FrameReference]: ...

    def sample_interval(
        self,
        video_id: str,
        start_pts_ms: int,
        end_pts_ms: int,
        interval_ms: int,
        *,
        max_frames: int,
    ) -> list[FrameReference]: ...

    def plan_interval(
        self,
        start_pts_ms: int,
        end_pts_ms: int,
        interval_ms: int,
        *,
        max_frames: int,
    ) -> list[int]: ...


@dataclass(frozen=True)
class DecodedVideoFrame:
    """One decoder result with the stream's actual presentation timestamp."""

    pts_ms: int
    image: Any
    frame_index: int | None = None


@runtime_checkable
class VideoFrameDecoder(Protocol):
    """Injectable local decoder used by YouCook2FrameProvider."""

    @property
    def name(self) -> str: ...

    def availability(self) -> tuple[bool, str | None]: ...

    def decode(
        self,
        video_path: Any,
        pts_ms: Sequence[int],
    ) -> list[DecodedVideoFrame]: ...


def _normalize_pts(pts_ms: Sequence[int]) -> list[int]:
    if isinstance(pts_ms, (str, bytes)):
        raise FrameRequestError(
            "pts_ms must be a sequence of integer milliseconds"
        )
    try:
        return sorted(
            {_validate_pts(value, name="pts_ms item") for value in pts_ms}
        )
    except TypeError as exc:
        raise FrameRequestError(
            "pts_ms must be an iterable of integer milliseconds"
        ) from exc


def _validate_pts(value: int, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise FrameRequestError(
            f"{name} must be an integer number of milliseconds"
        )
    normalized = int(value)
    if normalized < 0 or normalized > _MAX_PTS_MS:
        raise FrameRequestError(f"{name} must be in [0, {_MAX_PTS_MS}]")
    return normalized


def _validate_positive_int(value: int, *, name: str) -> int:
    normalized = _validate_pts(value, name=name)
    if normalized == 0:
        raise FrameRequestError(f"{name} must be positive")
    return normalized


def _validate_video_id(video_id: str) -> str:
    if not isinstance(video_id, str) or not video_id:
        raise VideoAssetNotFoundError("video_id must be a non-empty string")
    if video_id != video_id.strip() or "\x00" in video_id:
        raise VideoAssetNotFoundError(
            "video_id contains invalid whitespace or NUL"
        )
    if "/" in video_id or "\\" in video_id or video_id in {".", ".."}:
        raise VideoAssetNotFoundError(
            "video_id must be an opaque file stem"
        )
    return video_id
