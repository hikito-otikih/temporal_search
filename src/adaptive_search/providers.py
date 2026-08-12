"""Frame-provider boundaries and a local raw-video YouCook2 implementation.

Video decoding stays behind an injectable protocol so importing the API neve
imports or downloads an optional multimedia runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from numbers import Integral
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable


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
        video_path: Path,
        pts_ms: Sequence[int],
    ) -> list[DecodedVideoFrame]: ...


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


class PyAVFrameDecoder:
    """Lazy decoder that preserves real stream presentation timestamps."""

    @property
    def name(self) -> str:
        return "pyav"

    def availability(self) -> tuple[bool, str | None]:
        try:
            self._import_av()
        except FrameProviderConfigurationError as exc:
            return False, str(exc)
        return True, None

    def decode(
        self,
        video_path: Path,
        pts_ms: Sequence[int],
    ) -> list[DecodedVideoFrame]:
        av = self._import_av()
        results: list[DecodedVideoFrame] = []
        try:
            with av.open(str(video_path), mode="r") as container:
                if not container.streams.video:
                    raise FrameDecodeError(
                        f"video has no video stream: {video_path.name}"
                    )
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                duration_ms = _pyav_duration_ms(container, stream)
                for target_ms in pts_ms:
                    if duration_ms is not None and target_ms > duration_ms:
                        # Region boundaries can come from upstream keyframe
                        # timestamps that slightly exceed the real media
                        # duration; decode the last frame instead of aborting.
                        target_ms = duration_ms
                    frame = self._decode_nearest(container, stream, target_ms)
                    if frame is None:
                        raise FrameDecodeError(
                            f"could not decode a frame near {target_ms}ms from "
                            f"{video_path.name}"
                        )
                    results.append(
                        DecodedVideoFrame(
                            pts_ms=_frame_pts_ms(
                                frame,
                                stream.time_base,
                                getattr(stream, "start_time", None),
                            ),
                            frame_index=None,
                            image=frame.to_ndarray(format="rgb24"),
                        )
                    )
        except (FrameDecodeError, FrameRequestError):
            raise
        except Exception as exc:
            raise FrameDecodeError(
                f"failed to decode {video_path.name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return results

    @staticmethod
    def _decode_nearest(container: Any, stream: Any, target_ms: int) -> Any | None:
        if stream.time_base is None:
            raise FrameDecodeError("video stream does not expose a time base")
        time_base = Fraction(stream.time_base)
        start_ticks = getattr(stream, "start_time", None) or 0
        target_ticks = start_ticks + int(Fraction(target_ms, 1000) / time_base)
        container.seek(
            max(0, target_ticks),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual_ms = _frame_pts_ms(frame, stream.time_base, start_ticks)
            if actual_ms < 0:
                continue
            if actual_ms >= target_ms:
                if previous is None:
                    return frame
                previous_ms = _frame_pts_ms(
                    previous,
                    stream.time_base,
                    start_ticks,
                )
                if target_ms - previous_ms <= actual_ms - target_ms:
                    return previous
                return frame
            previous = frame
        # The nominal duration can be just beyond the final frame PTS.
        return previous

    @staticmethod
    def _import_av() -> Any:
        try:
            import av
        except Exception as exc:  # pragma: no cover - optional dependency
            raise FrameProviderConfigurationError(
                "PyAV is unavailable; install the optional 'av' package to "
                "enable local YouCook2 frame decoding"
            ) from exc
        return av


class YouCook2FrameProvider:
    """Decode local YouCook2 videos for dense adaptive refinement."""

    def __init__(
        self,
        video_root: str | os.PathLike[str] | None,
        *,
        metadata_root: str | os.PathLike[str] | None = None,
        decoder: VideoFrameDecoder | None = None,
        video_extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS,
        max_batch_frames: int = DEFAULT_MAX_BATCH_FRAMES,
    ) -> None:
        if (
            not isinstance(max_batch_frames, Integral)
            or isinstance(max_batch_frames, bool)
            or max_batch_frames <= 0
        ):
            raise FrameProviderConfigurationError(
                "max_batch_frames must be a positive integer"
            )
        self.catalog = YouCook2AssetCatalog(
            video_root,
            metadata_root=metadata_root,
            video_extensions=video_extensions,
        )
        self.decoder = decoder or PyAVFrameDecoder()
        self.max_batch_frames = int(max_batch_frames)

    @classmethod
    def from_environment(
        cls,
        *,
        decoder: VideoFrameDecoder | None = None,
        data_root_env: str = YOUCOOK2_DATA_ROOT_ENV,
        metadata_root_env: str = YOUCOOK2_METADATA_ROOT_ENV,
        max_batch_frames: int = DEFAULT_MAX_BATCH_FRAMES,
    ) -> "YouCook2FrameProvider":
        """Build from environment without failing application startup."""

        return cls(
            os.getenv(data_root_env),
            metadata_root=os.getenv(metadata_root_env),
            decoder=decoder,
            max_batch_frames=max_batch_frames,
        )

    def capabilities(self) -> FrameProviderCapabilities:
        catalog_reason = self.catalog.reason
        if catalog_reason is not None:
            return FrameProviderCapabilities(
                available=False,
                supports_batch=False,
                supports_dense_sampling=False,
                supports_thumbnails=False,
                reason=catalog_reason,
            )
        decoder_available, decoder_reason = self.decoder.availability()
        if not decoder_available:
            return FrameProviderCapabilities(
                available=False,
                supports_batch=False,
                supports_dense_sampling=False,
                supports_thumbnails=False,
                reason=decoder_reason
                or f"decoder {self.decoder.name} is unavailable",
            )
        return FrameProviderCapabilities(
            available=True,
            supports_batch=True,
            supports_dense_sampling=True,
            supports_thumbnails=False,
            reason=None,
        )

    def refresh_catalog(self) -> FrameProviderCapabilities:
        self.catalog.refresh()
        return self.capabilities()

    def get_frames(
        self,
        video_id: str,
        pts_ms: Sequence[int],
    ) -> list[FrameReference]:
        self._require_available()
        normalized_id = _validate_video_id(video_id)
        video_path = self.catalog.resolve(normalized_id)
        targets = _normalize_pts(pts_ms)
        if len(targets) > self.max_batch_frames:
            raise FrameRequestError(
                f"requested {len(targets)} distinct frames, exceeding provider "
                f"batch limit {self.max_batch_frames}"
            )
        if not targets:
            return []

        metadata = self.catalog.metadata(normalized_id)
        if (
            metadata is not None
            and metadata.duration_ms is not None
            and targets[-1] > metadata.duration_ms
        ):
            raise FrameRequestError(
                f"requested PTS {targets[-1]}ms exceeds metadata duration "
                f"{metadata.duration_ms}ms for video_id {normalized_id!r}"
            )

        decoded = self.decoder.decode(video_path, targets)
        if len(decoded) != len(targets):
            raise FrameDecodeError(
                f"decoder {self.decoder.name} returned {len(decoded)} frames for "
                f"{len(targets)} distinct requested PTS values"
            )

        references: list[FrameReference] = []
        seen_actual_pts: set[int] = set()
        try:
            ordered_frames = sorted(decoded, key=lambda item: item.pts_ms)
        except (AttributeError, TypeError) as exc:
            raise FrameDecodeError(
                f"decoder {self.decoder.name} returned malformed frame objects"
            ) from exc
        for frame in ordered_frames:
            actual_pts = _validate_pts(frame.pts_ms, name="decoded pts_ms")
            if frame.image is None:
                raise FrameDecodeError(
                    f"decoder {self.decoder.name} returned an empty image at "
                    f"{actual_pts}ms"
                )
            frame_index: int | None
            if frame.frame_index is None:
                frame_index = None
            elif (
                not isinstance(frame.frame_index, Integral)
                or isinstance(frame.frame_index, bool)
                or frame.frame_index < 0
            ):
                raise FrameDecodeError(
                    "decoder returned an invalid negative/non-integer frame_index"
                )
            else:
                frame_index = int(frame.frame_index)
            if actual_pts in seen_actual_pts:
                continue
            seen_actual_pts.add(actual_pts)
            references.append(
                FrameReference(
                    video_id=normalized_id,
                    pts_ms=actual_pts,
                    frame_index=frame_index,
                    image=frame.image,
                )
            )
        return references

    def plan_interval(
        self,
        start_pts_ms: int,
        end_pts_ms: int,
        interval_ms: int,
        *,
        max_frames: int,
    ) -> list[int]:
        """Compute the target PTS grid for an interval without decoding.

        Lets a caller collect several regions' target timestamps (even across
        different regions of the same video) and pass the union to
        ``get_frames`` once, instead of paying a container-open per region.
        """

        start = _validate_pts(start_pts_ms, name="start_pts_ms")
        end = _validate_pts(end_pts_ms, name="end_pts_ms")
        interval = _validate_positive_int(interval_ms, name="interval_ms")
        requested_budget = _validate_positive_int(max_frames, name="max_frames")
        if end < start:
            raise FrameRequestError(
                "end_pts_ms must be greater than or equal to start_pts_ms"
            )
        budget = min(requested_budget, self.max_batch_frames)
        return _budgeted_interval_grid(start, end, interval, budget)

    def sample_interval(
        self,
        video_id: str,
        start_pts_ms: int,
        end_pts_ms: int,
        interval_ms: int,
        *,
        max_frames: int,
    ) -> list[FrameReference]:
        targets = self.plan_interval(
            start_pts_ms, end_pts_ms, interval_ms, max_frames=max_frames
        )
        return self.get_frames(video_id, targets)

    def _require_available(self) -> None:
        capability = self.capabilities()
        if not capability.available:
            raise FrameProviderConfigurationError(
                capability.reason or "YouCook2 frame provider is unavailable"
            )


class UnavailableFrameProvider:
    """Explicit deployment state when only sparse keyframe metadata exists."""

    def __init__(self, reason: str = "no raw video or frame API is configured") -> None:
        self._reason = reason

    def capabilities(self) -> FrameProviderCapabilities:
        return FrameProviderCapabilities(
            available=False,
            supports_batch=False,
            supports_dense_sampling=False,
            supports_thumbnails=False,
            reason=self._reason,
        )

    def get_frames(
        self,
        video_id: str,
        pts_ms: Sequence[int],
    ) -> list[FrameReference]:
        raise RuntimeError(self._reason)

    def sample_interval(
        self,
        video_id: str,
        start_pts_ms: int,
        end_pts_ms: int,
        interval_ms: int,
        *,
        max_frames: int,
    ) -> list[FrameReference]:
        raise RuntimeError(self._reason)

    def plan_interval(
        self,
        start_pts_ms: int,
        end_pts_ms: int,
        interval_ms: int,
        *,
        max_frames: int,
    ) -> list[int]:
        raise RuntimeError(self._reason)


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


def _budgeted_interval_grid(
    start_pts_ms: int,
    end_pts_ms: int,
    interval_ms: int,
    max_frames: int,
) -> list[int]:
    """Create an inclusive grid, evenly downsampled across the full region."""

    if start_pts_ms == end_pts_ms:
        return [start_pts_ms]
    distance = end_pts_ms - start_pts_ms
    regular_steps, remainder = divmod(distance, interval_ms)
    total_points = regular_steps + 1 + int(remainder != 0)

    def point_at(index: int) -> int:
        if remainder and index == total_points - 1:
            return end_pts_ms
        return start_pts_ms + index * interval_ms

    if total_points <= max_frames:
        return [point_at(index) for index in range(total_points)]
    if max_frames == 1:
        return [start_pts_ms + distance // 2]
    indexes = [
        (sample_index * (total_points - 1)) // (max_frames - 1)
        for sample_index in range(max_frames)
    ]
    return [point_at(index) for index in indexes]


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


def _frame_pts_ms(
    frame: Any,
    time_base: Any,
    start_time: int | None = None,
) -> int:
    if frame.pts is None:
        raise FrameDecodeError(
            "decoded frame does not expose a presentation timestamp"
        )
    origin = 0 if start_time is None else int(start_time)
    milliseconds = Fraction(frame.pts - origin) * Fraction(time_base) * 1000
    return (milliseconds.numerator * 2 + milliseconds.denominator) // (
        2 * milliseconds.denominator
    )


def _pyav_duration_ms(container: Any, stream: Any) -> int | None:
    if stream.duration is not None and stream.time_base is not None:
        duration = Fraction(stream.duration) * Fraction(stream.time_base) * 1000
        return (
            duration.numerator + duration.denominator - 1
        ) // duration.denominator
    if container.duration is not None:
        # PyAV exposes container duration in AV_TIME_BASE microseconds.
        return (int(container.duration) + 999) // 1000
    return None
