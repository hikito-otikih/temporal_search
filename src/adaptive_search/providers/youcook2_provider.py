"""The concrete `FrameProvider` for local YouCook2 videos, composing the
asset catalog (`catalog.py`) with an injectable decoder (default:
`pyav_decoder.PyAVFrameDecoder`)."""

from __future__ import annotations

import os
from numbers import Integral
from typing import Sequence

from .base import (
    DEFAULT_MAX_BATCH_FRAMES,
    DEFAULT_VIDEO_EXTENSIONS,
    YOUCOOK2_DATA_ROOT_ENV,
    YOUCOOK2_METADATA_ROOT_ENV,
    FrameDecodeError,
    FrameProviderCapabilities,
    FrameProviderConfigurationError,
    FrameReference,
    FrameRequestError,
    VideoFrameDecoder,
    _normalize_pts,
    _validate_positive_int,
    _validate_pts,
    _validate_video_id,
)
from .catalog import YouCook2AssetCatalog
from .pyav_decoder import PyAVFrameDecoder


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
