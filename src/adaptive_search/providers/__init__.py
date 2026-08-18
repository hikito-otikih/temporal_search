"""Frame-provider boundaries and a local raw-video YouCook2 implementation.

Video decoding stays behind an injectable protocol so importing the API
never imports or downloads an optional multimedia runtime.

Split into base.py (errors, value types, protocols, shared validation),
catalog.py (YouCook2AssetCatalog - video ID resolution + keyframe metadata),
pyav_decoder.py (PyAVFrameDecoder), and youcook2_provider.py
(YouCook2FrameProvider, UnavailableFrameProvider). This module re-exports
everything so `from adaptive_search.providers import X` (or `from
.providers import X` within the package) keeps working unchanged - existing
callers and tests reference the flat module path, some down to private
helpers like `_frame_pts_ms`.
"""

from __future__ import annotations

from .base import (
    DEFAULT_MAX_BATCH_FRAMES,
    DEFAULT_VIDEO_EXTENSIONS,
    YOUCOOK2_DATA_ROOT_ENV,
    YOUCOOK2_METADATA_ROOT_ENV,
    DecodedVideoFrame,
    FrameDecodeError,
    FrameProvider,
    FrameProviderCapabilities,
    FrameProviderConfigurationError,
    FrameProviderError,
    FrameReference,
    FrameRequestError,
    VideoAssetNotFoundError,
    VideoFrameDecoder,
    _MAX_PTS_MS,
    _normalize_pts,
    _validate_positive_int,
    _validate_pts,
    _validate_video_id,
)
from .catalog import (
    YouCook2AssetCatalog,
    YouCook2VideoMetadata,
    _insert_unique_path,
    _optional_path,
    _read_youcook2_metadata,
    _require_within_root,
    _walk_files,
)
from .pyav_decoder import PyAVFrameDecoder, _frame_pts_ms, _pyav_duration_ms
from .youcook2_provider import (
    UnavailableFrameProvider,
    YouCook2FrameProvider,
    _budgeted_interval_grid,
)

__all__ = [
    "DEFAULT_MAX_BATCH_FRAMES",
    "DEFAULT_VIDEO_EXTENSIONS",
    "YOUCOOK2_DATA_ROOT_ENV",
    "YOUCOOK2_METADATA_ROOT_ENV",
    "DecodedVideoFrame",
    "FrameDecodeError",
    "FrameProvider",
    "FrameProviderCapabilities",
    "FrameProviderConfigurationError",
    "FrameProviderError",
    "FrameReference",
    "FrameRequestError",
    "PyAVFrameDecoder",
    "UnavailableFrameProvider",
    "VideoAssetNotFoundError",
    "VideoFrameDecoder",
    "YouCook2AssetCatalog",
    "YouCook2FrameProvider",
    "YouCook2VideoMetadata",
]
