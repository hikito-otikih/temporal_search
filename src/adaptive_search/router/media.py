"""Real native-fps frame preview, for manual frame correction."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ..dependencies import boundary_refinement_runtime
from ..exceptions import LiveRefinementUnavailableError
from ..providers import VideoAssetNotFoundError

from .errors import _raise_api_error

router = APIRouter()


@router.get("/media/{video_id}/frame-preview")
def get_frame_preview(
    video_id: str,
    anchor_seconds: float = Query(ge=0.0),
    radius_frames: int = Query(default=15, ge=1, le=60),
    stride: int = Query(default=1, ge=1, le=10),
):
    """Real native-fps frames around an anchor, for manual frame correction.

    Only needs the frame provider (no embedder/scoring) - lighter weight than
    boundary refinement and works even when just the decoder is configured.
    """
    try:
        provider = boundary_refinement_runtime.frame_provider
        capability = provider.capabilities()
        if not capability.available:
            raise LiveRefinementUnavailableError(
                capability.reason or "frame provider is unavailable"
            )
        metadata = provider.catalog.metadata(video_id)
        if metadata is None or metadata.fps is None:
            raise VideoAssetNotFoundError(
                f"no fps metadata available for video_id {video_id!r}"
            )
        fps = metadata.fps
        duration_ms = metadata.duration_ms
        anchor_frame = round(anchor_seconds * fps)

        def frame_to_ms(index: int) -> int:
            ms = max(0, round(index / fps * 1000))
            return ms if duration_ms is None else min(ms, duration_ms)

        offsets = range(-radius_frames, radius_frames + 1)
        pts_by_offset = {offset: frame_to_ms(anchor_frame + offset * stride) for offset in offsets}
        target_pts_ms = sorted(set(pts_by_offset.values()))
        frames = provider.get_frames(video_id, target_pts_ms)
        frames_by_pts = {frame.pts_ms: frame for frame in frames}

        seen_pts: set[int] = set()
        items: list[dict[str, Any]] = []
        for offset in offsets:
            pts_ms = pts_by_offset[offset]
            if pts_ms in seen_pts:
                continue
            seen_pts.add(pts_ms)
            frame = frames_by_pts.get(pts_ms)
            if frame is None or frame.image is None:
                continue
            items.append(
                {
                    "offset": offset,
                    "pts_ms": pts_ms,
                    "timestamp_seconds": pts_ms / 1000.0,
                    "image_base64": _encode_frame_jpeg(frame.image),
                }
            )
        return {
            "video_id": video_id,
            "anchor_seconds": anchor_seconds,
            "fps": fps,
            "frames": items,
        }
    except Exception as exc:
        _raise_api_error(exc)


def _encode_frame_jpeg(image: Any) -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
