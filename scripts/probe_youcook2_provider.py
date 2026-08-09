"""Probe the local YouCook2 catalog and optional decoder without loading a VLM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from adaptive_search.providers import YouCook2FrameProvider


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("PTS values must be non-negative")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-root")
    parser.add_argument("--video-id")
    parser.add_argument(
        "--decode-pts-ms",
        type=_csv_ints,
        help="optional comma-separated PTS values; requires the av package",
    )
    args = parser.parse_args()

    provider = YouCook2FrameProvider(
        args.data_root,
        metadata_root=args.metadata_root,
    )
    capability = provider.capabilities()
    result = {
        "capabilities": asdict(capability),
        "asset_count": provider.catalog.asset_count,
        "video_root": str(provider.catalog.video_root)
        if provider.catalog.video_root is not None
        else None,
    }
    if args.video_id:
        result["video_id"] = args.video_id
        result["video_path"] = str(provider.catalog.resolve(args.video_id))
        metadata = provider.catalog.metadata(args.video_id)
        result["metadata"] = asdict(metadata) if metadata is not None else None
        if args.decode_pts_ms is not None:
            frames = provider.get_frames(args.video_id, args.decode_pts_ms)
            result["decoded_frames"] = [
                {
                    "pts_ms": frame.pts_ms,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "frame_index": frame.frame_index,
                    "image_shape": list(frame.image.shape)
                    if hasattr(frame.image, "shape")
                    else None,
                }
                for frame in frames
            ]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if capability.available or args.decode_pts_ms is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
