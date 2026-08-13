"""Generate moment-oriented ("first moment X" / "last moment X") query variants.

Reuses the existing YouCook2 annotations as-is - no new ground truth is
collected. Each event's already-annotated `(start, end)` interval already *is*
onset/offset ground truth; this only rewrites the event text to ask for a
specific boundary and copies the interval verbatim into two new query files
per source video.

`load_query_directory_grouped()` rejects duplicate `video_id` values within a
single directory, and a video's "first" and "last" variants share the same
`video_id` - so they must land in separate subdirectories, each loaded
independently by the experiment orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .core import VideoQueryGroup, load_query_directory_grouped

MomentType = Literal["first", "last"]

_WRAP_TEMPLATES: dict[MomentType, str] = {
    "first": "khoảnh khắc đầu tiên {text}",
    "last": "khoảnh khắc cuối cùng {text}",
}
_INTRO_TEMPLATES: dict[MomentType, str] = {
    "first": "Xác định khoảnh khắc ĐẦU TIÊN của mỗi sự kiện sau:",
    "last": "Xác định khoảnh khắc CUỐI CÙNG của mỗi sự kiện sau:",
}


def _format_timestamp(total_seconds: float) -> str:
    """Render seconds as `M:SS` or `H:MM:SS`, round-tripping through `parse_timestamp`."""

    total = round(total_seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _render_variant(group: VideoQueryGroup, moment_type: MomentType) -> str:
    wrap = _WRAP_TEMPLATES[moment_type]
    lines: list[str] = []
    if group.context:
        lines.append(group.context)
    lines.append(_INTRO_TEMPLATES[moment_type])
    for event_id, text in group.events:
        lines.append(f"{event_id}: {wrap.format(text=text)}")
    lines.append("**Answer")
    # Synthetic path: VideoQueryGroup does not retain the original video_path
    # string (its `source` field is the query file's own path). canonical_video_id()
    # only strips directories/known suffixes and never validates existence, so
    # this round-trips correctly.
    lines.append(f'video_path: "{group.video_id}.mp4"')
    for event_id, _ in group.events:
        start, end = group.answers[event_id]
        lines.append(f"{event_id}: {_format_timestamp(start)} - {_format_timestamp(end)}")
    return "\n".join(lines) + "\n"


def generate_augmented_queries(source_dir: Path | str, output_root: Path | str) -> dict[str, int]:
    """Write `output_root/first/*.txt` and `output_root/last/*.txt` variants.

    Returns a small summary dict (`video_count`) for caller-side logging.
    """

    groups = load_query_directory_grouped(source_dir)
    output_root = Path(output_root)
    first_dir = output_root / "first"
    last_dir = output_root / "last"
    first_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        (first_dir / f"{group.video_id}__first.txt").write_text(
            _render_variant(group, "first"), encoding="utf-8"
        )
        (last_dir / f"{group.video_id}__last.txt").write_text(
            _render_variant(group, "last"), encoding="utf-8"
        )

    return {"video_count": len(groups)}
