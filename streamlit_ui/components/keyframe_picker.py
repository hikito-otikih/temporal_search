"""Keyframe filmstrip picker and exact-timestamp fix control.

Thumbnails resolve through the media resolver (lazy, paginated).  A frame can
also be fixed by typing an exact ``timestamp_seconds`` when no thumbnail is
available.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from models.ui_models import format_timestamp
from state import keys as K


def render_keyframe_picker(
    *,
    video_id: str,
    event_id: str,
    region_id: str | None,
    media_resolver: Any,
    on_fix: Callable[[int, float, str | None], Any],
    default_timestamp: float | None = None,
) -> None:
    st.subheader(f"Keyframes · event `{event_id}`")
    keyframes = (
        media_resolver.list_keyframes(video_id, limit=50)
        if media_resolver is not None
        else []
    )

    if keyframes:
        page_size = 6
        page = st.number_input(
            "Thumbnail page",
            min_value=1,
            max_value=max(1, (len(keyframes) + page_size - 1) // page_size),
            value=1,
            step=1,
            key=f"thumb_page_{event_id}",
        )
        chunk = keyframes[(int(page) - 1) * page_size : int(page) * page_size]
        cols = st.columns(len(chunk))
        frame_ids = []
        for column, (frame_id, path) in zip(cols, chunk):
            with column:
                st.image(str(path), use_container_width=True)
                st.caption(f"frame {frame_id}")
                if st.button("Fix", key=f"fix_thumb_{event_id}_{frame_id}", use_container_width=True):
                    frame_ids.append(frame_id)
                    on_fix(frame_id, float(frame_id) * _frame_seconds_guess(), region_id)
    else:
        st.caption("No keyframe thumbnails — use exact timestamp below.")

    st.divider()
    st.caption("Fix by exact timestamp (works when no thumbnail exists).")
    timestamp = st.number_input(
        "timestamp_seconds",
        value=float(default_timestamp or 0.0),
        min_value=0.0,
        step=0.1,
        format="%.3f",
        key=f"fix_ts_{event_id}",
    )
    if st.button("Fix for event", key=f"fix_event_{event_id}", type="primary"):
        on_fix(int(round(timestamp * 10)), float(timestamp), region_id)
    st.caption(f"Display: {format_timestamp(float(timestamp))}")


def _frame_seconds_guess() -> float:
    return 0.1
