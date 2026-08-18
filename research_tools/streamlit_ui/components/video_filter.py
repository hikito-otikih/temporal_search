"""Video browser and filter panel.

View filters (search, sort, pin, compare) only change what the page displays.
Backend constraints (allowlist/reject) are sent only when the user explicitly
chooses **Apply as search constraint**.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from state import keys as K


def render_video_browser(
    store: Any,
    videos: list[dict[str, Any]],
    *,
    on_allow: Callable[[list[str]], Any],
    on_reject: Callable[[str], Any] | None = None,
) -> str | None:
    """Render the ranked video table + filters. Returns selected video_id."""
    if not videos:
        st.info("No videos available. Ingest sparse candidates first.")
        return None

    df = pd.DataFrame(videos)
    display_cols = [
        col for col in ("video_id", "event_coverage", "priority_score", "mean_best_event_score")
        if col in df.columns
    ]
    if not display_cols:
        st.error("Video table missing expected columns.")
        return None

    filtered = _apply_view_filters(df)

    search = st.text_input("Filter by video ID", key="video_search")
    if search:
        mask = filtered["video_id"].astype(str).str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    sort_col = st.selectbox(
        "Sort by", options=display_cols, index=0, key="video_sort_col"
    )
    filtered = filtered.sort_values(sort_col, ascending=False)

    st.dataframe(filtered[display_cols], use_container_width=True, height=320)

    col1, col2, col3 = st.columns(3)
    selected = st.multiselect(
        "Select videos", options=list(filtered["video_id"]), key="video_multiselect",
        label_visibility="collapsed",
    )
    if col1.button("View only", use_container_width=True):
        store[K.VIEW_FILTERS] = {"video_ids": list(selected)} if selected else {}
    if col2.button("Apply as search constraint", use_container_width=True):
        if selected:
            on_allow(list(selected))
        else:
            st.warning("Select at least one video.")
    if col3.button("Pin selected", use_container_width=True):
        store[K.PINNED_VIDEOS] = list(selected)

    pinned = store.get(K.PINNED_VIDEOS, [])
    if pinned:
        st.caption(f"Pinned: {', '.join(str(x) for x in pinned)}")

    options = list(filtered["video_id"])
    if not options:
        st.warning("No videos match the current view filter.")
        return None
    selected_id = st.selectbox(
        "Inspect video", options=options, key="video_inspect_select"
    )
    return str(selected_id)


def _apply_view_filters(df: pd.DataFrame) -> pd.DataFrame:
    return df


def render_video_preview(
    store: Any,
    video_id: str,
    *,
    media_resolver: Any,
    on_reject: Callable[[str], Any] | None = None,
) -> None:
    st.subheader(f"Video `{video_id}`")
    media = None
    if media_resolver is not None:
        media = media_resolver.resolve_video(video_id)
    if media:
        st.video(str(media))
    else:
        st.caption("Preview unavailable — metadata mode (media root missing or file not found).")

    rejected = store.get(K.VIEW_FILTERS, {}).get("rejected_video_ids", [])
    if on_reject is not None:
        if st.button("Reject video", use_container_width=True):
            on_reject(video_id)
        if video_id in rejected:
            st.warning("This video is marked rejected in the view filter.")
