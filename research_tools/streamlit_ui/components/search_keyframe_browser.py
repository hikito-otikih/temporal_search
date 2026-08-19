"""Two ways to manually pick the right frame for one event/video, both
ending in `search_constraints.confirm_fixed_frame`:

- `render_frame_fixer`: unscored raw consecutive native frames around an
  anchor (GET .../frame-preview), for fine-grained manual picking when
  nothing scored nearby is quite right.
- `render_event_keyframe_browser`: this video's real, scored retrieval
  candidates for one event (GET /regions), ranked by relevance - what
  retrieval actually found, paginated and cached rather than a single
  unlabeled top-N cutoff.
"""

from __future__ import annotations

import base64
from typing import Any

import streamlit as st

from components.search_constraints import confirm_fixed_frame
from models.ui_models import format_timestamp
from services.api_client import ApiError, TemporalApiClient
from services.media_resolver import MediaResolver

REGIONS_PAGE_SIZE = 30


def render_frame_fixer(
    store: Any,
    client: TemporalApiClient,
    *,
    session_id: str,
    event_id: str,
    video_id: str,
    anchor_seconds: float,
    flag_key: str,
) -> None:
    # The 31-frame window is centered on `center_key`, which starts at the
    # detected/default anchor but can be moved anywhere in the video with the
    # step buttons or the jump-to box below - not limited to browsing only
    # around whatever the pipeline originally found.
    center_key = f"center_{flag_key}"
    if center_key not in st.session_state:
        st.session_state[center_key] = float(anchor_seconds)

    nav = st.columns([1, 1, 1, 1, 2])
    if nav[0].button("- 10s", key=f"back10_{flag_key}"):
        st.session_state[center_key] = max(0.0, st.session_state[center_key] - 10.0)
    if nav[1].button("- 1s", key=f"back1_{flag_key}"):
        st.session_state[center_key] = max(0.0, st.session_state[center_key] - 1.0)
    if nav[2].button("+ 1s", key=f"fwd1_{flag_key}"):
        st.session_state[center_key] += 1.0
    if nav[3].button("+ 10s", key=f"fwd10_{flag_key}"):
        st.session_state[center_key] += 10.0
    with nav[4]:
        st.number_input(
            "Jump to (seconds)",
            min_value=0.0,
            step=1.0,
            key=center_key,
            label_visibility="collapsed",
        )
    center_seconds = float(st.session_state[center_key])

    try:
        preview = client.get_frame_preview(video_id, anchor_seconds=center_seconds, radius_frames=15)
    except ApiError as exc:
        st.error(f"Could not load frames: {exc.message}")
        return
    frames = preview.get("frames", [])
    if not frames:
        st.caption("No frames available for this video.")
        return
    st.caption(f"{len(frames)} frames around {format_timestamp(center_seconds)} - click one to pick it.")
    columns_per_row = 6
    for row_start in range(0, len(frames), columns_per_row):
        row = frames[row_start : row_start + columns_per_row]
        cols = st.columns(len(row))
        for col, frame in zip(cols, row):
            with col:
                image_bytes = base64.b64decode(frame["image_base64"])
                st.image(image_bytes, use_container_width=True)
                choose_label = format_timestamp(frame["timestamp_seconds"])
                if frame["offset"] == 0:
                    choose_label += " *"
                if st.button(choose_label, key=f"pick_{flag_key}_{frame['pts_ms']}"):
                    with st.spinner("Saving…"):
                        confirm_fixed_frame(
                            store,
                            client,
                            session_id=session_id,
                            event_id=event_id,
                            video_id=video_id,
                            frame=frame,
                        )
                    st.session_state[flag_key] = False
                    st.session_state.pop(center_key, None)
                    st.rerun()


def render_event_keyframe_browser(
    store: Any,
    client: TemporalApiClient,
    resolver: MediaResolver,
    *,
    session_id: str,
    video_id: str,
    all_events: list[dict[str, Any]],
    index: int,
) -> None:
    """Dropdown next to the video player: pick an event, see this video's
    real candidate keyframes for it, ranked by relevance score
    (GET /regions). Distinct from `render_frame_fixer`, which browses
    unscored raw consecutive frames for fine-grained manual picking - this
    shows what retrieval actually found and scored, and still lets you pick
    one of them as the fixed frame."""

    event_options = {
        f"{e['event_id']}: {e['label']}": e["event_id"] for e in all_events if e.get("event_id")
    }
    if not event_options:
        return
    # Every widget/session_state key below is scoped by session_id, not just
    # index/video_id - result position and video_id can both repeat across
    # unrelated searches (a new session created by clicking "Search" again),
    # and without session_id a stale "keyframe_loaded" flag (or a stale
    # selectbox choice) from a previous session would silently carry over:
    # the grid would render as already-loaded for a row nobody has opened
    # in the *current* session, or GET /regions would be issued against the
    # right session_id but the state driving whether/what to fetch would
    # still reflect a completely different search.
    select_key = f"keyframe_event_select_{session_id}_{index}_{video_id}"
    chosen_label = st.selectbox(
        "Browse keyframes for event", options=list(event_options.keys()), key=select_key
    )
    chosen_event_id = event_options[chosen_label]

    # Streamlit re-runs the code inside a `with st.expander(...):` block on
    # every script run regardless of whether that expander is visually
    # collapsed - without this gate, every result row issues a GET /regions
    # call (and builds its full thumbnail/button grid) on every render, even
    # for rows nobody has expanded. At the default 20 results that was 20
    # requests before anyone opened anything; 50 results x several moments
    # could mean thousands of unused thumbnails/buttons. Loading is opt-in
    # per row (persists in session_state once clicked), not per expander-open.
    load_key = f"keyframe_loaded_{session_id}_{index}_{video_id}"
    if not st.session_state.get(load_key):
        if st.button("Load keyframes", key=f"keyframe_load_btn_{session_id}_{index}_{video_id}"):
            st.session_state[load_key] = True
            st.rerun()
        return

    # Fetched regions are cached in session_state, keyed by the exact
    # (session, row, video, event) they belong to - without this, every
    # script rerun (any widget interaction anywhere on the page, not just
    # ones in this expander) re-issued the same GET /regions call, even
    # though the underlying candidates never change once fetched. "Load
    # more" extends the cache with the next page instead of refetching
    # what's already loaded.
    regions_key = f"keyframe_regions_{session_id}_{index}_{video_id}_{chosen_event_id}"
    cached = st.session_state.get(regions_key)
    if cached is None:
        try:
            page = client.get_regions(
                session_id, event_id=chosen_event_id, video_id=video_id,
                offset=0, limit=REGIONS_PAGE_SIZE,
            )
        except ApiError as exc:
            st.error(f"Could not load keyframes: {exc.message}")
            return
        cached = {"items": page.get("items", []), "total": page.get("total", 0)}
        st.session_state[regions_key] = cached

    if not cached["items"]:
        st.caption("No candidate keyframes found for this event in this video.")
        return
    regions = sorted(
        cached["items"],
        key=lambda r: (
            r.get("normalized_coarse_score")
            if r.get("normalized_coarse_score") is not None
            else r.get("raw_coarse_score", 0.0)
        ),
        reverse=True,
    )
    loaded = len(cached["items"])
    total = cached["total"]
    # Honest about how much of the real candidate pool is actually shown -
    # a silent, unlabeled 50-item cap looked like "these are all the
    # candidates" even when many more existed.
    st.caption(f"Showing {loaded} of {total} candidate keyframe{'s' if total != 1 else ''}, ranked by relevance.")
    columns_per_row = 5
    for row_start in range(0, len(regions), columns_per_row):
        row = regions[row_start : row_start + columns_per_row]
        cols = st.columns(len(row))
        for col, region in zip(cols, row):
            with col:
                seconds = float(region.get("start_seconds", 0.0))
                thumbnail = resolver.resolve_keyframe_near(video_id, seconds) if resolver.available() else None
                if thumbnail is not None:
                    st.image(str(thumbnail), use_container_width=True)
                score = region.get("normalized_coarse_score")
                score_label = f"{score:.2f}" if score is not None else "—"
                st.caption(f"{format_timestamp(seconds)} · score {score_label}")
                pick_key = f"pick_region_{session_id}_{index}_{video_id}_{chosen_event_id}_{region['id']}"
                if st.button("Use this frame", key=pick_key, use_container_width=True):
                    with st.spinner("Saving…"):
                        confirm_fixed_frame(
                            store,
                            client,
                            session_id=session_id,
                            event_id=chosen_event_id,
                            video_id=video_id,
                            frame={"pts_ms": int(seconds * 1000), "timestamp_seconds": seconds},
                            # Preserves which real, scored candidate region
                            # this pick actually came from - without it,
                            # fix_frame() falls back to a synthetic
                            # placeholder id with no link back to /regions.
                            region_id=region["id"],
                        )
                    st.rerun()
    if loaded < total:
        remaining = total - loaded
        more_key = f"keyframe_load_more_{regions_key}"
        if st.button(f"Load {min(REGIONS_PAGE_SIZE, remaining)} more", key=more_key):
            try:
                more_page = client.get_regions(
                    session_id, event_id=chosen_event_id, video_id=video_id,
                    offset=loaded, limit=REGIONS_PAGE_SIZE,
                )
            except ApiError as exc:
                st.error(f"Could not load more keyframes: {exc.message}")
            else:
                cached["items"] = cached["items"] + more_page.get("items", [])
                cached["total"] = more_page.get("total", cached["total"])
                st.session_state[regions_key] = cached
                st.rerun()
