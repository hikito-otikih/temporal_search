"""03 Region Inspector.

Interactive timeline, region detail, keyframe filmstrip, and score curves for
a selected region.  Works in metadata mode when media is missing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client, get_media_resolver
from components import api_status
from components.keyframe_picker import render_keyframe_picker
from components.region_timeline import render_region_table, render_region_timeline
from components.score_curves import render_score_curves
from models.ui_models import format_timestamp
from state import keys as K
from state import session_state as SS

bootstrap()
configure_sidebar()

client = get_api_client()
media_resolver = get_media_resolver()
SS.init_ui_state(st.session_state)

st.title("Region Inspector")

session_id = SS.active_session_id(st.session_state)
if not session_id:
    st.info("No active adaptive session. Create/open one in the Adaptive Session Lab first.")
    st.stop()

response = SS.session_response(st.session_state)
if response is None:
    response = SS.refresh_session(st.session_state, client)
    if response is None:
        st.error("Could not load the active session.")
        st.stop()

st.caption(f"session `{session_id[:8]}…` · revision {SS.session_revision(st.session_state)}")

event_ids = [event.get("event_id") for event in response.session.get("events", [])]
event_color_map = {event_id: _event_color(index) for index, event_id in enumerate(event_ids)}

regions = client.get_regions(session_id, limit=1000).get("items", [])
if not regions:
    st.info("No regions yet. Ingest candidates in the Adaptive Session Lab.")
    st.stop()

# Event / video filters ----------------------------------------------------
c1, c2 = st.columns(2)
event_filter = c1.selectbox("Event filter", options=["all", *event_ids], key="inspector_event_filter")
video_filter = c2.selectbox("Video filter", options=["all"] + sorted({r.get("video_id") for r in regions}), key="inspector_video_filter")
visible = [
    r for r in regions
    if (event_filter == "all" or r.get("event_id") == event_filter)
    and (video_filter == "all" or r.get("video_id") == video_filter)
]

if not visible:
    st.warning("No regions match the current filter.")
    st.stop()

selected_id = render_region_table(visible)
if selected_id:
    st.session_state[K.SELECTED_REGION_ID] = selected_id
else:
    selected_id = st.session_state.get(K.SELECTED_REGION_ID)

render_region_timeline(visible, event_color_map=event_color_map)

region = next((r for r in visible if r.get("id") == selected_id), visible[0])
if selected_id is None:
    selected_id = region.get("id")
    st.session_state[K.SELECTED_REGION_ID] = selected_id

st.subheader(f"Region `{selected_id}`")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("start", format_timestamp(region.get("start_seconds", 0)))
c2.metric("end", format_timestamp(region.get("end_seconds", 0)))
c3.metric("duration", f"{region.get('end_seconds', 0) - region.get('start_seconds', 0):.2f}s")
c4.metric("coarse score", region.get("raw_coarse_score"))
c5.metric("status", region.get("refinement_status"))

candidate_ids = region.get("candidate_ids", [])
if candidate_ids:
    st.caption(f"candidate ids: {len(candidate_ids)} · {', '.join(str(x) for x in candidate_ids[:10])}" + ("…" if len(candidate_ids) > 10 else ""))

st.divider()
video_id = region.get("video_id")
event_id = region.get("event_id")

tab_video, tab_curves, tab_constraint = st.tabs(["Video / keyframes", "Score curves", "Constraints"])

with tab_video:
    media = media_resolver.resolve_video(video_id) if media_resolver else None
    if media:
        st.video(str(media))
    else:
        st.caption("Video preview unavailable — metadata mode.")
    render_keyframe_picker(
        video_id=video_id,
        event_id=event_id,
        region_id=region.get("id"),
        media_resolver=media_resolver,
        on_fix=lambda frame_id, ts, region_id: _fix_frame(frame_id, ts, region_id),
        default_timestamp=float(region.get("start_seconds", 0)),
    )

with tab_curves:
    # GET .../frame-scores was retired along with session-level frame-score
    # ingestion (see docs/ADAPTIVE_PIPELINE_MIGRATION.md) - no code path can
    # ever populate samples for a region anymore, so render_score_curves'
    # own "No frame-score samples for this region" empty state below is the
    # permanent, correct outcome here.
    samples: list[dict[str, Any]] = []
    proposals = client.get_proposals(session_id, event_id=event_id, limit=200).get("items", [])
    region_proposals = [p for p in proposals if p.get("region_id") == region.get("id")]
    show_raw = st.checkbox("Show raw score overlay", value=False, key="show_raw_curves")
    render_score_curves(samples, proposals=region_proposals, show_raw=show_raw)

with tab_constraint:
    st.write("Event constraint (server state):")
    event_constraints = response.session.get("constraints", {}).get("event_constraints", {})
    st.json(event_constraints.get(event_id, {}))
    if st.button("Clear this event's constraint", use_container_width=True):
        result, error = SS.apply_mutation(
            st.session_state,
            client,
            lambda rev: client.clear_event_constraint(session_id, expected_revision=rev, event_id=event_id),
        )
        if error:
            st.error(error)
        else:
            st.info(f"Cleared: invalidated {result.invalidated_stages}")
            SS.refresh_session(st.session_state, client)
            st.rerun()

st.markdown("---")
api_status.show_api_status(st.session_state, client)


def _fix_frame(frame_id: int, timestamp_seconds: float, region_id: str | None) -> None:
    result, error = SS.apply_mutation(
        st.session_state,
        client,
        lambda rev: client.fix_frame(
            session_id, expected_revision=rev,
            event_id=event_id, video_id=video_id,
            frame_id=frame_id, timestamp_seconds=timestamp_seconds,
            region_id=region_id,
        ),
    )
    if error:
        st.error(error)
    else:
        st.info(f"Fixed frame {frame_id} @ {timestamp_seconds}s: invalidated {result.invalidated_stages}")
        SS.refresh_session(st.session_state, client)


def _event_color(index: int) -> str:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    return palette[index % len(palette)]
