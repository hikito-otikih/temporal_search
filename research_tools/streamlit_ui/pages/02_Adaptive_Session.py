"""02 Adaptive Session Lab.

Stepper workflow: 1 Events → 2 Candidates → 3 Regions → 4 Frame scores →
5 Proposals → 6 Tuples → 7 Feedback/recompute.

Every mutation carries ``expected_revision``; conflicts reload the session and
never auto-retry.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client, get_media_resolver, default_hyperparameters
from components import api_status
from components.event_editor import edit_drafts, patch_existing_event
from components.hyperparameter_editor import render_hyperparameter_editor
from components.json_inspector import render_json_inspector
from components.region_timeline import render_region_table, render_region_timeline
from components.video_filter import render_video_browser
from services.api_client import ApiError
from services.export_service import json_bytes
from state import keys as K
from state import session_state as SS

bootstrap()
configure_sidebar()

client = get_api_client()
media_resolver = get_media_resolver()
SS.init_ui_state(st.session_state)

st.title("Adaptive Session Lab")

STEP_LABELS = [
    "1 Events",
    "2 Candidates",
    "3 Regions",
    "4 Frame scores",
    "5 Proposals",
    "6 Tuples",
    "7 Feedback/recompute",
]

session_id = SS.active_session_id(st.session_state)

# ---------------------------------------------------------------------------
# Session management bar
# ---------------------------------------------------------------------------

bar = st.columns([1, 1, 1, 2])
if bar[0].button("New session", use_container_width=True):
    SS.clear_active_session(st.session_state)
    st.session_state[K.EVENT_DRAFTS] = [{
        "event_id": "event_1",
        "original_query": "",
        "anchor_query": "",
        "pre_state": "",
        "post_state": "",
        "boundary_type": "transition",
    }]
    st.rerun()

loaded = None
if bar[1].button("Open session", use_container_width=True):
    session_id = SS.active_session_id(st.session_state)
    if session_id:
        loaded = SS.refresh_session(st.session_state, client)
        if loaded is None:
            st.error("Could not reload the active session.")
    else:
        st.warning("No active session. Create one first.")

if bar[2].button("Delete session", use_container_width=True):
    if session_id:
        result, error = SS.apply_mutation(
            st.session_state, client, lambda rev: client.delete_session(session_id, rev)
        )
        if error:
            st.error(error)
        else:
            SS.clear_active_session(st.session_state)
            st.rerun()

session_id = SS.active_session_id(st.session_state)
if session_id:
    bar[3].write(f"**Active session** `{session_id[:8]}…` · revision **{SS.session_revision(st.session_state)}**")
else:
    bar[3].info("No active session — create one in Step 1.")

# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------

step = st.radio(
    "Workflow step",
    options=list(STEP_LABELS),
    index=min(st.session_state.get("adaptive_step_index", 0), len(STEP_LABELS) - 1),
    horizontal=True,
    key="adaptive_step_radio",
)
st.session_state["adaptive_step_index"] = STEP_LABELS.index(step)

response = SS.session_response(st.session_state)
counts = response.artifact_counts if response else None
if counts:
    st.caption(
        f"candidates {counts.candidates} · regions {counts.regions} · "
        f"proposals {counts.proposals} · runs {counts.runs}"
    )

invalidated = (response.session.get("last_invalidated_stages", []) if response else [])
if invalidated:
    api_status.show_invalidated_badges(invalidated)

# ---------------------------------------------------------------------------
# Step 1: Events
# ---------------------------------------------------------------------------

if step == STEP_LABELS[0]:
    st.subheader("Events")
    if session_id is None:
        drafts = edit_drafts(st.session_state)
        common_query = st.text_area("common_query (optional)", value=st.session_state.get(K.COMMON_QUERY, ""), key="common_query_input")
        with st.expander("Session constraints (optional)"):
            allowed_text = st.text_input("allowed_video_ids (comma separated)", key="create_allowed_videos")
            max_span = st.number_input("max_tuple_span_seconds (0 = none)", value=0.0, min_value=0.0, step=1.0, key="create_max_span")
        if st.button("Create adaptive session", type="primary"):
            from models.ui_models import event_draft_to_definition
            events = [event_draft_to_definition(d) for d in drafts]
            invalid = [d for d in drafts if not d["event_id"].strip() or not d["anchor_query"].strip()]
            if invalid:
                st.error("Each event needs a non-empty event_id and anchor_query.")
            else:
                constraints: dict[str, Any] = {}
                if allowed_text.strip():
                    constraints["allowed_video_ids"] = [v.strip() for v in allowed_text.split(",") if v.strip()]
                if max_span > 0:
                    constraints["max_tuple_span_seconds"] = float(max_span)
                try:
                    created = client.create_session(
                        events=events,
                        common_query=common_query or None,
                        hyperparameters=default_hyperparameters(),
                        constraints=constraints,
                    )
                    SS.set_active_session(st.session_state, created)
                    st.session_state[K.APPLIED_HYPERPARAMETERS] = created.session.get("hyperparameters")
                    st.success(f"Session created: `{created.session['id']}`")
                    st.rerun()
                except ApiError as exc:
                    st.error(f"{type(exc).__name__}: {exc.message}")
    else:
        session = response.session
        st.caption(f"session status: {session.get('status')}")
        for event in session.get("events", []):
            def apply_patch(event_id: str, patch: dict[str, Any]) -> None:
                result, error = SS.apply_mutation(
                    st.session_state,
                    client,
                    lambda rev, eid=event_id, p=patch: client.patch_event(session_id, eid, expected_revision=rev, patch=p),
                )
                if error:
                    st.error(error)
                else:
                    st.info(f"Patched {event_id}: invalidated {result.invalidated_stages}")
                    SS.refresh_session(st.session_state, client)
            patch_existing_event(st.session_state, client, session_id, event, apply=apply_patch)

# ---------------------------------------------------------------------------
# Step 2: Candidates
# ---------------------------------------------------------------------------

elif step == STEP_LABELS[1]:
    st.subheader("Sparse candidates")
    if session_id is None:
        st.warning("Create a session in Step 1 first.")
    else:
        event_ids = [event.get("event_id") for event in response.session.get("events", [])]
        st.multiselect("Ingest candidates for events", options=event_ids, default=event_ids, key="candidate_event_select")
        with st.expander("Candidate JSON template"):
            template = [
                {
                    "id": "c1",
                    "session_id": session_id,
                    "event_id": event_ids[0] if event_ids else "event_1",
                    "video_id": "L21_V001",
                    "frame_id": 120,
                    "timestamp_seconds": 4.0,
                    "raw_relevance_score": 0.9,
                    "query_variant": "anchor-en-0",
                }
            ]
            st.code(json.dumps(template, indent=2, ensure_ascii=False))
            st.download_button("Download template", data=json_bytes(template), file_name="candidates_template.json", mime="application/json")
        uploaded = st.file_uploader("Upload candidates JSON/CSV", type=["json", "csv"], key="candidates_upload")
        candidates_payload = st.text_area("Or paste candidates JSON", height=200, key="candidates_paste")
        if st.button("Ingest candidates", type="primary"):
            payload = None
            if uploaded is not None:
                payload = json.loads(uploaded.read().decode("utf-8"))
            elif candidates_payload.strip():
                payload = json.loads(candidates_payload)
            if payload is None:
                st.error("Provide candidates via upload or paste.")
            else:
                chosen_events = st.session_state.get("candidate_event_select") or event_ids
                def _ingest(rev):
                    return client.ingest_candidates(
                        session_id, expected_revision=rev,
                        event_ids=list(chosen_events), candidates=payload,
                    )
                result, error = SS.apply_mutation(st.session_state, client, _ingest)
                if error:
                    st.error(error)
                else:
                    st.success(f"Run {result.run_id} · status {result.run_status} · regions {result.artifact_counts.regions}")
                    SS.refresh_session(st.session_state, client)
                    st.rerun()

# ---------------------------------------------------------------------------
# Step 3: Regions
# ---------------------------------------------------------------------------

elif step == STEP_LABELS[2]:
    st.subheader("Temporal regions & video ranking")
    if session_id is None:
        st.warning("Create a session and ingest candidates first.")
    else:
        regions = client.get_regions(session_id, limit=1000).get("items", [])
        if not regions:
            st.info("No regions yet. Ingest candidates in Step 2.")
        else:
            event_ids = [event.get("event_id") for event in response.session.get("events", [])]
            event_color_map = {event_id: _event_color(index) for index, event_id in enumerate(event_ids)}
            selected_region = render_region_table(regions)
            render_region_timeline(regions, event_color_map=event_color_map)
            if selected_region:
                st.session_state[K.SELECTED_REGION_ID] = selected_region

        videos = _video_rows(regions)
        selected_video = render_video_browser(
            st.session_state,
            videos,
            on_allow=lambda video_ids: _apply_video_constraint(video_ids),
        )
        if selected_video:
            st.session_state[K.SELECTED_VIDEO_ID] = selected_video
        if selected_video:
            from components.video_filter import render_video_preview
            render_video_preview(st.session_state, selected_video, media_resolver=media_resolver)

# ---------------------------------------------------------------------------
# Step 4: Frame scores
# ---------------------------------------------------------------------------

elif step == STEP_LABELS[3]:
    st.subheader("Frame scores")
    st.info(
        "`POST .../artifacts/frame-scores` was retired from the backend along "
        "with the refinement frontier it fed (see "
        "docs/ADAPTIVE_PIPELINE_MIGRATION.md) — this product does not support "
        "an external client precomputing frame-level scores and submitting "
        "them back into a session. Live boundary refinement "
        "(`apply_boundary_refinement` on Step 3/6) is a separate, unaffected "
        "mechanism that computes its own scores per request."
    )

# ---------------------------------------------------------------------------
# Step 5: Proposals
# ---------------------------------------------------------------------------

elif step == STEP_LABELS[4]:
    st.subheader("Proposals")
    if session_id is None:
        st.warning("Create a session first.")
    else:
        proposals = client.get_proposals(session_id, limit=1000).get("items", [])
        if not proposals:
            st.info("No proposals yet - proposals are created by fixing a frame "
                     "(commands/fix-frame), not generated automatically.")
        else:
            from components.proposal_table import render_proposal_table
            render_proposal_table(
                proposals,
                on_fix=lambda proposal: _fix_from_proposal(proposal),
            )

# ---------------------------------------------------------------------------
# Step 6: Tuples
# ---------------------------------------------------------------------------

elif step == STEP_LABELS[5]:
    st.subheader("Ordered tuples")
    if session_id is None:
        st.warning("Create a session and complete earlier steps first.")
    else:
        st.info(
            "`GET .../tuples` was retired from the backend (see "
            "docs/ADAPTIVE_PIPELINE_MIGRATION.md) — tuples are still built internally "
            "and feed `commands/fix-frame`, but there's no HTTP surface left to list "
            "their contents here. See Step 3's video-priorities view for the "
            "winning tuple's per-event fields instead."
        )

# ---------------------------------------------------------------------------
# Step 7: Feedback/recompute
# ---------------------------------------------------------------------------

elif step == STEP_LABELS[6]:
    st.subheader("Feedback & recompute")
    if session_id is None:
        st.warning("Create a session first.")
    else:
        applied = st.session_state.get(K.APPLIED_HYPERPARAMETERS) or response.session.get("hyperparameters") or {}
        render_hyperparameter_editor(
            st.session_state,
            applied=applied,
            on_apply=lambda draft: _apply_hyperparameters(draft),
            on_reset_preset=lambda name: st.rerun(),
        )
        st.divider()
        st.subheader("Session commands")
        c1, c2 = st.columns(2)
        event_options = [event.get("event_id") for event in response.session.get("events", [])]
        target_event = c1.selectbox("Event", options=event_options, key="clear_constraint_event")
        if c2.button("Clear event constraint", use_container_width=True):
            result, error = SS.apply_mutation(
                st.session_state,
                client,
                lambda rev: client.clear_event_constraint(session_id, expected_revision=rev, event_id=target_event),
            )
            if error:
                st.error(error)
            else:
                st.info(f"Cleared constraint for {target_event}: invalidated {result.invalidated_stages}")
                SS.refresh_session(st.session_state, client)
                st.rerun()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

st.markdown("---")
api_status.show_api_status(st.session_state, client)


def _apply_hyperparameters(draft: dict[str, Any]) -> None:
    result, error = SS.apply_mutation(
        st.session_state,
        client,
        lambda rev: client.patch_hyperparameters(
            session_id, expected_revision=rev, patch=draft
        ),
    )
    if error:
        st.error(error)
    else:
        st.success(f"Applied: invalidated {result.invalidated_stages}")
        SS.refresh_session(st.session_state, client)
        st.rerun()


def _apply_video_constraint(video_ids: list[str]) -> None:
    result, error = SS.apply_mutation(
        st.session_state,
        client,
        lambda rev: client.mark_videos(session_id, expected_revision=rev, video_ids=video_ids),
    )
    if error:
        st.error(error)
    else:
        st.info(f"Allowed {len(video_ids)} videos: invalidated {result.invalidated_stages}")
        SS.refresh_session(st.session_state, client)


def _fix_from_proposal(proposal: dict[str, Any]) -> None:
    result, error = SS.apply_mutation(
        st.session_state,
        client,
        lambda rev: client.fix_frame(
            session_id,
            expected_revision=rev,
            event_id=proposal.get("event_id"),
            video_id=proposal.get("video_id"),
            frame_id=proposal.get("frame_id"),
            timestamp_seconds=proposal.get("timestamp_seconds"),
            region_id=proposal.get("region_id"),
        ),
    )
    if error:
        st.error(error)
    else:
        st.info(f"Fixed frame for {proposal.get('event_id')}: invalidated {result.invalidated_stages}")
        SS.refresh_session(st.session_state, client)




def _video_rows(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}
    for region in regions:
        video_id = region.get("video_id")
        row = by_video.setdefault(
            video_id, {"video_id": video_id, "event_coverage": 0, "priority_score": 0.0}
        )
        row["event_coverage"] += 1
        row["priority_score"] = max(row.get("priority_score", 0.0), float(region.get("raw_coarse_score", 0.0) or 0.0))
    return sorted(by_video.values(), key=lambda item: -item["priority_score"])


def _event_color(index: int) -> str:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    return palette[index % len(palette)]
