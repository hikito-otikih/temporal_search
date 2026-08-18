"""Proposal table with fix/reject and score decomposition."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st


def render_proposal_table(
    proposals: list[dict[str, Any]],
    *,
    on_fix: Callable[[dict[str, Any]], Any] | None = None,
    on_reject: Callable[[dict[str, Any]], Any] | None = None,
) -> str | None:
    if not proposals:
        st.info("No proposals. Ingest frame scores for refined regions first.")
        return None

    rows = []
    for proposal in proposals:
        rows.append(
            {
                "proposal_id": proposal.get("id"),
                "event_id": proposal.get("event_id"),
                "video_id": proposal.get("video_id"),
                "region_id": proposal.get("region_id"),
                "timestamp_s": proposal.get("timestamp_seconds"),
                "frame_id": proposal.get("frame_id"),
                "final": proposal.get("final_event_score"),
                "semantic": proposal.get("normalized_semantic_score"),
                "boundary": proposal.get("normalized_boundary_score"),
                "pre": proposal.get("pre_consistency_score"),
                "post": proposal.get("post_persistence_score"),
                "source": proposal.get("source"),
                "status": proposal.get("user_status"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)

    options = {row["proposal_id"]: row for row in rows}
    if not options:
        return None
    selected = st.selectbox("Select proposal", options=list(options), key="proposal_table_select")
    proposal = options[selected]

    with st.expander(f"Score decomposition · `{selected}`"):
        decomposition = {
            "raw_semantic_score": proposal.get("raw_semantic_score"),
            "normalized_semantic_score": proposal.get("normalized_semantic_score"),
            "raw_boundary_score": proposal.get("raw_boundary_score"),
            "normalized_boundary_score": proposal.get("normalized_boundary_score"),
            "raw_motion_score": proposal.get("raw_motion_score"),
            "normalized_motion_score": proposal.get("normalized_motion_score"),
            "pre_consistency_score": proposal.get("pre_consistency_score"),
            "post_persistence_score": proposal.get("post_persistence_score"),
            "final_event_score": proposal.get("final_event_score"),
            "left_window_seconds": proposal.get("left_window_seconds"),
            "right_window_seconds": proposal.get("right_window_seconds"),
            "source": proposal.get("source"),
        }
        st.json(decomposition)

    if on_fix or on_reject:
        col1, col2 = st.columns(2)
        if on_fix is not None and col1.button("Fix for event", use_container_width=True, key=f"fix_prop_{selected}"):
            on_fix(proposal)
        if on_reject is not None and col2.button("Reject proposal", use_container_width=True, key=f"reject_prop_{selected}"):
            on_reject(proposal)

    return selected
