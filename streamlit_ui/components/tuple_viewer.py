"""Ordered tuple cards with score/gap decomposition and export."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from models.ui_models import format_timestamp
from services.export_service import csv_bytes, json_bytes


def render_tuple_list(tuples: list[dict[str, Any]]) -> str | None:
    if not tuples:
        st.info("No ordered tuples yet. Complete the pipeline or adjust constraints.")
        return None

    options = []
    for item in tuples:
        label = (
            f"{item.get('id')} · {item.get('video_id')} · "
            f"final {item.get('normalized_final_score', item.get('raw_final_score', 'n/a'))}"
        )
        options.append((item.get("id"), label, item))
    selected = st.selectbox(
        "Ordered tuple",
        options=[label for _, label, _ in options],
        key="tuple_list_select",
    )
    chosen = next(item for _, label, item in options if label == selected)
    render_tuple_card(chosen)
    return chosen.get("id")


def render_tuple_card(tuple_item: dict[str, Any]) -> None:
    video_id = tuple_item.get("video_id", "?")
    proposals = tuple_item.get("proposals", [])
    gaps = tuple_item.get("adjacent_gaps_seconds", [])
    penalties = tuple_item.get("adjacent_gap_penalties", [])

    st.subheader(f"Tuple `{tuple_item.get('id')}` · video `{video_id}`")
    if len(proposals) != len(gaps) + 1:
        st.warning("Proposal/gap count mismatch.")

    cols = st.columns(len(proposals))
    for index, (column, proposal) in enumerate(zip(cols, proposals)):
        with column:
            st.metric(f"E{index + 1} · {proposal.get('event_id')}", format_timestamp(proposal.get("timestamp_seconds", 0)))
            st.caption(f"frame {proposal.get('frame_id')} · final {proposal.get('final_event_score')}")
            if index < len(gaps):
                st.caption(f"→ gap {gaps[index]:.2f}s")
                st.caption(f"penalty {penalties[index]:.4f}")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("raw event mean", tuple_item.get("raw_event_mean_score"))
    c2.metric("gap penalty", tuple_item.get("raw_gap_penalty"))
    c3.metric("constraint bonus", tuple_item.get("raw_constraint_bonus"))
    c4.metric("raw final", tuple_item.get("raw_final_score"))
    st.metric("normalized final", tuple_item.get("normalized_final_score"))

    st.subheader("Export tuple")
    c1, c2 = st.columns(2)
    export_payload = {"tuple": tuple_item, "note": "exported for demo/annotation"}
    c1.download_button(
        "Download tuple JSON",
        data=json_bytes(export_payload),
        file_name=f"{tuple_item.get('id')}.json",
        mime="application/json",
        key="export_tuple_json",
    )
    rows = []
    for proposal in proposals:
        rows.append(
            {
                "tuple_id": tuple_item.get("id"),
                "video_id": video_id,
                "event_id": proposal.get("event_id"),
                "proposal_id": proposal.get("id"),
                "frame_id": proposal.get("frame_id"),
                "timestamp_seconds": proposal.get("timestamp_seconds"),
                "final_event_score": proposal.get("final_event_score"),
                "source": proposal.get("source"),
            }
        )
    c2.download_button(
        "Download tuple CSV",
        data=csv_bytes(rows),
        file_name=f"{tuple_item.get('id')}.csv",
        mime="text/csv",
        key="export_tuple_csv",
    )


def render_tuple_summary_table(tuples: list[dict[str, Any]]) -> None:
    rows = []
    for item in tuples:
        rows.append(
            {
                "tuple_id": item.get("id"),
                "video_id": item.get("video_id"),
                "events": ",".join(p.get("event_id") for p in item.get("proposals", [])),
                "gaps_s": item.get("adjacent_gaps_seconds"),
                "gap_penalties": item.get("adjacent_gap_penalties"),
                "raw_final": item.get("raw_final_score"),
                "norm_final": item.get("normalized_final_score"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)
