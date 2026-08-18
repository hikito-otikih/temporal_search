"""Plotly temporal-region timeline.

Interaction relies primarily on the data-table selection (Streamlit reruns are
more stable for selection than Plotly clicks); the chart is the visualization
companion with hover/zoom.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_region_timeline(
    regions: list[dict[str, Any]],
    *,
    event_color_map: dict[str, str],
    proposals: list[dict[str, Any]] | None = None,
    show_proposals: bool = True,
    fixed_frames: list[dict[str, Any]] | None = None,
) -> None:
    if not regions:
        st.info("No temporal regions to display.")
        return

    figure = go.Figure()
    for region in regions:
        event_id = region.get("event_id", "?")
        color = event_color_map.get(event_id, "#888")
        status = region.get("refinement_status", "pending")
        if status == "dense":
            color = _darken(color)
        y_center = _event_y(event_id, sorted({r.get("event_id") for r in regions}))
        figure.add_trace(
            go.Bar(
                x=[region.get("end_seconds", 0) - region.get("start_seconds", 0)],
                y=[y_center],
                base=[region.get("start_seconds", 0)],
                orientation="h",
                marker=dict(color=color, opacity=0.55),
                width=0.5,
                name=region.get("id", ""),
                customdata=[[region.get("id")]],
                hovertemplate=(
                    f"<b>{region.get('id')}</b><br>"
                    f"{region.get('video_id')} · event {region.get('event_id')}<br>"
                    f"start {region.get('start_seconds', 0):.2f}s · "
                    f"end {region.get('end_seconds', 0):.2f}s<br>"
                    f"coarse {region.get('raw_coarse_score', 'n/a')}<br>"
                    f"status {region.get('refinement_status')} · "
                    f"user {region.get('user_status', 'active')}<extra></extra>"
                ),
            )
        )

    if show_proposals and proposals:
        for proposal in proposals:
            figure.add_trace(
                go.Scatter(
                    x=[proposal.get("timestamp_seconds", 0)],
                    y=[_event_y(proposal.get("event_id", "?"), sorted({r.get("event_id") for r in regions}))],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=10,
                        color=_source_color(proposal.get("source", "dense")),
                    ),
                    name=proposal.get("id", ""),
                    customdata=[[proposal.get("id")]],
                    hovertemplate=(
                        f"<b>{proposal.get('id')}</b><br>"
                        f"{proposal.get('timestamp_seconds', 0):.3f}s · "
                        f"final {proposal.get('final_event_score', 'n/a')}<br>"
                        f"source {proposal.get('source')}<extra></extra>"
                    ),
                )
            )

    if fixed_frames:
        for fixed in fixed_frames:
            figure.add_trace(
                go.Scatter(
                    x=[fixed.get("timestamp_seconds", 0)],
                    y=[_event_y(fixed.get("event_id", "?"), sorted({r.get("event_id") for r in regions}))],
                    mode="markers",
                    marker=dict(symbol="star", size=16, color="#FFD700"),
                    name=fixed.get("event_id", "fixed"),
                    customdata=[[fixed.get("event_id")]],
                    hovertemplate="fixed frame<br>event %{customdata[0]}<extra></extra>",
                )
            )

    figure.update_layout(
        height=320,
        barmode="overlay",
        xaxis_title="time (seconds)",
        yaxis_title="event",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_region_table(regions: list[dict[str, Any]]) -> str | None:
    """Render regions as a selectable table. Returns the selected region id."""
    if not regions:
        return None
    rows = []
    for region in regions:
        rows.append(
            {
                "region_id": region.get("id"),
                "event_id": region.get("event_id"),
                "video_id": region.get("video_id"),
                "start_s": region.get("start_seconds"),
                "end_s": region.get("end_seconds"),
                "duration_s": round(region.get("end_seconds", 0) - region.get("start_seconds", 0), 3),
                "coarse_score": region.get("raw_coarse_score"),
                "status": region.get("refinement_status"),
                "user_status": region.get("user_status"),
            }
        )
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)
    options = {row["region_id"]: row["region_id"] for row in rows}
    if not options:
        return None
    return st.selectbox("Select region", options=list(options), key="region_table_select")


def _event_y(event_id: str, ordered: list[str]) -> int:
    return len(ordered) - ordered.index(event_id)


def _source_color(source: str) -> str:
    return {
        "sparse": "#1f77b4",
        "medium": "#ff7f0e",
        "dense": "#d62728",
        "user": "#2ca02c",
    }.get(source, "#888")


def _darken(color: str) -> str:
    try:
        value = color.lstrip("#")
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        return f"#{max(0, r - 40):02x}{max(0, g - 40):02x}{max(0, b - 40):02x}"
    except Exception:
        return color
