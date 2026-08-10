"""Score curve charts for a selected frame/region.

Raw and normalized scores are kept on separate charts by default so the two
scales never overlap.  ``show_raw`` toggles the raw scale overlay.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_score_curves(
    samples: list[dict[str, Any]],
    *,
    proposals: list[dict[str, Any]] | None = None,
    show_raw: bool = False,
) -> None:
    if not samples:
        st.info("No frame-score samples for this region.")
        return

    ordered = sorted(samples, key=lambda item: item.get("timestamp_seconds", 0))
    timestamps = [item.get("timestamp_seconds", 0) for item in ordered]

    if show_raw:
        _curve_figure(
            timestamps,
            series={
                "raw_anchor": _values(ordered, "raw_anchor_score"),
                "raw_pre": _values(ordered, "raw_pre_score"),
                "raw_post": _values(ordered, "raw_post_score"),
                "raw_motion": _values(ordered, "raw_motion_score"),
            },
            proposals=proposals,
            title="Raw anchor/pre/post/motion scores",
        )

    _curve_figure(
        timestamps,
        series={
            "norm_anchor": _values(ordered, "normalized_anchor_score"),
            "norm_pre": _values(ordered, "normalized_pre_score"),
            "norm_post": _values(ordered, "normalized_post_score"),
            "norm_motion": _values(ordered, "normalized_motion_score"),
        },
        proposals=proposals,
        title="Normalized anchor/pre/post/motion scores",
    )


def _values(items: list[dict[str, Any]], key: str) -> list[float | None]:
    return [item.get(key) for item in items]


def _curve_figure(
    timestamps: list[float],
    *,
    series: dict[str, list[float | None]],
    proposals: list[dict[str, Any]] | None,
    title: str,
) -> None:
    figure = go.Figure()
    colors = {
        "raw_anchor": "#1f77b4",
        "raw_pre": "#ff7f0e",
        "raw_post": "#2ca02c",
        "raw_motion": "#9467bd",
        "norm_anchor": "#1f77b4",
        "norm_pre": "#ff7f0e",
        "norm_post": "#2ca02c",
        "norm_motion": "#9467bd",
    }
    for name, values in series.items():
        if any(value is not None for value in values):
            figure.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=colors.get(name, "#888")),
                )
            )

    if proposals:
        for proposal in proposals:
            figure.add_trace(
                go.Scatter(
                    x=[proposal.get("timestamp_seconds", 0)],
                    y=[proposal.get("final_event_score", 0)],
                    mode="markers",
                    marker=dict(symbol="diamond", size=11, color="#d62728"),
                    name=proposal.get("id", ""),
                    hovertemplate=(
                        f"<b>{proposal.get('id')}</b><br>"
                        f"final {proposal.get('final_event_score', 'n/a')} · "
                        f"semantic {proposal.get('normalized_semantic_score', 'n/a')} · "
                        f"boundary {proposal.get('normalized_boundary_score', 'n/a')}<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        height=280,
        title=title,
        xaxis_title="time (seconds)",
        yaxis_title="score",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
