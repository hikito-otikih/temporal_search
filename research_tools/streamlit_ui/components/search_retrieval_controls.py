"""'Advanced options': result count, timestamp refinement, and (adaptive
only) retrieval-tuning hyperparameter overrides."""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st


@dataclass(frozen=True)
class RetrievalControls:
    result_limit: int
    apply_refinement: bool
    retrieval_overrides: dict[str, int] = field(default_factory=dict)


def render_retrieval_controls(*, pipeline: str) -> RetrievalControls:
    with st.expander("Advanced options"):
        result_limit = st.slider("Number of results", min_value=5, max_value=50, value=20, step=5)
        apply_refinement = st.checkbox(
            "Refine timestamps (slower, experimental)",
            value=False,
            help=(
                "Runs a real GPU decode+embed scan around every matched moment of every "
                "result to snap onto the exact frame, instead of the coarse retrieval "
                "anchor. Cost scales with results x moments, no other cap - e.g. 50 "
                "results x 3 moments measured at ~65s. In practice the timestamp "
                "adjustment itself is usually small (a fraction of a second)."
            ),
        )
        if apply_refinement and result_limit > 20:
            st.caption(
                f"{result_limit} results with refinement on can take a while (roughly "
                f"1-1.5s per result per moment) - lower 'Number of results' for a faster search."
            )
        retrieval_overrides: dict[str, int] = {}
        if pipeline == "adaptive_coarse":
            st.divider()
            st.caption("Retrieval tuning (adaptive only) - hyperparameters.retrieval, applied before commands/retrieve.")
            rc1, rc2 = st.columns(2)
            with rc1:
                top_n_per_variant = st.number_input(
                    "top_n_per_variant", min_value=1, max_value=10_000, value=500, step=50,
                    help="Raw upstream hits kept per query variant, before fusion (server default 500).",
                )
                rrf_k = st.number_input(
                    "rrf_k", min_value=1, max_value=10_000, value=60, step=10,
                    help="Reciprocal Rank Fusion constant (server default 60).",
                )
            with rc2:
                top_n_fused = st.number_input(
                    "top_n_fused", min_value=1, max_value=10_000, value=1000, step=50,
                    help="Fused candidates kept per event after RRF, across every video combined (server default 1000).",
                )
                query_variants_per_event = st.number_input(
                    "query_variants_per_event", min_value=4, max_value=32, value=4, step=1,
                    help=(
                        "Max distinct query_variant tags accepted per event during "
                        "fusion (server default 4). /rewrite always produces exactly "
                        "4 (2 Vietnamese + 2 English, schema-enforced) - a lower "
                        "value here is not 'fewer variants', it's a guaranteed "
                        "fusion failure once those 4 arrive, so it's floored at 4."
                    ),
                )
            retrieval_overrides = {
                "top_n_per_variant": int(top_n_per_variant),
                "top_n_fused": int(top_n_fused),
                "rrf_k": int(rrf_k),
                "query_variants_per_event": int(query_variants_per_event),
            }

    return RetrievalControls(
        result_limit=result_limit,
        apply_refinement=apply_refinement,
        retrieval_overrides=retrieval_overrides,
    )
