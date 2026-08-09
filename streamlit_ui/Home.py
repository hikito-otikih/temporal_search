"""Temporal Search Console — research/debug UI for the adaptive pipeline.

This is a research console, not a production frontend.  The FastAPI backend is
the source of truth; this app only renders, inspects, and mutates it.
"""

from __future__ import annotations

import streamlit as st

from _bootstrap import APP_VERSION, bootstrap, configure_sidebar, get_api_client
from components import api_status
from state import keys as K

bootstrap()
configure_sidebar()

client = get_api_client()
capabilities = None
try:
    capabilities = client.get_capabilities()
    st.session_state[K.CAPABILITIES] = capabilities
except Exception as exc:  # noqa: BLE001 - surface friendly message
    st.error(f"Backend unavailable: {exc}")

st.title("Temporal Search Console")
st.caption(f"app version {APP_VERSION} · research/debug console for the FastAPI temporal-search backend")

st.markdown(
    """
| Lab | Purpose |
|---|---|
| **01 Legacy Search** | One-shot `/temporal-search` with ordered queries, object filter, raw scores. |
| **02 Adaptive Session** | Full session workflow: events → candidates → regions → frame scores → proposals → tuples. |
| **03 Region Inspector** | Temporal timeline, keyframe filmstrip, score curves. |
| **04 Tuple Explorer** | Ordered tuples, gap penalties, score decomposition, export. |
| **05 YouCook2 Evaluation** | Corpus Video Recall@K with leakage guard and dedup. |
| **06 Run Comparison** | Compare runs/hyperparameter snapshots. |
"""
)

if capabilities is not None:
    st.subheader("Capability")
    adaptive = capabilities.adaptive()
    if adaptive is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm core", "available" if adaptive.algorithm_core_available else "unavailable")
        c2.metric("Live frame refinement", "available" if adaptive.live_refinement_available else "unavailable")
        c3.metric("Runtime model", adaptive.runtime_model or "n/a")
        if adaptive.frame_provider.reason:
            st.caption(f"Frame provider: {adaptive.frame_provider.reason}")
        st.json(
            {
                "supported_relations": adaptive.supported_relations,
                "supported_boundary_profiles": adaptive.supported_boundary_profiles,
                "quality_embedding_model": adaptive.quality_embedding_model,
                "quality_reranker_model": adaptive.quality_reranker_model,
            }
        )
else:
    st.info("Run a backend health check from the sidebar to load capability flags.")

st.markdown("---")
st.caption("Reproducibility: every mutation carries `expected_revision`; revision "
           "conflicts reload the session and ask you to re-apply. Secrets are never "
           "logged or exported.")
