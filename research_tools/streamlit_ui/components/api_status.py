"""Shared sidebar status panel: health, capability badges, session summary.

Also exports small rendering helpers used across pages so widget/formatting
conventions stay consistent.
"""

from __future__ import annotations

from typing import Any, Iterable

import streamlit as st

from models.ui_models import CapabilityResponse, SearcherCapability
from services.api_client import ApiError, ConnectionFailure, TemporalApiClient
from state import keys as K
from state import session_state as SS

STAGE_ORDER = ("rewrite", "retrieval", "region", "refinement", "proposal", "tuple")


def show_api_status(store: Any, client: TemporalApiClient) -> CapabilityResponse | None:
    """Sidebar block with backend/upstream health and capability badges."""
    with st.sidebar:
        st.subheader("Backend status")

        def _health_check(button_label: str, check):
            if st.button(button_label, use_container_width=True):
                try:
                    check()
                    st.success("healthy")
                except ApiError as exc:
                    st.error(f"{type(exc).__name__}: {exc.message}")

        _health_check("Check backend health", client.check_backend_health)
        _health_check("Check upstream health", client.check_upstream_health)

        capabilities = SS.capabilities(store)
        if capabilities is None:
            st.caption("Capability not loaded yet — use a page with a health check, "
                       "or run 'Discover capabilities' below.")
            if st.button("Discover capabilities", use_container_width=True):
                try:
                    capabilities = client.get_capabilities()
                    store[K.CAPABILITIES] = capabilities
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.message)
        else:
            show_capability_badges(capabilities)

        _show_session_summary(store)

        st.caption("Nothing secret is logged or exported. Diagnostic bundles "
                   "only include the URLs/capability flags stored in the UI.")


def show_capability_badges(capabilities: CapabilityResponse) -> None:
    adaptive = capabilities.adaptive()
    legacy = capabilities.legacy()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Legacy", "ready" if legacy and legacy.available else "down")
    with col2:
        st.metric("Adaptive core", "ready" if adaptive and adaptive.algorithm_core_available else "down")
    if adaptive is not None:
        st.metric(
            "Live frame refinement",
            "available" if adaptive.live_refinement_available else "unavailable",
            help=adaptive.frame_provider.reason,
        )
    if adaptive is not None and adaptive.runtime_model:
        st.caption(f"runtime model: {adaptive.runtime_model}")


def _show_session_summary(store: Any) -> None:
    st.divider()
    st.subheader("Session")
    session_id = SS.active_session_id(store)
    if not session_id:
        st.caption("No adaptive session selected.")
        return
    st.write(f"**ID** `{session_id[:8]}…`")
    st.write(f"**Revision** {SS.session_revision(store)}")
    dirty = store.get(K.SESSION_DIRTY, False)
    st.write(f"**Dirty** {'yes' if dirty else 'no'}")


def show_invalidated_badges(invalidated_stages: Iterable[str]) -> None:
    stages = list(invalidated_stages)
    if not stages:
        return
    st.warning("Stale artifacts: " + ", ".join(f"`{s}`" for s in stages))


def stage_status_badge(name: str, active: bool) -> str:
    if active:
        return f"✅ **{name}**"
    return f"🔘 {name}"


def show_connection_error(exc: ConnectionFailure, *, key: str = "connection_error") -> None:
    """`key` must be unique per call site on a page - two independent error
    states (e.g. a failed Preview and a failed Search) can both be visible
    on the same render, and Streamlit crashes with StreamlitDuplicateElementId
    if their "Retry" buttons share the same auto-generated key."""
    st.error(f"Cannot reach backend: {exc.message}")
    if st.button("Retry", use_container_width=True, key=key):
        st.rerun()
