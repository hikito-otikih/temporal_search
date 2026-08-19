"""Rewrite preview: shows what the LLM rewrite step will produce for the
current query, *authoritatively* - `get_authoritative_analysis` is what lets
Search create its session directly from a still-fresh preview (via
`create_session_from_rewrite`) instead of rewriting a second, independent
time. A preview stops being usable the moment the query text it was
generated from changes; there is no partial/fuzzy freshness, only exact
match or not.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.api_status import show_connection_error
from services.api_client import ApiError, ConnectionFailure, TemporalApiClient
from state import keys as K


def get_authoritative_analysis(store: Any, query_text: str) -> dict[str, Any] | None:
    """The cached preview, but only if it still matches `query_text` exactly
    - a stale preview (query edited since) has nothing valid to reuse."""
    preview = store.get(K.SEARCH_REWRITE_PREVIEW)
    if not preview:
        return None
    if store.get(K.SEARCH_REWRITE_PREVIEW_INPUT) != query_text:
        return None
    return preview


def render_preview_controls(
    store: Any,
    client: TemporalApiClient,
    *,
    common_query: str | None,
    lines: list[str],
    query_text: str,
) -> None:
    preview_clicked = st.button("Preview rewrite")
    if preview_clicked:
        if not lines:
            store[K.SEARCH_REWRITE_PREVIEW] = None
            store[K.SEARCH_REWRITE_PREVIEW_ERROR] = "Describe at least one moment."
        else:
            store[K.SEARCH_REWRITE_PREVIEW_ERROR] = None
            try:
                with st.spinner("Running rewrite…"):
                    store[K.SEARCH_REWRITE_PREVIEW] = client.preview_rewrite(
                        queries=lines, common_query=common_query
                    )
                    store[K.SEARCH_REWRITE_PREVIEW_INPUT] = query_text
            except ConnectionFailure as exc:
                store[K.SEARCH_REWRITE_PREVIEW] = None
                store[K.SEARCH_REWRITE_PREVIEW_ERROR] = ("connection", exc)
            except ApiError as exc:
                store[K.SEARCH_REWRITE_PREVIEW] = None
                store[K.SEARCH_REWRITE_PREVIEW_ERROR] = ("api", exc)

    preview_error = store.get(K.SEARCH_REWRITE_PREVIEW_ERROR)
    if preview_error:
        if isinstance(preview_error, tuple):
            kind, payload = preview_error
            if kind == "connection":
                show_connection_error(payload, key="preview_retry")
            else:
                st.error(f"{type(payload).__name__}: {payload.message}")
        else:
            st.error(preview_error)

    preview = store.get(K.SEARCH_REWRITE_PREVIEW)
    if not preview:
        return

    # Fingerprinted to the exact query text it was generated from - without
    # this, editing the query after previewing left the old preview
    # displayed as if it still applied, with nothing indicating it now
    # describes different input entirely (not just "the LLM may phrase it
    # slightly differently").
    if store.get(K.SEARCH_REWRITE_PREVIEW_INPUT) != query_text:
        st.warning(
            "This preview was generated for different input - the query text "
            "has changed since. Search will run its own fresh rewrite instead "
            "of reusing it; press 'Preview rewrite' again to make this the "
            "exact analysis Search uses."
        )
    else:
        st.caption(
            "This is exactly what Search will use - pressing Search below "
            "creates the session from this analysis directly, with no "
            "second LLM call."
        )
    for event in preview.get("events", []):
        with st.expander(f"{event.get('event_id')}: {event.get('original_query')}"):
            st.markdown(f"**Target moment:** {event.get('target_moment_vi')}")
            st.markdown(f"**Anchor query:** {event.get('anchor_query')}")
            if event.get("pre_state") or event.get("post_state"):
                st.caption(f"Pre-state: {event.get('pre_state') or '—'} · Post-state: {event.get('post_state') or '—'}")
            relation = event.get("temporal_relation") or {}
            st.caption(f"Boundary: {event.get('boundary')} · Relation: {relation.get('relation')}"
                       + (f" (ref: {relation.get('reference_event_id')})" if relation.get("reference_event_id") is not None else ""))
            st.markdown("**Retrieval queries (VI):** " + "; ".join(event.get("retrieval_queries_vi", [])))
            st.markdown("**Retrieval queries (EN):** " + "; ".join(event.get("retrieval_queries_en", [])))
