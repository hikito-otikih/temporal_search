"""Event editor: draft editing before session creation and patching after.

Draft mode is fully local: reorder/duplicate/delete operate on
``EVENT_DRAFTS`` in session state and never hit the backend.  Once a session
exists, each event is edited through ``PATCH .../events/{event_id}`` and the
returned invalidation is surfaced.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from models.ui_models import BOUNDARY_TYPES, event_completeness_issue
from state import keys as K


def _event_card(index: int, draft: dict[str, Any], *, expanded: bool) -> None:
    event_id = draft["event_id"]
    with st.expander(f"Event {index + 1} · `{event_id}`", expanded=expanded):
        st.text_input(
            "event_id (read-only after creation)",
            value=event_id,
            disabled=True,
            key=f"event_id_display_{index}",
        )
        draft["original_query"] = st.text_area(
            "original_query",
            value=draft.get("original_query", ""),
            height=80,
            key=f"original_query_{index}",
        )
        draft["anchor_query"] = st.text_area(
            "anchor_query",
            value=draft.get("anchor_query", ""),
            height=80,
            key=f"anchor_query_{index}",
        )
        pre, post = st.columns(2)
        with pre:
            draft["pre_state"] = st.text_area(
                "pre_state",
                value=draft.get("pre_state", "") or "",
                height=70,
                key=f"pre_state_{index}",
            )
        with post:
            draft["post_state"] = st.text_area(
                "post_state",
                value=draft.get("post_state", "") or "",
                height=70,
                key=f"post_state_{index}",
            )
        boundary_index = BOUNDARY_TYPES.index(draft.get("boundary_type", "transition"))
        draft["boundary_type"] = st.selectbox(
            "boundary_type",
            options=list(BOUNDARY_TYPES),
            index=max(0, boundary_index),
            key=f"boundary_type_{index}",
        )
        issue = event_completeness_issue(draft)
        if issue:
            st.warning(issue)


def edit_drafts(store: Any) -> list[dict[str, Any]]:
    """Render draft cards with add/remove/reorder controls. Returns drafts."""
    drafts = store.get(K.EVENT_DRAFTS)
    if not drafts:
        drafts = []
        store[K.EVENT_DRAFTS] = drafts

    if not drafts:
        st.info("No events yet. Add an event below.")
    else:
        for index, draft in enumerate(drafts):
            _event_card(index, draft, expanded=False)
            col1, col2, col3, col4 = st.columns(4)
            if col1.button("⬆️ Move up", key=f"move_up_{index}", use_container_width=True):
                reorder_drafts(store, index, -1)
                st.rerun()
            if col2.button("⬇️ Move down", key=f"move_down_{index}", use_container_width=True):
                reorder_drafts(store, index, +1)
                st.rerun()
            if col3.button("⧉ Duplicate", key=f"duplicate_{index}", use_container_width=True):
                duplicate_draft(store, index)
                st.rerun()
            if col4.button("🗑️ Delete", key=f"delete_{index}", use_container_width=True):
                remove_draft(store, index)
                st.rerun()

    if st.button("➕ Add event", use_container_width=True):
        drafts.append(
            {
                "event_id": f"event_{len(drafts) + 1}",
                "original_query": "",
                "anchor_query": "",
                "pre_state": "",
                "post_state": "",
                "boundary_type": "transition",
            }
        )
        st.rerun()
    return drafts


def reorder_drafts(store: Any, index: int, delta: int) -> None:
    drafts = store.get(K.EVENT_DRAFTS)
    if not drafts:
        return
    target = index + delta
    if not (0 <= target < len(drafts)):
        return
    drafts[index], drafts[target] = drafts[target], drafts[index]
    store[K.EVENT_DRAFTS] = drafts


def duplicate_draft(store: Any, index: int) -> None:
    drafts = store.get(K.EVENT_DRAFTS)
    if not drafts:
        return
    copy = dict(drafts[index])
    copy["event_id"] = f"{copy['event_id']}_copy"
    drafts.insert(index + 1, copy)
    store[K.EVENT_DRAFTS] = drafts


def remove_draft(store: Any, index: int) -> None:
    drafts = store.get(K.EVENT_DRAFTS)
    if not drafts:
        return
    del drafts[index]
    store[K.EVENT_DRAFTS] = drafts


def patch_existing_event(
    store: Any,
    client: Any,
    session_id: str,
    event: dict[str, Any],
    *,
    apply: Callable[[str, dict[str, Any]], Any],
) -> None:
    """Render a read-only summary + patch controls for a created event."""
    event_id = event["event_id"]
    with st.expander(f"Event · `{event_id}`", expanded=False):
        st.caption(f"boundary_type={event.get('boundary_type')}")
        st.write(f"**original_query:** {event.get('original_query')}")
        st.write(f"**anchor_query:** {event.get('anchor_query')}")
        pre = event.get("pre_state")
        post = event.get("post_state")
        st.write(f"**pre_state:** {pre or '—'}  |  **post_state:** {post or '—'}")
        st.divider()
        new_anchor = st.text_area(
            "New anchor_query (patch)",
            value="",
            height=60,
            key=f"patch_anchor_{event_id}",
        )
        boundary = st.selectbox(
            "New boundary_type",
            options=list(BOUNDARY_TYPES),
            index=BOUNDARY_TYPES.index(event.get("boundary_type", "unknown")),
            key=f"patch_boundary_{event_id}",
        )
        if st.button("Apply patch", key=f"apply_patch_{event_id}"):
            patch: dict[str, Any] = {"boundary_type": boundary}
            if new_anchor.strip():
                patch["anchor_query"] = new_anchor.strip()
            apply(event_id, patch)
