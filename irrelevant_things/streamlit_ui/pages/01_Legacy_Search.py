"""01 Legacy Search Lab.

One-shot ``POST /temporal-search``.  No session, no region, no proposal, no
adaptive normalized score.  Reorderable event queries, legacy hyperparameters,
raw tuple scores and request/response JSON for regression/debugging.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client, get_media_resolver
from components import api_status
from components.json_inspector import render_json_inspector
from services.api_client import ApiError
from state import keys as K
from state import session_state as SS


def _chunks(sequence, size):
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


bootstrap()
configure_sidebar()

client = get_api_client()
media_resolver = get_media_resolver()
SS.init_ui_state(st.session_state)

st.title("Legacy Search Lab")
st.caption("One-shot temporal search. Legacy scores are raw and never labeled as probability/confidence.")

queries = st.session_state.get("legacy_queries", [])
if not queries:
    queries = [""]
    st.session_state["legacy_queries"] = queries

for index, _ in enumerate(queries):
    queries[index] = st.text_area(
        f"Event query {index + 1}",
        value=queries[index],
        key=f"legacy_query_{index}",
        height=70,
    )

c1, c2, _, _ = st.columns(4)
if c1.button("➕ Add", use_container_width=True):
    queries.append("")
    st.rerun()
if queries and len(queries) > 1 and c2.button("🗑️ Remove last", use_container_width=True):
    queries.pop()
    st.rerun()

c1, c2, c3, c4 = st.columns(4)
move_index = c1.number_input("Move query #", min_value=1, max_value=len(queries) or 1, value=1, step=1, key="legacy_move_index")
if c2.button("⬆️ Up", use_container_width=True):
    idx = int(move_index) - 1
    if idx > 0:
        queries[idx - 1], queries[idx] = queries[idx], queries[idx - 1]
        st.rerun()
if c3.button("⬇️ Down", use_container_width=True):
    idx = int(move_index) - 1
    if idx < len(queries) - 1:
        queries[idx], queries[idx + 1] = queries[idx + 1], queries[idx]
        st.rerun()

searcher_type = st.selectbox(
    "Searcher", options=["TemporalSearcher", "AmbiguousSearcher"], index=0
)

c1, c2, c3 = st.columns(3)
top_k_tuple = int(c1.number_input("top_k_tuple", value=100, min_value=1, max_value=10_000))
top_k_each_query = int(c2.number_input("top_k_each_query", value=100, min_value=1, max_value=10_000))
gamma = c3.number_input("gamma", value=0.05, min_value=0.0, step=0.01, format="%.3f")

with st.expander("Object filter (legacy)"):
    object_filter = st.checkbox("objectFilterMode")
    object_names = st.text_input("object_name_list (comma separated)", value="")
    object_threshold = st.number_input("objectThreshold", value=0.5, min_value=0.0, max_value=1.0, step=0.05)
object_name_list = [name.strip() for name in object_names.split(",") if name.strip()] if object_names.strip() else None

st.markdown("---")
run_pressed = st.button("Run legacy search", type="primary")

request_payload = {
    "query": [q for q in queries if q.strip()],
    "top_k_tuple": int(top_k_tuple),
    "top_k_each_query": int(top_k_each_query),
    "gamma": float(gamma),
    "searcher_type": searcher_type,
    "objectFilterMode": bool(object_filter),
    "object_name_list": object_name_list,
    "objectThreshold": float(object_threshold),
}

if run_pressed:
    non_empty = [q for q in queries if q.strip()]
    if not non_empty:
        st.error("Enter at least one event query.")
    else:
        request_payload["query"] = non_empty
        with st.spinner("Running legacy search…"):
            try:
                response = client.legacy_search(
                    queries=non_empty,
                    top_k_tuple=int(top_k_tuple),
                    top_k_each_query=int(top_k_each_query),
                    gamma=float(gamma),
                    searcher_type=searcher_type,
                    object_filter=bool(object_filter),
                    object_name_list=object_name_list,
                    object_threshold=float(object_threshold),
                )
                st.session_state["legacy_response"] = response
            except ApiError as exc:
                st.error(f"{type(exc).__name__}: {exc.message}")

if st.session_state.get("legacy_response") is not None:
    response = st.session_state["legacy_response"]
    results = response.get("results", [])
    st.subheader(f"Results · {len(results)} tuples")

    rows = []
    for item in results:
        for position, frame in enumerate(item.get("tuple", [])):
            rows.append(
                {
                    "video_name": item.get("video_name"),
                    "score": item.get("score"),
                    "query_id": frame.get("query_id"),
                    "frame_index": frame.get("frame_index"),
                    "timestamp": frame.get("timestamp"),
                    "frame_score": frame.get("score"),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)

    previews = [
        {
            "video_id": item.get("video_name"),
            "frame_index": max(
                item.get("tuple", []), key=lambda f: f.get("score", 0.0)
            ).get("frame_index"),
            "path": media_resolver.resolve_keyframe(
                item.get("video_name"),
                max(
                    item.get("tuple", []), key=lambda f: f.get("score", 0.0)
                ).get("frame_index"),
            )
            if media_resolver is not None
            else None,
            "score": item.get("score"),
        }
        for item in results
        if item.get("tuple")
    ]
    if previews:
        st.subheader("Keyframe previews")
        st.caption("Best-scored frame per tuple. Blank means media root is missing or the frame is not extracted.")
        for chunk in _chunks(previews, 6):
            cols = st.columns(len(chunk))
            for col, preview in zip(cols, chunk):
                with col:
                    if preview["path"]:
                        st.image(str(preview["path"]), use_container_width=True)
                    else:
                        st.caption("no media")
                    video = preview["video_id"] or "?"
                    st.caption(f"{video} · frame {preview['frame_index']}")

    with st.expander("Raw response JSON"):
        st.json(response)
    st.download_button(
        "Download response JSON",
        data=__import__("json").dumps(response, indent=2, ensure_ascii=False),
        file_name="legacy_search_response.json",
        mime="application/json",
    )

with st.expander("Request JSON"):
    render_json_inspector(request_payload, title="Legacy request")

st.markdown("---")
api_status.show_api_status(st.session_state, client)
