"""04 Tuple Explorer.

Ordered tuple cards, adjacent gaps, gap penalties, and score decomposition.
Raw vs normalized scores are shown separately and never merged onto one axis.
"""

from __future__ import annotations

import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client
from components import api_status
from components.tuple_viewer import render_tuple_list, render_tuple_summary_table
from state import keys as K
from state import session_state as SS

bootstrap()
configure_sidebar()

client = get_api_client()
SS.init_ui_state(st.session_state)

st.title("Tuple Explorer")

session_id = SS.active_session_id(st.session_state)
if not session_id:
    st.info("No active adaptive session. Open one in the Adaptive Session Lab first.")
    st.stop()

response = SS.session_response(st.session_state)
if response is None:
    response = SS.refresh_session(st.session_state, client)
    if response is None:
        st.error("Could not load the active session.")
        st.stop()

st.caption(f"session `{session_id[:8]}…` · revision {SS.session_revision(st.session_state)}")

page = client.get_tuples(session_id, limit=200)
tuples = page.get("items", [])
st.caption(f"{page.get('total', len(tuples))} tuples (showing {len(tuples)})")

if not tuples:
    st.info("No tuples. Complete the pipeline in the Adaptive Session Lab.")
else:
    render_tuple_summary_table(tuples)
    st.divider()
    render_tuple_list(tuples)

st.markdown("---")
api_status.show_api_status(st.session_state, client)
