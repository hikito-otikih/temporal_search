"""04 Tuple Explorer.

`GET /v1/search-sessions/{id}/tuples` was retired from the backend (see
docs/ADAPTIVE_PIPELINE_MIGRATION.md section 1) - tuples are still assembled
internally and feed `commands/fix-frame`, but there is no HTTP surface left
to list their contents, so this page can only report the tuple count. See
Region Inspector for score curves and Adaptive Session Lab for the rest of
the pipeline.
"""

from __future__ import annotations

import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client
from components import api_status
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

st.info(
    "`GET .../tuples` was retired from the backend - this page can no longer list "
    "tuple contents. Use Region Inspector for score curves, or Adaptive Session Lab "
    "for the rest of the pipeline."
)
st.metric("Tuple count", response.artifact_counts.tuples)

st.markdown("---")
api_status.show_api_status(st.session_state, client)
