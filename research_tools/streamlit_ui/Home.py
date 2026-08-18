"""Entrypoint: declares navigation, then hands off to the selected page.

Two sections: "Search" is the real product (type a query, get ranked video
results with timestamps and playback). "Developer Tools" is the original
research/debug console for inspecting pipeline internals (regions, proposals,
tuples, hyperparameters, run comparisons) — still here, just no longer the
first thing anyone sees.

No ``st.*`` call happens here before ``nav.run()`` so that whichever page
runs first can safely call ``st.set_page_config(...)`` as its own first
Streamlit command (every page still calls ``bootstrap()`` for itself,
unchanged from before this file existed).
"""

from __future__ import annotations

import streamlit as st

search_page = st.Page("pages/00_Search.py", title="Search", icon=":material/search:", default=True)
legacy_page = st.Page("pages/01_Legacy_Search.py", title="Legacy Search Lab", icon=":material/science:")
adaptive_page = st.Page("pages/02_Adaptive_Session.py", title="Adaptive Session Lab", icon=":material/science:")
region_page = st.Page("pages/03_Region_Inspector.py", title="Region Inspector", icon=":material/science:")
tuple_page = st.Page("pages/04_Tuple_Explorer.py", title="Tuple Explorer", icon=":material/science:")
youcook_page = st.Page("pages/05_YouCook2_Evaluation.py", title="YouCook2 Evaluation", icon=":material/science:")
compare_page = st.Page("pages/06_Run_Comparison.py", title="Run Comparison", icon=":material/science:")

nav = st.navigation(
    {
        "Search": [search_page],
        "Developer Tools": [
            legacy_page,
            adaptive_page,
            region_page,
            tuple_page,
            youcook_page,
            compare_page,
        ],
    }
)
nav.run()
