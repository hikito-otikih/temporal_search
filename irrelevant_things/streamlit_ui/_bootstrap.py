"""Shared page bootstrap: env loading, cached clients, shared UI setup.

Every page calls ``bootstrap()`` first.  It guarantees a stable URL set, cached
API client / media resolver, and a consistent page config before any widget is
rendered.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import streamlit as st

from services.api_client import ClientSettings, TemporalApiClient
from services.media_resolver import MediaResolver
from state import keys as K
from state import session_state as SS

APP_VERSION = "0.1.0"

load_dotenv()


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def bootstrap() -> None:
    st.set_page_config(page_title="Temporal Search", page_icon=":material/search:", layout="wide")

    # Must run before init_ui_state(): that function unconditionally seeds
    # these same four keys with hardcoded defaults for any key not yet in
    # session_state, which on a session's first page load is all of them -
    # so if it ran first, the `.env` values below would never be read (the
    # keys would already be truthy from the hardcoded defaults). Using
    # `not in` rather than falsy-checking also means an explicit empty
    # string a user enters in the sidebar (e.g. to clear the media root)
    # sticks, instead of being treated as unset and re-defaulted.
    if K.BACKEND_URL not in st.session_state:
        st.session_state[K.BACKEND_URL] = os.getenv(
            "BACKEND_URL", "http://127.0.0.1:8001"
        )
    if K.UPSTREAM_URL not in st.session_state:
        st.session_state[K.UPSTREAM_URL] = os.getenv(
            "UPSTREAM_URL", "http://127.0.0.1:8000"
        )
    if K.MEDIA_ROOT not in st.session_state:
        st.session_state[K.MEDIA_ROOT] = os.getenv("MEDIA_ROOT", "")
    if K.WORKSPACE not in st.session_state:
        st.session_state[K.WORKSPACE] = os.getenv("WORKSPACE", "default")

    SS.init_ui_state(st.session_state)


def configure_sidebar() -> None:
    with st.sidebar.expander("Connection settings"):
        st.session_state[K.BACKEND_URL] = st.text_input(
            "Backend URL", value=str(st.session_state.get(K.BACKEND_URL, ""))
        )
        st.session_state[K.UPSTREAM_URL] = st.text_input(
            "Upstream URL", value=str(st.session_state.get(K.UPSTREAM_URL, ""))
        )
        st.session_state[K.MEDIA_ROOT] = st.text_input(
            "Media root", value=str(st.session_state.get(K.MEDIA_ROOT, ""))
        )
        st.session_state[K.WORKSPACE] = st.text_input(
            "Workspace", value=str(st.session_state.get(K.WORKSPACE, ""))
        )


def get_api_client() -> TemporalApiClient:
    backend = str(st.session_state.get(K.BACKEND_URL, "http://127.0.0.1:8001"))
    upstream = str(st.session_state.get(K.UPSTREAM_URL, "http://127.0.0.1:8000"))

    def _make(backend_url: str, upstream_url: str) -> TemporalApiClient:
        return TemporalApiClient(
            ClientSettings(backend_url=backend_url, upstream_url=upstream_url)
        )

    return st.cache_resource(_make, show_spinner=False)(backend, upstream)


def get_media_resolver() -> MediaResolver:
    media_root = str(st.session_state.get(K.MEDIA_ROOT, ""))

    def _make(root: str) -> MediaResolver:
        return MediaResolver(root or None)

    return st.cache_resource(_make, show_spinner=False)(media_root)


def default_hyperparameters() -> dict[str, Any]:
    from .models.ui_models import default_hyperparameters

    return default_hyperparameters()
