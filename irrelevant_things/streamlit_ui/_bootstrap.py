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
    st.set_page_config(page_title="Temporal Search Console", layout="wide")
    SS.init_ui_state(st.session_state)

    if not st.session_state.get(K.BACKEND_URL):
        st.session_state[K.BACKEND_URL] = os.getenv(
            "BACKEND_URL", "http://127.0.0.1:8001"
        )
    if not st.session_state.get(K.UPSTREAM_URL):
        st.session_state[K.UPSTREAM_URL] = os.getenv(
            "UPSTREAM_URL", "http://127.0.0.1:8000"
        )
    if not st.session_state.get(K.MEDIA_ROOT):
        st.session_state[K.MEDIA_ROOT] = os.getenv("MEDIA_ROOT", "")
    if not st.session_state.get(K.WORKSPACE):
        st.session_state[K.WORKSPACE] = os.getenv("WORKSPACE", "default")


def configure_sidebar() -> None:
    st.session_state[K.BACKEND_URL] = st.sidebar.text_input(
        "Backend URL", value=str(st.session_state.get(K.BACKEND_URL, ""))
    )
    st.session_state[K.UPSTREAM_URL] = st.sidebar.text_input(
        "Upstream URL", value=str(st.session_state.get(K.UPSTREAM_URL, ""))
    )
    st.session_state[K.MEDIA_ROOT] = st.sidebar.text_input(
        "Media root", value=str(st.session_state.get(K.MEDIA_ROOT, ""))
    )
    st.session_state[K.WORKSPACE] = st.sidebar.text_input(
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
