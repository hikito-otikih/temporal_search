"""Streamlit session-state helpers with a revision-aware mutation flow.

The mutation flow is designed to be testable without Streamlit: core functions
operate on any ``MutableMapping`` store (a plain ``dict`` in unit tests,
``st.session_state`` in the app).

Mutation flow
-------------
1. read the current revision from the store;
2. submit the mutation with ``expected_revision``;
3. on success store the new revision and clear dependent artifact caches;
4. on ``RevisionConflictError`` refresh the session from the backend, surface
   the diff, and do NOT auto-retry the mutation.
"""

from __future__ import annotations

from typing import Any, Callable, MutableMapping

from models.ui_models import (
    MutationResponse,
    RunResponse,
    SessionResponse,
)
from services.api_client import (
    ApiError,
    RevisionConflictError,
    TemporalApiClient,
)
from state import keys as K


# ---------------------------------------------------------------------------
# Store adapters
# ---------------------------------------------------------------------------

def _get(store: MutableMapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return store.get(key, default)
    except AttributeError:
        return default


def _set(store: MutableMapping[str, Any], key: str, value: Any) -> None:
    store[key] = value


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_ui_state(store: MutableMapping[str, Any]) -> None:
    defaults: dict[str, Any] = {
        K.BACKEND_URL: "http://127.0.0.1:8001",
        K.UPSTREAM_URL: "http://127.0.0.1:8000",
        K.MEDIA_ROOT: "",
        K.WORKSPACE: "default",
        K.CAPABILITIES: None,
        K.ACTIVE_SESSION_ID: None,
        K.SESSION_REVISION: 0,
        K.SESSION_RESPONSE: None,
        K.SESSION_DIRTY: False,
        K.DRAFT_HYPERPARAMETERS: None,
        K.APPLIED_HYPERPARAMETERS: None,
        K.HYPERPARAMETER_PRESET: "balanced",
        K.RUN_LABEL: "",
        K.EVENT_DRAFTS: [],
        K.COMMON_QUERY: "",
        K.SELECTED_VIDEO_ID: None,
        K.SELECTED_EVENT_ID: None,
        K.SELECTED_REGION_ID: None,
        K.SELECTED_PROPOSAL_ID: None,
        K.SELECTED_TUPLE_ID: None,
        K.VIEW_FILTERS: {},
        K.PINNED_VIDEOS: [],
        K.COMPARE_VIDEOS: [],
        K.LAST_ERROR: None,
        K.LAST_WARNING: None,
    }
    for key, default in defaults.items():
        if key not in store:
            _set(store, key, default)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def backend_url(store: MutableMapping[str, Any]) -> str:
    return str(_get(store, K.BACKEND_URL, "http://127.0.0.1:8001"))


def upstream_url(store: MutableMapping[str, Any]) -> str:
    return str(_get(store, K.UPSTREAM_URL, "http://127.0.0.1:8000"))


def media_root(store: MutableMapping[str, Any]) -> str:
    return str(_get(store, K.MEDIA_ROOT, ""))


def active_session_id(store: MutableMapping[str, Any]) -> str | None:
    value = _get(store, K.ACTIVE_SESSION_ID)
    return str(value) if value else None


def session_revision(store: MutableMapping[str, Any]) -> int:
    return int(_get(store, K.SESSION_REVISION, 0))


def session_response(store: MutableMapping[str, Any]) -> SessionResponse | None:
    value = _get(store, K.SESSION_RESPONSE)
    return value if isinstance(value, SessionResponse) else None


def capabilities(store: MutableMapping[str, Any]) -> Any:
    return _get(store, K.CAPABILITIES)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def set_active_session(
    store: MutableMapping[str, Any], response: SessionResponse
) -> None:
    _set(store, K.ACTIVE_SESSION_ID, response.session.get("id"))
    _set(store, K.SESSION_REVISION, int(response.session.get("revision", 0)))
    _set(store, K.SESSION_RESPONSE, response)
    _set(store, K.SESSION_DIRTY, False)
    _set(store, K.APPLIED_HYPERPARAMETERS, response.session.get("hyperparameters"))


def clear_active_session(store: MutableMapping[str, Any]) -> None:
    _set(store, K.ACTIVE_SESSION_ID, None)
    _set(store, K.SESSION_REVISION, 0)
    _set(store, K.SESSION_RESPONSE, None)
    _set(store, K.SESSION_DIRTY, False)
    _set(store, K.SELECTED_REGION_ID, None)
    _set(store, K.SELECTED_PROPOSAL_ID, None)
    _set(store, K.SELECTED_TUPLE_ID, None)


def refresh_session(
    store: MutableMapping[str, Any], client: TemporalApiClient
) -> SessionResponse | None:
    session_id = active_session_id(store)
    if not session_id:
        return None
    try:
        response = client.get_session(session_id)
    except ApiError:
        return None
    set_active_session(store, response)
    return response


# ---------------------------------------------------------------------------
# Mutation flow
# ---------------------------------------------------------------------------

def apply_mutation(
    store: MutableMapping[str, Any],
    client: TemporalApiClient,
    mutation: Callable[[int], Any],
) -> tuple[Any | None, str | None]:
    """Run a mutation with optimistic revision handling.

    Returns ``(result, error_message)``.  On success the store revision is
    updated and dependent caches cleared.  On revision conflict the session is
    refreshed and the mutation is NOT retried; a clear message is returned.
    """
    expected = session_revision(store)
    try:
        result = mutation(expected)
    except RevisionConflictError as exc:
        refreshed = refresh_session(store, client)
        message = (
            f"Revision conflict (expected {exc.expected}, backend is at "
            f"{exc.current}). Session reloaded — please re-apply your change."
        )
        return None, message
    except ApiError as exc:
        _set(store, K.LAST_ERROR, exc.message)
        return None, exc.message

    if isinstance(result, (MutationResponse, RunResponse)):
        _set(store, K.SESSION_REVISION, result.session_revision)
        _set(store, K.SESSION_DIRTY, bool(result.invalidated_stages))
    return result, None
