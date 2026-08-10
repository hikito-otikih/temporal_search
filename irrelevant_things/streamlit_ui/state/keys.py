"""Canonical ``st.session_state`` keys.

Keeping the keys in one module prevents silent key collisions between pages.
Only UI-relevant data is stored; large artifacts stay server-side and are
re-fetched per page/limit.
"""

from __future__ import annotations

# Configuration -----------------------------------------------------------
BACKEND_URL = "backend_url"
UPSTREAM_URL = "upstream_url"
MEDIA_ROOT = "media_root"
WORKSPACE = "workspace"

# Capability --------------------------------------------------------------
CAPABILITIES = "capabilities"

# Active adaptive session -------------------------------------------------
ACTIVE_SESSION_ID = "active_session_id"
SESSION_REVISION = "session_revision"
SESSION_RESPONSE = "session_response"
SESSION_DIRTY = "session_dirty"

# Hyperparameters ---------------------------------------------------------
DRAFT_HYPERPARAMETERS = "draft_hyperparameters"
APPLIED_HYPERPARAMETERS = "applied_hyperparameters"
HYPERPARAMETER_PRESET = "hyperparameter_preset"
RUN_LABEL = "run_label"

# Event drafts (pre-session) ----------------------------------------------
EVENT_DRAFTS = "event_drafts"
COMMON_QUERY = "common_query"

# Selections ---------------------------------------------------------------
SELECTED_VIDEO_ID = "selected_video_id"
SELECTED_EVENT_ID = "selected_event_id"
SELECTED_REGION_ID = "selected_region_id"
SELECTED_PROPOSAL_ID = "selected_proposal_id"
SELECTED_TUPLE_ID = "selected_tuple_id"

# View filters (view only, never sent as backend constraints) --------------
VIEW_FILTERS = "view_filters"
PINNED_VIDEOS = "pinned_videos"
COMPARE_VIDEOS = "compare_videos"

# Errors -------------------------------------------------------------------
LAST_ERROR = "last_error"
LAST_WARNING = "last_warning"
