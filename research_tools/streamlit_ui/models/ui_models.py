"""Typed UI-facing models and hyperparameter presets.

The backend remains the source of truth for validation.  These models only
help the UI read API payloads safely and keep widget defaults in sync with the
backend seed configuration in ``docs/ADAPTIVE_TEMPORAL_SEARCH.md``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class UiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Capability discovery
# ---------------------------------------------------------------------------

class FrameProviderCapability(UiModel):
    available: bool = False
    supports_batch: bool = False
    supports_dense_sampling: bool = False
    supports_thumbnails: bool = False
    reason: str | None = None


class SearcherCapability(UiModel):
    searcher_type: str
    available: bool
    session_api: bool = False
    ordered_events: bool = True
    endpoint: str | None = None
    legacy_request_value: str | None = None
    algorithm_core_available: bool = False
    live_refinement_available: bool = False
    frame_provider: FrameProviderCapability = Field(default_factory=FrameProviderCapability)
    runtime_model: str | None = None
    quality_embedding_model: str | None = None
    quality_reranker_model: str | None = None
    supported_relations: list[str] = Field(default_factory=list)
    supported_boundary_profiles: list[str] = Field(default_factory=list)


class CapabilityResponse(UiModel):
    searchers: list[SearcherCapability] = Field(default_factory=list)

    def adaptive(self) -> SearcherCapability | None:
        return next(
            (item for item in self.searchers if item.searcher_type == "adaptive_temporal"),
            None,
        )

    def legacy(self) -> SearcherCapability | None:
        return next(
            (item for item in self.searchers if item.session_api is False),
            None,
        )


# ---------------------------------------------------------------------------
# Session / artifact summary models
# ---------------------------------------------------------------------------

class ArtifactCounts(UiModel):
    candidates: int = 0
    regions: int = 0
    proposals: int = 0
    runs: int = 0


class SessionSummary(UiModel):
    id: str
    revision: int = 0
    searcher_type: str = "adaptive_temporal"
    status: str = "draft"
    event_count: int = 0
    last_invalidated_stages: list[str] = Field(default_factory=list)


class SessionResponse(UiModel):
    session: dict[str, Any] = Field(default_factory=dict)
    artifact_counts: ArtifactCounts = Field(default_factory=ArtifactCounts)
    live_refinement: dict[str, Any] = Field(default_factory=dict)


class MutationResponse(UiModel):
    session_revision: int = 0
    invalidated_stages: list[str] = Field(default_factory=list)
    artifact_counts: ArtifactCounts = Field(default_factory=ArtifactCounts)


class RunResponse(UiModel):
    session_revision: int = 0
    run_id: str
    run_status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_counts: ArtifactCounts = Field(default_factory=ArtifactCounts)


PageResponse = TypeAdapter(dict)


# ---------------------------------------------------------------------------
# Hyperparameter presets
# ---------------------------------------------------------------------------

def _balanced_preset() -> dict[str, Any]:
    return {
        "retrieval": {
            "top_n_per_variant": 500,
            "top_n_fused": 1000,
            "rrf_k": 60,
            "query_variants_per_event": 4,
        },
        "refinement": {
            "quality_profile_enabled": False,
            "use_reranker": False,
            "video_coverage_weight": 0.0,
            "video_mean_weight": 1.0,
            "video_min_weight": 0.0,
            "video_distinctness_weight": 0.0,
            "distinctness_norm_seconds": 3.0,
        },
        "boundary": {
            "window_options_seconds": [0.5, 1.0, 2.0, 3.0],
            "min_samples_per_side": 2,
            "pairwise_temperature": 1.0,
            "anchor_clip_z": 8.0,
            "post_contrast_weight": 0.5,
            "pre_contrast_weight": 0.5,
            "motion_contrast_weight": 0.1,
            "window_length_regularization": 0.01,
            "window_asymmetry_regularization": 0.01,
            "semantic_weight": 0.4,
            "boundary_weight": 0.3,
            "pre_weight": 0.1,
            "post_weight": 0.2,
            "nms_radius_seconds": 0.5,
            "max_proposals_per_region": 5,
        },
    }


def _quality_preset() -> dict[str, Any]:
    # use_reranker=True is deliberately not set here: the backend rejects it
    # unconditionally (ValueError - "not supported until the ordered-frame
    # reranker runtime is implemented"), so setting it made this preset 422
    # on every Apply. quality_profile_enabled=True is the supported half of
    # "quality" - it switches to quality_embedding_model.
    preset = _balanced_preset()
    preset["refinement"]["quality_profile_enabled"] = True
    return preset


def _paper_full_preset() -> dict[str, Any]:
    preset = _quality_preset()
    preset["retrieval"]["top_n_per_variant"] = 1000
    preset["retrieval"]["top_n_fused"] = 2000
    return preset


PRESET_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "balanced": _balanced_preset(),
    "quality": _quality_preset(),
    "paper_full": _paper_full_preset(),
    "custom": {},
}


def preset_by_name(name: str) -> dict[str, Any]:
    key = name if name in PRESET_HYPERPARAMETERS else "balanced"
    if key == "custom":
        return {}
    return deepcopy(PRESET_HYPERPARAMETERS[key])


def default_hyperparameters() -> dict[str, Any]:
    """Seed configuration matching the backend defaults."""
    return deepcopy(PRESET_HYPERPARAMETERS["balanced"])


# ---------------------------------------------------------------------------
# Event draft helpers
# ---------------------------------------------------------------------------

BOUNDARY_TYPES: tuple[str, ...] = (
    "onset",
    "offset",
    "transition",
    "plateau_start",
    "symmetric_peak",
    "state",
    "unknown",
)

_TRANSITION_PROFILES = {"onset", "offset", "transition", "plateau_start"}


def new_event_draft(index: int) -> dict[str, Any]:
    return {
        "event_id": f"event_{index}",
        "original_query": "",
        "anchor_query": "",
        "pre_state": "",
        "post_state": "",
        "boundary_type": "transition",
    }


def event_draft_to_definition(draft: dict[str, Any]) -> dict[str, Any]:
    """Convert a UI draft (empty strings allowed) to an API event definition."""
    definition = {
        "event_id": draft["event_id"].strip(),
        "original_query": draft["original_query"].strip(),
        "anchor_query": draft["anchor_query"].strip(),
        "pre_state": _nullable_text(draft.get("pre_state")),
        "post_state": _nullable_text(draft.get("post_state")),
        "boundary_type": draft.get("boundary_type") or "unknown",
    }
    return definition


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def event_completeness_issue(draft: dict[str, Any]) -> str | None:
    """Return a human readable completeness warning for an event draft."""
    boundary = draft.get("boundary_type") or "unknown"
    pre = str(draft.get("pre_state") or "").strip()
    post = str(draft.get("post_state") or "").strip()
    if boundary in _TRANSITION_PROFILES and (not pre or not post):
        return f"boundary_type='{boundary}' requires both pre_state and post_state"
    if not str(draft.get("original_query") or "").strip():
        return "original_query is required"
    if not str(draft.get("anchor_query") or "").strip():
        return "anchor_query is required"
    return None


def format_timestamp(seconds: float) -> str:
    """Render float seconds as ``m:ss.cc`` for display only."""
    whole = max(0, int(seconds))
    minutes, secs = divmod(whole, 60)
    centis = int(round((seconds - whole) * 100))
    return f"{minutes}:{secs:02d}.{centis:02d}"


def normalize_adaptive_items(page: dict[str, Any], event_count: int) -> list[dict[str, Any]]:
    """Shape a GET /video-priorities page into the flat item dicts the
    Search page's result list and its constraint-mutation actions
    (reject/prioritize/fix-frame) both render/re-render from. Shared here
    (rather than living in the page) so those actions - which live in
    components/search_constraints.py - can re-normalize a fresh page after
    a mutation without importing back into the page module."""
    items = []
    for entry in page.get("items", []):
        boundary = entry.get("boundary_refinement") or {}
        moments = []
        if boundary.get("status") == "applied":
            for position, event in enumerate(boundary.get("events") or [], start=1):
                moments.append(
                    {
                        "label": f"moment {position}",
                        "event_id": event.get("event_id"),
                        "seconds": event.get("refined_seconds"),
                        "refined": not event.get("used_fallback", False),
                        "source": event.get("source", "auto"),
                    }
                )
        video_id = entry.get("video_id")
        items.append(
            {
                "video_id": video_id,
                "reject_key": video_id,
                "coverage_label": f"Matched {entry.get('event_coverage', 0)}/{event_count} moments",
                "priority_score": entry.get("priority_score"),
                "moments": moments,
            }
        )
    return items
