"""Nested hyperparameter editor with diff, preset reset, and apply.

The form edits a local ``DRAFT_HYPERPARAMETERS`` copy.  Nothing is sent to the
backend until the user presses **Apply & recompute**, and sliders/inputs never
auto-submit on rerun.  ``Diff from session`` shows which changed leaf paths
would be invalidated.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from models.ui_models import preset_by_name
from state import keys as K


def init_draft(store: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    draft = store.get(K.DRAFT_HYPERPARAMETERS)
    if draft is None:
        draft = defaults
        store[K.DRAFT_HYPERPARAMETERS] = draft
    return draft


def render_hyperparameter_editor(
    store: Any,
    *,
    applied: dict[str, Any],
    on_apply: Callable[[dict[str, Any]], Any],
    on_reset_preset: Callable[[str], None] | None = None,
) -> None:
    draft = init_draft(store, applied)

    col1, col2 = st.columns([3, 1])
    with col1:
        preset = st.selectbox(
            "Preset",
            options=["balanced", "quality", "paper_full", "custom"],
            index=0,
            key="hp_preset",
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Reset preset", use_container_width=True):
            store[K.HYPERPARAMETER_PRESET] = preset
            store[K.DRAFT_HYPERPARAMETERS] = preset_by_name(preset) or applied
            st.rerun()

    st.caption("Client-side validation mirrors backend bounds. Nothing is sent "
               "until **Apply & recompute** is pressed.")

    # No "Tuple Ranking" tab: this editor never grew one for
    # TupleRankingHyperparameters (the sole ranking algorithm's own
    # settings) - it only ever exposed the now-removed RankingHyperparameters
    # ("ranking" section, retired with assemble_ordered_tuples). Adjusting
    # tuple_ranking.* currently requires calling PATCH .../hyperparameters
    # directly; a real tab for it is a feature gap, not something this
    # cleanup pass added.
    tabs = st.tabs(["Retrieval", "Clustering", "Refinement", "Boundary"])

    with tabs[0]:
        _retrieval_tab(draft)
    with tabs[1]:
        _clustering_tab(draft)
    with tabs[2]:
        _refinement_tab(draft)
    with tabs[3]:
        _boundary_tab(draft)

    col1, col2, col3 = st.columns(3)
    if col1.button("Diff from session", use_container_width=True):
        _show_diff(draft, applied)
    run_label = col2.text_input("Run label", value="", key="run_label_input",
                                label_visibility="collapsed", placeholder="run label/note")
    if col3.button("Apply & recompute", use_container_width=True, type="primary"):
        on_apply(dict(draft))


def _show_diff(draft: dict[str, Any], applied: dict[str, Any]) -> None:
    changed = _changed_paths(applied, draft)
    if not changed:
        st.success("No differences from the applied session hyperparameters.")
        return
    st.warning("Would invalidate these leaf paths:")
    st.code("\n".join(sorted(changed)))


def _changed_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(child)
            else:
                paths.extend(_changed_paths(before[key], after[key], child))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        return [prefix or "$"] if before != after else []
    return [prefix or "$"] if before != after else []


def _number(prefix: str, label: str, value: Any, min_value: float, max_value: float, step: float) -> Any:
    return st.number_input(
        label, value=float(value), min_value=min_value, max_value=max_value,
        step=step, key=f"{prefix}_{label}", format="%.4g",
    )


def _bool(prefix: str, label: str, value: Any) -> bool:
    return st.checkbox(label, value=bool(value), key=f"{prefix}_{label}")


def _retrieval_tab(draft: dict[str, Any]) -> None:
    section = draft.setdefault("retrieval", {})
    c1, c2 = st.columns(2)
    with c1:
        section["top_n_per_variant"] = int(_number("r", "top_n_per_variant", section.get("top_n_per_variant", 50), 1, 10_000, 1))
        section["top_n_fused"] = int(_number("r", "top_n_fused", section.get("top_n_fused", 100), 1, 10_000, 1))
    with c2:
        section["rrf_k"] = int(_number("r", "rrf_k", section.get("rrf_k", 60), 1, 10_000, 1))
        section["query_variants_per_event"] = int(_number("r", "query_variants_per_event", section.get("query_variants_per_event", 4), 1, 32, 1))


def _clustering_tab(draft: dict[str, Any]) -> None:
    section = draft.setdefault("clustering", {})
    section["gap_seconds"] = float(_number("c", "gap_seconds", section.get("gap_seconds", 3.0), 0.0, 3600.0, 0.1))
    section["margin_seconds"] = float(_number("c", "margin_seconds", section.get("margin_seconds", 3.0), 0.0, 3600.0, 0.1))
    section["max_region_seconds"] = float(_number("c", "max_region_seconds", section.get("max_region_seconds", 30.0), 0.1, 3600.0, 1.0))
    if section["max_region_seconds"] < 2.0 * section["margin_seconds"]:
        st.warning("max_region_seconds must be >= 2 * margin_seconds")


def _refinement_tab(draft: dict[str, Any]) -> None:
    section = draft.setdefault("refinement", {})
    section["quality_profile_enabled"] = _bool("f", "quality_profile_enabled", section.get("quality_profile_enabled", False))
    section["use_reranker"] = _bool("f", "use_reranker", section.get("use_reranker", False))
    c1, c2 = st.columns(2)
    with c1:
        section["max_initial_videos"] = int(_number("f", "max_initial_videos", section.get("max_initial_videos", 5), 1, 1000, 1))
        section["max_regions_per_event_per_video"] = int(_number("f", "max_regions_per_event_per_video", section.get("max_regions_per_event_per_video", 3), 1, 100, 1))
        section["max_total_regions"] = int(_number("f", "max_total_regions", section.get("max_total_regions", 60), 1, 10_000, 1))
    with c2:
        section["exploration_region_ratio"] = float(_number("f", "exploration_region_ratio", section.get("exploration_region_ratio", 0.15), 0.0, 1.0, 0.01))
    c1, c2, c3 = st.columns(3)
    with c1:
        section["video_coverage_weight"] = float(_number("f", "video_coverage_weight", section.get("video_coverage_weight", 0.5), 0.0, 1.0, 0.05))
    with c2:
        section["video_mean_weight"] = float(_number("f", "video_mean_weight", section.get("video_mean_weight", 0.3), 0.0, 1.0, 0.05))
    with c3:
        section["video_min_weight"] = float(_number("f", "video_min_weight", section.get("video_min_weight", 0.2), 0.0, 1.0, 0.05))


def _boundary_tab(draft: dict[str, Any]) -> None:
    section = draft.setdefault("boundary", {})
    c1, c2 = st.columns(2)
    with c1:
        section["window_options_seconds"] = [float(x) for x in st.text_input(
            "window_options_seconds (comma separated)", value=",".join(
                str(x) for x in section.get("window_options_seconds", [0.5, 1.0, 2.0, 3.0])
            ), key="b_window_options",
        ).split(",") if x.strip()]
        section["min_samples_per_side"] = int(_number("b", "min_samples_per_side", section.get("min_samples_per_side", 2), 1, 100, 1))
        section["pairwise_temperature"] = float(_number("b", "pairwise_temperature", section.get("pairwise_temperature", 1.0), 0.01, 100.0, 0.01))
        section["anchor_clip_z"] = float(_number("b", "anchor_clip_z", section.get("anchor_clip_z", 8.0), 0.1, 100.0, 0.1))
    with c2:
        section["nms_radius_seconds"] = float(_number("b", "nms_radius_seconds", section.get("nms_radius_seconds", 0.5), 0.0, 100.0, 0.05))
        section["max_proposals_per_region"] = int(_number("b", "max_proposals_per_region", section.get("max_proposals_per_region", 5), 1, 100, 1))
    st.subheader("Contrast weights (must sum > 0)")
    c1, c2, c3 = st.columns(3)
    with c1:
        section["post_contrast_weight"] = float(_number("b", "post_contrast_weight", section.get("post_contrast_weight", 0.5), 0.0, 1.0, 0.05))
    with c2:
        section["pre_contrast_weight"] = float(_number("b", "pre_contrast_weight", section.get("pre_contrast_weight", 0.5), 0.0, 1.0, 0.05))
    with c3:
        section["motion_contrast_weight"] = float(_number("b", "motion_contrast_weight", section.get("motion_contrast_weight", 0.1), 0.0, 1.0, 0.05))
    st.subheader("Score weights (must sum > 0)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        section["semantic_weight"] = float(_number("b", "semantic_weight", section.get("semantic_weight", 0.4), 0.0, 1.0, 0.05))
    with c2:
        section["boundary_weight"] = float(_number("b", "boundary_weight", section.get("boundary_weight", 0.3), 0.0, 1.0, 0.05))
    with c3:
        section["pre_weight"] = float(_number("b", "pre_weight", section.get("pre_weight", 0.1), 0.0, 1.0, 0.05))
    with c4:
        section["post_weight"] = float(_number("b", "post_weight", section.get("post_weight", 0.2), 0.0, 1.0, 0.05))
    c1, c2 = st.columns(2)
    with c1:
        section["window_length_regularization"] = float(_number("b", "window_length_regularization", section.get("window_length_regularization", 0.01), 0.0, 1.0, 0.01))
    with c2:
        section["window_asymmetry_regularization"] = float(_number("b", "window_asymmetry_regularization", section.get("window_asymmetry_regularization", 0.01), 0.0, 1.0, 0.01))


