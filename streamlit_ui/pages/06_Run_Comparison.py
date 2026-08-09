"""06 Run Comparison.

Compare two or more runs (or hyperparameter presets).  Runs carry the full
hyperparameter/constraint snapshot at execution time, so comparisons always
include model revision, preprocessing, and parameters.  Legacy raw scores and
adaptive normalized scores are never drawn on the same axis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client
from components import api_status
from components.hyperparameter_editor import _changed_paths
from models.ui_models import PRESET_HYPERPARAMETERS, preset_by_name
from services.export_service import csv_bytes, json_bytes
from state import keys as K
from state import session_state as SS

bootstrap()
configure_sidebar()

client = get_api_client()
SS.init_ui_state(st.session_state)

st.title("Run Comparison")

session_id = SS.active_session_id(st.session_state)
runs = []
if session_id:
    runs = client.get_runs(session_id, limit=200).get("items", [])

st.subheader("Runs from active session")
if runs:
    options = {run.get("id"): run for run in runs}
    selected_ids = st.multiselect("Compare runs", options=list(options), key="compare_run_select")
    if selected_ids:
        selected = [options[item] for item in selected_ids]
        _render_run_comparison(selected)
else:
    st.caption("No runs in the active session. Runs appear after candidate/frame-score ingest.")

st.divider()
st.subheader("Hyperparameter presets")
preset_a = st.selectbox("Preset A", options=list(PRESET_HYPERPARAMETERS), index=0, key="preset_a")
preset_b = st.selectbox("Preset B", options=list(PRESET_HYPERPARAMETERS), index=2, key="preset_b")
if st.button("Compare presets", key="compare_presets"):
    a = preset_by_name(preset_a) or PRESET_HYPERPARAMETERS["balanced"]
    b = preset_by_name(preset_b) or PRESET_HYPERPARAMETERS["balanced"]
    _render_preset_diff(preset_a, preset_b, a, b)

st.markdown("---")
api_status.show_api_status(st.session_state, client)


def _render_run_comparison(runs: list[dict[str, Any]]) -> None:
    st.write("Runs are compared on their own snapshots; scores are NOT merged across runs.")
    metrics_rows = []
    for run in runs:
        row = {
            "run_id": run.get("id")[:8],
            "status": run.get("status"),
            "revision": run.get("session_revision"),
        }
        row.update(run.get("metrics", {}))
        metrics_rows.append(row)
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, height=240)

    if len(runs) == 2:
        a, b = runs
        a_hp = a.get("hyperparameter_snapshot", {})
        b_hp = b.get("hyperparameter_snapshot", {})
        changed = _changed_paths(a_hp, b_hp)
        st.write("Hyperparameter diffs (A → B):")
        if not changed:
            st.success("No hyperparameter differences between the two runs.")
        else:
            st.code("\n".join(sorted(changed)))

    export = {
        "runs": [
            {
                "run_id": run.get("id"),
                "session_revision": run.get("session_revision"),
                "status": run.get("status"),
                "metrics": run.get("metrics", {}),
                "hyperparameters": run.get("hyperparameter_snapshot", {}),
                "constraints": run.get("constraint_snapshot", {}),
            }
            for run in runs
        ]
    }
    st.download_button("Download comparison JSON", data=json_bytes(export), file_name="run_comparison.json", mime="application/json")


def _render_preset_diff(name_a: str, name_b: str, a: dict[str, Any], b: dict[str, Any]) -> None:
    changed = _changed_paths(a, b)
    st.write(f"**{name_a}** vs **{name_b}**")
    if not changed:
        st.success("Presets are identical.")
        return
    rows = []
    for path in sorted(changed):
        rows.append({"parameter": path, name_a: _value_at(a, path), name_b: _value_at(b, path)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)
    st.download_button(
        "Download preset diff CSV", data=csv_bytes(rows), file_name="preset_diff.csv", mime="text/csv"
    )


def _value_at(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        if isinstance(node, dict):
            node = node.get(part, "—")
        else:
            return "—"
    return node
