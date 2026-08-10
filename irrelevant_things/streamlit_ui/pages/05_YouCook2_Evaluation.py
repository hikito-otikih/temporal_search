"""05 YouCook2 Retrieval Evaluation.

MVP measures corpus-level Video Recall@K only.  Ground truth lives exclusively
in the evaluator: the retrieval payload sent to ``/search`` contains only event
text and top-K (see ``benchmark_client.build_retrieval_payload``).  Frame hits
are deduplicated per video before ranking.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client
from components import api_status
from components.json_inspector import render_sanitized_payload
from services.benchmark_client import (
    BenchmarkRun,
    ManifestError,
    build_retrieval_payload,
    parse_manifest,
    rank_of_ground_truth,
    rank_videos,
)
from services.export_service import build_run_manifest, csv_bytes, json_bytes
from services.api_client import ApiError
from state import keys as K
from state import session_state as SS

bootstrap()
configure_sidebar()

client = get_api_client()
SS.init_ui_state(st.session_state)

st.title("YouCook2 Retrieval Evaluation")
st.caption("Video Recall@K (corpus level). Dense localization and tIoU are out of scope for this MVP.")

st.markdown("""
### Leakage guard
- `video_path` from the manifest is parsed into `ground_truth_video_id` **only in the evaluator**.
- Retrieval payloads contain exactly `{"query": "...", "top_k": N}` — never the video path or hints.
- Frame hits are deduplicated by video before Recall@K is computed.
""")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

with st.expander("Query manifest", expanded=True):
    uploaded = st.file_uploader("Upload query manifest (JSON)", type=["json"], key="benchmark_manifest_upload")
    paste = st.text_area("Or paste manifest JSON", height=180, key="benchmark_manifest_paste")
    manifest_source = st.session_state.get("benchmark_manifest_data")

if uploaded is not None:
    try:
        manifest_source = json.loads(uploaded.read().decode("utf-8"))
        st.session_state["benchmark_manifest_data"] = manifest_source
    except ValueError as exc:
        st.error(f"Invalid JSON: {exc}")

try:
    queries = parse_manifest(manifest_source) if manifest_source is not None else []
    manifest_error = None
except ManifestError as exc:
    queries = []
    manifest_error = str(exc)

if manifest_error:
    st.error(manifest_error)
elif manifest_source is not None:
    with_gt = sum(1 for q in queries if q.ground_truth_video_id)
    st.caption(f"Parsed {len(queries)} queries · {with_gt} with ground-truth video")

c1, c2, c3, c4 = st.columns(4)
corpus_name = c1.text_input("Corpus/index name", value="youcook2", key="benchmark_corpus")
split = c2.selectbox("Split", options=["validation", "training", "test"], index=0, key="benchmark_split")
model_variant = c3.selectbox("Model variant", options=["balanced", "quality", "paper_full"], key="benchmark_model")
limit_queries = int(c4.number_input("Max queries (0 = all)", value=10, min_value=0, key="benchmark_limit"))

c1, c2, c3, c4 = st.columns(4)
frame_top_n = int(c1.number_input("Frame top-N", value=100, min_value=1, max_value=10_000, key="benchmark_topn"))
method = c2.selectbox("Video aggregation", options=["max", "mean_top_m", "logsumexp", "rrf"], index=0, key="benchmark_method")
top_m = int(c3.number_input("Mean top-M", value=3, min_value=1, max_value=50, key="benchmark_topm"))
k_list = [int(x.strip()) for x in c4.text_input("K list", value="1,5,10,20,50", key="benchmark_kl").split(",") if x.strip().isdigit()]

run_name = st.text_input("Run name", value="", key="benchmark_run_name")
note = st.text_area("Note", height=60, key="benchmark_note")

st.subheader("Sanitized retrieval payload")
render_sanitized_payload(build_retrieval_payload("example event query", frame_top_n))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _init_run(queries: list[Any]) -> BenchmarkRun:
    return BenchmarkRun(
        queries=queries,
        method=method,
        top_m=top_m,
        frame_top_n=frame_top_n,
        k_list=k_list,
    )

if st.button("Run evaluation", type="primary", disabled=not queries):
    selected = queries[:limit_queries] if limit_queries > 0 else queries
    if not selected:
        st.error("No queries to run.")
    else:
        run = _init_run(selected)
        pending = run.resume()
        progress = st.progress(0.0, text="Retrieving corpus candidates…")
        status = st.empty()
        failures = 0
        for index, item in enumerate(pending):
            status.write(f"({index + 1}/{len(pending)}) {item.query[:70]}")
            try:
                run.run_query(item, search=lambda q, topk: client.upstream_search(q, topk).get("results", []))
            except ApiError as exc:
                failures += 1
                run.record(item.index, [], {"payload": build_retrieval_payload(item.query, frame_top_n), "frame_hits": 0, "error": exc.message})
            progress.progress((index + 1) / max(1, len(pending)))
        status.write("")
        progress.empty()
        if failures:
            st.warning(f"{failures} queries failed to retrieve (recorded as miss).")
        st.session_state["benchmark_run"] = run

run: BenchmarkRun | None = st.session_state.get("benchmark_run")
if run is None and queries:
    run = _init_run(queries[:limit_queries] if limit_queries > 0 else queries)
    st.session_state["benchmark_run"] = run

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

if run is not None and run.completed:
    summary = run.summary()
    st.subheader("Metrics")
    recall = summary["recall"]
    cards = st.columns(len(k_list) + 3)
    for column, k in zip(cards, k_list):
        column.metric(f"Recall@{k}", f"{recall.get(k, 0.0):.3f}")
    cards[len(k_list)].metric("MRR", f"{summary['mrr']:.3f}")
    cards[len(k_list) + 1].metric("Median rank", f"{summary['median_rank']:.1f}" if summary["median_rank"] is not None else "—")
    cards[len(k_list) + 2].metric("Missing GT", summary["missing_ground_truth"])

    st.subheader("Recall curve")
    import plotly.graph_objects as go
    ks = sorted(recall)
    figure = go.Figure(
        go.Scatter(x=ks, y=[recall[k] for k in ks], mode="lines+markers")
    )
    figure.update_layout(xaxis_title="K", yaxis_title="Recall@K", height=300)
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Per-query results")
    rows = run.per_query_rows()
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

    with st.expander("Query detail"):
        detail_index = st.number_input("Query index", min_value=0, max_value=max(0, len(run.queries) - 1), value=0, step=1, key="benchmark_detail_index")
        record = run.completed.get(int(detail_index))
        if record:
            item = next(q for q in run.queries if q.index == int(detail_index))
            st.write(f"**Query:** {item.query}")
            st.write(f"**Ground truth video:** `{item.ground_truth_video_id}`")
            rank = record["rank"]
            st.write(f"**Rank:** {rank if rank else 'missing'}")
            ranked = record["ranked_videos"]
            top_videos = ranked[:20]
            st.write(f"**Top videos:** {', '.join(top_videos) if top_videos else 'none'}")
            gt_position = rank_of_ground_truth(ranked, item.ground_truth_video_id) if item.ground_truth_video_id else None
            if gt_position is None and item.ground_truth_video_id:
                st.warning("Ground-truth video is missing from ranked results.")

    st.subheader("Failure buckets")
    _render_failure_buckets(run)

    st.subheader("Export")
    manifest = build_run_manifest(
        run_summary=summary,
        per_query=rows,
        config={
            "corpus": corpus_name,
            "split": split,
            "model_variant": model_variant,
            "method": method,
            "frame_top_n": frame_top_n,
            "k_list": k_list,
            "run_name": run_name,
            "note": note,
        },
    )
    c1, c2 = st.columns(2)
    c1.download_button("Download run_manifest.json", data=json_bytes(manifest), file_name="run_manifest.json", mime="application/json")
    c2.download_button("Download per_query.csv", data=csv_bytes(rows), file_name="per_query.csv", mime="text/csv")

st.markdown("---")
api_status.show_api_status(st.session_state, client)


def _render_failure_buckets(run: BenchmarkRun) -> None:
    buckets: dict[str, int] = {
        "missing corpus item": 0,
        "translation failure": 0,
        "low score / wrong recipe": 0,
        "duplicate domination": 0,
    }
    for item in run.queries:
        record = run.completed.get(item.index)
        if record is None:
            continue
        if record["rank"] is None and item.ground_truth_video_id:
            if record["raw"].get("error"):
                buckets["translation failure"] += 1
            elif not record["ranked_videos"]:
                buckets["missing corpus item"] += 1
            else:
                buckets["low score / wrong recipe"] += 1
    st.json(buckets)
