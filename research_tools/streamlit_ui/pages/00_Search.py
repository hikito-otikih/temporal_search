"""00 Search — the primary, consumer-facing search experience.

Type what you're looking for (one moment per line, in order) and get back
ranked videos with the matched moment's timestamp, a thumbnail, and
on-demand playback. Everything else in this app ("Developer Tools") is for
inspecting pipeline internals; this page is the product.

Orchestration only - the reusable, independently-tested pieces live in
components/search_*.py:
- search_rewrite_preview: preview + making it authoritative for Search
- search_retrieval_controls: "Advanced options"
- search_constraints: reject/prioritize/fix-frame session mutations
- search_keyframe_browser: the two manual frame-picking UIs
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client, get_media_resolver
from components.api_status import show_connection_error
from components.search_constraints import prioritize_item, reject_item
from components.search_keyframe_browser import render_event_keyframe_browser, render_frame_fixer
from components.search_retrieval_controls import render_retrieval_controls
from components.search_rewrite_preview import get_authoritative_analysis, render_preview_controls
from models.ui_models import format_timestamp, normalize_adaptive_items
from services.api_client import ApiError, ConnectionFailure, TemporalApiClient
from services.media_resolver import MediaResolver
from state import keys as K

RETRIEVE_TOP_K = 50
# legacy has no session/rank-again server call, so it over-fetches once and
# "reject" just reveals the next already-fetched item client-side.
LEGACY_REJECT_BUFFER = 20
LEGACY_MAX_FETCH = 100

PIPELINES: dict[str, str] = {
    "Adaptive (recommended)": "adaptive_coarse",
    "Legacy — ordered": "legacy_temporal",
    "Legacy — flexible": "legacy_ambiguous",
}


# Same query-block format the YouCook2 benchmark's real query files use
# (research_tools/benchmarks/youcook2/core.py: EVENT_RE/ANSWER_RE) - lets
# people paste a whole "context + E1: ... E2: ..." block straight in instead
# of retyping it as one moment per line.
EVENT_LINE = re.compile(r"^\s*E\d+\s*:\s*(.*?)\s*$", re.IGNORECASE)
ANSWER_MARKER = re.compile(r"^\s*\*\*\s*Answer\s*$", re.IGNORECASE)


def _parse_query_block(text: str) -> tuple[str | None, list[str]]:
    """Accept either plain one-moment-per-line text, or a full YouCook2-style
    query block. Returns (common_query, ordered moment strings).

    If any "E<n>: ..." lines are found, every other non-empty line before
    them is treated as shared context (common_query) and stripped from the
    moment list. A stray "**Answer" section (if someone pastes a whole
    ground-truth query file) is ignored. Otherwise falls back to treating
    every non-empty line as its own moment, unchanged from before.
    """
    context_lines: list[str] = []
    events: list[str] = []
    for line in text.splitlines():
        if ANSWER_MARKER.match(line):
            break
        match = EVENT_LINE.match(line)
        if match:
            events.append(match.group(1))
        elif line.strip():
            context_lines.append(line.strip())
    if events:
        return " ".join(context_lines).strip() or None, events
    return None, [line.strip() for line in text.splitlines() if line.strip()]


def _normalize_video_id(name: str) -> str:
    return Path(name).stem


def _parse_display_timestamp(value: str | None) -> float | None:
    """Parse a legacy 'M:SS' / 'H:MM:SS' display timestamp into seconds."""
    if not value or not value.strip():
        return None
    parts = value.strip().split(":")
    if len(parts) > 3:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    padded = [0.0] * (3 - len(numbers)) + numbers
    hours, minutes, seconds = padded
    return hours * 3600 + minutes * 60 + seconds


def _run_adaptive_search(
    client: TemporalApiClient,
    lines: list[str],
    *,
    limit: int,
    apply_refinement: bool,
    common_query: str | None = None,
    retrieval_overrides: dict[str, int] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # `analysis` is the exact, already-computed rewrite result a fresh
    # Preview produced - when given, the session is built from it directly
    # (zero further LLM calls, guaranteed to match what was previewed).
    # Without a fresh preview on hand there is nothing to reuse, so this
    # still falls back to rewriting `lines` itself, same as before Preview
    # existed.
    if analysis is not None:
        session = client.create_session_from_rewrite(analysis=analysis, common_query=common_query)
    else:
        session = client.create_session_from_queries(queries=lines, common_query=common_query)
    session_id = session.session.get("id")
    if retrieval_overrides:
        # commands/retrieve re-fetches the session's own current revision
        # server-side rather than trusting a client-supplied one, so this
        # patch (which bumps the revision by 1) needs no revision tracking
        # of its own here - it just has to land before retrieve_session().
        client.patch_hyperparameters(
            session_id,
            expected_revision=int(session.session.get("revision", 0)),
            patch={"retrieval": retrieval_overrides},
        )
    all_events = [
        {
            "event_id": event.get("event_id"),
            "label": event.get("original_query") or event.get("anchor_query") or event.get("event_id"),
        }
        for event in session.session.get("events", [])
    ]
    # commands/retrieve's own top_k is the raw per-variant fetch count from
    # the upstream sparse search - hyperparameters.retrieval.top_n_per_variant
    # only *trims* what was already fetched (fuse_candidates_rrf), so top_k
    # must be at least as large as top_n_per_variant or that control (and
    # top_n_fused, downstream of it) is silently capped by whatever top_k
    # happens to be, no matter what the user configured.
    top_k = (retrieval_overrides or {}).get("top_n_per_variant", RETRIEVE_TOP_K)
    client.retrieve_session(session_id, top_k=top_k)
    page = client.get_video_priorities(
        session_id,
        limit=limit,
        apply_boundary_refinement=apply_refinement,
    )
    return {
        "pipeline": "adaptive_coarse",
        "session_id": session_id,
        "event_count": len(lines),
        "all_events": all_events,
        "display_limit": limit,
        "apply_refinement": apply_refinement,
        "capability": page.get("boundary_refinement_capability") or {},
        "items": normalize_adaptive_items(page, len(lines)),
    }


def _run_legacy_search(
    client: TemporalApiClient,
    lines: list[str],
    *,
    searcher_type: str,
    limit: int,
    apply_refinement: bool,
) -> dict[str, Any]:
    # Legacy has no session to re-rank server-side, so fetch a buffer beyond
    # `limit` up front - "reject" then just reveals the next buffered item
    # instead of making a new request.
    fetch_count = min(limit + LEGACY_REJECT_BUFFER, LEGACY_MAX_FETCH)
    response = client.legacy_search(
        queries=lines,
        top_k_tuple=fetch_count,
        searcher_type=searcher_type,
        apply_boundary_refinement=apply_refinement,
    )
    items = []
    for index, result in enumerate(response.get("results", [])):
        video_id = _normalize_video_id(str(result.get("video_name") or ""))
        moments = []
        best_frame_index = None
        best_score = None
        for position, candidate in enumerate(result.get("tuple", []), start=1):
            seconds = candidate.get("refined_timestamp_seconds")
            refined = seconds is not None
            if seconds is None:
                seconds = _parse_display_timestamp(candidate.get("timestamp"))
            moments.append({"label": f"moment {position}", "seconds": seconds, "refined": refined})
            score = candidate.get("score")
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_frame_index = candidate.get("frame_index")
        items.append(
            {
                "video_id": video_id,
                "reject_key": f"{index}:{video_id}",
                "coverage_label": f"{len(moments)} moment match",
                "priority_score": result.get("score"),
                "moments": moments,
                "frame_index": best_frame_index,
            }
        )
    return {
        "pipeline": searcher_type,
        "session_id": None,
        "event_count": len(lines),
        "display_limit": limit,
        "apply_refinement": apply_refinement,
        "capability": response.get("boundary_refinement_capability") or {},
        "items": items,
        # True if a video's backtracking search hit its node budget before
        # exploring exhaustively (searchers/*.MAX_TRAVERSAL_NODES) - results
        # are still the best found so far, but not guaranteed complete.
        "search_truncated": bool(response.get("search_truncated")),
    }


def _render_moments(
    video_id: str,
    index: int,
    moments: list[dict[str, Any]],
    resolver: MediaResolver,
    *,
    client: TemporalApiClient,
    session_id: str | None,
    all_events: list[dict[str, Any]] | None = None,
) -> None:
    playable_moments = [m for m in moments if m.get("seconds") is not None]
    if playable_moments:
        for moment in playable_moments:
            label = f"{moment['label']} @ {format_timestamp(moment['seconds'])}"
            if not moment.get("refined", True):
                label += " (approx.)"
            if moment.get("source") == "user_fixed":
                label += " [fixed]"
            can_fix = session_id is not None and moment.get("event_id") is not None
            if can_fix:
                row = st.columns([4, 1])
                row[0].caption(label)
                # session_id-scoped - video_id/event_id/index can all repeat
                # across unrelated searches (a new session from clicking
                # "Search" again), and without it a stale "show the frame
                # picker" toggle (or its jump-to-seconds position) from a
                # previous session would silently carry over into a new one.
                flag_key = f"show_frames_{session_id}_{index}_{video_id}_{moment['event_id']}"
                if row[1].button("Fix", key=f"fixbtn_{flag_key}"):
                    st.session_state[flag_key] = not st.session_state.get(flag_key, False)
                if st.session_state.get(flag_key):
                    render_frame_fixer(
                        st.session_state,
                        client,
                        session_id=session_id,
                        event_id=moment["event_id"],
                        video_id=video_id,
                        anchor_seconds=moment["seconds"],
                        flag_key=flag_key,
                    )
            else:
                st.caption(label)
    else:
        st.caption("No moment timestamps available — turn on 'Refine timestamps' to see exact moments.")

    # Events that retrieval never found anything for in this video at all
    # (not even an approximate anchor) get no chip above - but the video may
    # still genuinely contain that moment, so offer the same fixer, seeded
    # near this video's other found moments instead of an arbitrary default.
    if session_id and all_events:
        covered_event_ids = {m.get("event_id") for m in moments if m.get("event_id")}
        missing_events = [e for e in all_events if e.get("event_id") not in covered_event_ids]
        if missing_events:
            default_anchor = playable_moments[-1]["seconds"] if playable_moments else 0.0
            for missing_event in missing_events:
                event_id = missing_event.get("event_id")
                row = st.columns([4, 1])
                row[0].caption(f"{missing_event.get('label')} - not found in this video")
                flag_key = f"show_frames_{session_id}_{index}_{video_id}_{event_id}"
                if row[1].button("Fix", key=f"fixbtn_{flag_key}"):
                    st.session_state[flag_key] = not st.session_state.get(flag_key, False)
                if st.session_state.get(flag_key):
                    render_frame_fixer(
                        st.session_state,
                        client,
                        session_id=session_id,
                        event_id=event_id,
                        video_id=video_id,
                        anchor_seconds=default_anchor,
                        flag_key=flag_key,
                    )

    if session_id and all_events:
        with st.expander("Browse keyframes by event"):
            render_event_keyframe_browser(
                st.session_state, client, resolver,
                session_id=session_id, video_id=video_id, all_events=all_events, index=index,
            )

    video_path = resolver.resolve_video(video_id) if resolver.available() else None
    if video_path is None:
        st.caption("Video not available locally.")
        return
    with st.expander("Play"):
        if playable_moments:
            options = {
                f"{m['label']} ({format_timestamp(m['seconds'])})": m["seconds"] for m in playable_moments
            }
            chosen_label = st.selectbox(
                "Jump to", options=list(options.keys()), key=f"jump_{session_id}_{index}_{video_id}"
            )
            start_time = int(round(options[chosen_label]))
        else:
            start_time = 0
        st.video(str(video_path), start_time=start_time)


def _render_results(
    payload: dict[str, Any], resolver: MediaResolver, client: TemporalApiClient
) -> None:
    all_items = payload.get("items", [])
    rejected: set[str] = st.session_state.get(K.SEARCH_REJECTED_IDS, set())
    display_limit = payload.get("display_limit", len(all_items))
    items = [item for item in all_items if item.get("reject_key") not in rejected][:display_limit]
    capability = payload.get("capability", {})
    if capability.get("requested") and not capability.get("available"):
        st.info(
            "Timestamp refinement is unavailable right now "
            f"({capability.get('reason') or 'runtime not configured'}). "
            "Showing unrefined results."
        )
    if payload.get("search_truncated"):
        st.warning(
            "The search hit its traversal budget before exploring every "
            "combination for at least one video - these are the best "
            "results found so far, not guaranteed to be exhaustive."
        )

    if not items:
        if rejected:
            st.warning("No more results - everything else has been marked not relevant.")
        else:
            st.warning("No matching videos found.")
        return

    st.subheader(f"Results - {len(items)} video{'s' if len(items) != 1 else ''}")
    for index, item in enumerate(items):
        video_id = item.get("video_id") or "?"
        with st.container(border=True):
            cols = st.columns([1, 3, 1])
            with cols[0]:
                thumbnail = None
                if resolver.available():
                    first_seconds = next(
                        (m["seconds"] for m in item["moments"] if m.get("seconds") is not None), None
                    )
                    if item.get("frame_index") is not None:
                        thumbnail = resolver.resolve_keyframe(video_id, item["frame_index"])
                    elif first_seconds is not None:
                        thumbnail = resolver.resolve_keyframe_near(video_id, first_seconds)
                if thumbnail is not None:
                    st.image(str(thumbnail), use_container_width=True)
                else:
                    st.caption("no thumbnail")
            with cols[1]:
                st.markdown(f"**{video_id}**")
                st.caption(item.get("coverage_label", ""))
                _render_moments(
                    video_id,
                    index,
                    item.get("moments", []),
                    resolver,
                    client=client,
                    session_id=payload.get("session_id"),
                    all_events=payload.get("all_events"),
                )
            with cols[2]:
                # Prioritize (reorder to top-1) is deliberately separate
                # from "Not this one" (filter, changes set membership) -
                # neither one touches the other's mechanism.
                if payload.get("pipeline") == "adaptive_coarse" and index != 0:
                    if st.button("Prioritize to top", key=f"prioritize_{index}_{video_id}"):
                        with st.spinner("Updating…"):
                            prioritize_item(st.session_state, client, video_id)
                        st.rerun()
                reject_key = item.get("reject_key")
                if st.button("Not this one", key=f"reject_{index}_{reject_key}"):
                    with st.spinner("Updating…"):
                        reject_item(st.session_state, client, reject_key)
                    st.rerun()


bootstrap()
configure_sidebar()

client = get_api_client()
media_resolver = get_media_resolver()

st.title("Search")
st.caption(
    "Describe what happens, one moment per line, in order. You can also paste a full "
    "query block (context line, then 'E1: ...', 'E2: ...' moments) and it's parsed "
    "automatically - no need to split it apart yourself."
)

query_text = st.text_area(
    "What are you looking for?",
    value=st.session_state.get(K.SEARCH_QUERY_TEXT, ""),
    placeholder=(
        "person cuts an onion\nperson fries the onion in a pan\n\n"
        "-- or paste a full block --\n"
        "A video about frying onion rings:\n"
        "E1: cuts the onion into rings\n"
        "E2: dips the rings in batter\n"
        "E3: fries the rings in oil"
    ),
    height=120,
)

preview_common_query, preview_moments = _parse_query_block(query_text)
if preview_moments:
    detected = f"Detected {len(preview_moments)} moment{'s' if len(preview_moments) != 1 else ''}"
    if preview_common_query:
        detected += f" - shared context: \"{preview_common_query}\""
    st.caption(detected)

pipeline_label = st.radio("Search mode", options=list(PIPELINES.keys()), horizontal=True)
pipeline = PIPELINES[pipeline_label]

retrieval_controls = render_retrieval_controls(pipeline=pipeline)
result_limit = retrieval_controls.result_limit
apply_refinement = retrieval_controls.apply_refinement
retrieval_overrides = retrieval_controls.retrieval_overrides

if pipeline == "adaptive_coarse":
    common_query, lines = _parse_query_block(query_text)
    render_preview_controls(
        st.session_state, client,
        common_query=common_query, lines=lines, query_text=query_text,
    )

search_clicked = st.button("Search", type="primary")

if search_clicked:
    common_query, lines = _parse_query_block(query_text)
    st.session_state[K.SEARCH_QUERY_TEXT] = query_text
    st.session_state[K.SEARCH_PIPELINE] = pipeline
    st.session_state[K.SEARCH_APPLY_REFINEMENT] = apply_refinement
    st.session_state[K.SEARCH_ERROR] = None
    st.session_state[K.SEARCH_REJECTED_IDS] = set()
    if not lines:
        st.session_state[K.SEARCH_ERROR] = ("input", "Describe at least one moment.")
        st.session_state[K.SEARCH_RESULTS] = None
    else:
        try:
            with st.spinner("Searching…"):
                if pipeline == "adaptive_coarse":
                    # Reuse a fresh preview (still matching the current query
                    # text) verbatim instead of rewriting again - the whole
                    # point of previewing is that what you reviewed is what
                    # Search actually uses. A missing or stale preview (never
                    # previewed, or the text changed since) has nothing valid
                    # to reuse, so this falls back to _run_adaptive_search's
                    # own fresh rewrite, same as if Preview had never existed.
                    results = _run_adaptive_search(
                        client,
                        lines,
                        limit=result_limit,
                        apply_refinement=apply_refinement,
                        common_query=common_query,
                        retrieval_overrides=retrieval_overrides,
                        analysis=get_authoritative_analysis(st.session_state, query_text),
                    )
                else:
                    searcher_type = "TemporalSearcher" if pipeline == "legacy_temporal" else "AmbiguousSearcher"
                    results = _run_legacy_search(
                        client,
                        lines,
                        searcher_type=searcher_type,
                        limit=result_limit,
                        apply_refinement=apply_refinement,
                    )
            st.session_state[K.SEARCH_RESULTS] = results
        except ConnectionFailure as exc:
            st.session_state[K.SEARCH_RESULTS] = None
            st.session_state[K.SEARCH_ERROR] = ("connection", exc)
        except ApiError as exc:
            st.session_state[K.SEARCH_RESULTS] = None
            st.session_state[K.SEARCH_ERROR] = ("api", exc)

error = st.session_state.get(K.SEARCH_ERROR)
if error:
    kind, payload = error
    if kind == "connection":
        show_connection_error(payload, key="search_retry")
    elif kind == "api":
        st.error(f"{type(payload).__name__}: {payload.message}")
    else:
        st.error(str(payload))

results = st.session_state.get(K.SEARCH_RESULTS)
if results is not None:
    _render_results(results, media_resolver, client)
