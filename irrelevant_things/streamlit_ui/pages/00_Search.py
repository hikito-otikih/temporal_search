"""00 Search — the primary, consumer-facing search experience.

Type what you're looking for (one moment per line, in order) and get back
ranked videos with the matched moment's timestamp, a thumbnail, and
on-demand playback. Everything else in this app ("Developer Tools") is for
inspecting pipeline internals; this page is the product.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import streamlit as st

from _bootstrap import bootstrap, configure_sidebar, get_api_client, get_media_resolver
from components.api_status import show_connection_error
from models.ui_models import format_timestamp
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
# (irrelevant_things/benchmarks/youcook2/core.py: EVENT_RE/ANSWER_RE) - lets
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


def _normalize_adaptive_items(page: dict[str, Any], event_count: int) -> list[dict[str, Any]]:
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


def _run_adaptive_search(
    client: TemporalApiClient,
    lines: list[str],
    *,
    limit: int,
    apply_refinement: bool,
    apply_tuple_ranking: bool = False,
    common_query: str | None = None,
) -> dict[str, Any]:
    session = client.create_session_from_queries(queries=lines, common_query=common_query)
    session_id = session.session.get("id")
    all_events = [
        {
            "event_id": event.get("event_id"),
            "label": event.get("original_query") or event.get("anchor_query") or event.get("event_id"),
        }
        for event in session.session.get("events", [])
    ]
    client.retrieve_session(session_id, top_k=RETRIEVE_TOP_K)
    page = client.get_video_priorities(
        session_id,
        limit=limit,
        apply_boundary_refinement=apply_refinement,
        apply_tuple_ranking=apply_tuple_ranking,
    )
    return {
        "pipeline": "adaptive_coarse",
        "session_id": session_id,
        "event_count": len(lines),
        "all_events": all_events,
        "display_limit": limit,
        "apply_refinement": apply_refinement,
        "apply_tuple_ranking": apply_tuple_ranking,
        "capability": page.get("boundary_refinement_capability") or {},
        "items": _normalize_adaptive_items(page, len(lines)),
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
    }


def _reject_item(client: TemporalApiClient, reject_key: str) -> None:
    """Mark one result wrong. Adaptive: reject its video server-side and
    re-rank, so the next-best video (already computed, just outside the
    previous cutoff) takes its place. Legacy: no session to mutate, so this
    just hides it and reveals the next item already sitting in the
    over-fetched buffer.
    """
    rejected: set[str] = st.session_state.setdefault(K.SEARCH_REJECTED_IDS, set())
    rejected.add(reject_key)

    payload = st.session_state.get(K.SEARCH_RESULTS)
    if not payload or payload.get("pipeline") != "adaptive_coarse" or not payload.get("session_id"):
        return
    try:
        session = client.get_session(payload["session_id"])
        revision = int(session.session.get("revision", 0))
        client.replace_constraints(
            payload["session_id"],
            expected_revision=revision,
            constraints={"rejected_video_ids": sorted(rejected)},
        )
        page = client.get_video_priorities(
            payload["session_id"],
            limit=payload["display_limit"],
            apply_boundary_refinement=payload["apply_refinement"],
            apply_tuple_ranking=payload.get("apply_tuple_ranking", False),
        )
        payload["items"] = _normalize_adaptive_items(page, payload["event_count"])
        payload["capability"] = page.get("boundary_refinement_capability") or {}
        st.session_state[K.SEARCH_RESULTS] = payload
    except ApiError as exc:
        st.session_state[K.SEARCH_ERROR] = ("api", exc)


def _confirm_fixed_frame(
    client: TemporalApiClient,
    *,
    session_id: str,
    event_id: str,
    video_id: str,
    frame: dict[str, Any],
) -> None:
    """Persist a manually-picked frame, then re-fetch so the fix is
    reflected (video-priorities treats a fixed event as authoritative and
    skips re-refining it).
    """
    try:
        session = client.get_session(session_id)
        revision = int(session.session.get("revision", 0))
        client.fix_frame(
            session_id,
            expected_revision=revision,
            event_id=event_id,
            video_id=video_id,
            frame_id=int(frame["pts_ms"]),
            timestamp_seconds=float(frame["timestamp_seconds"]),
        )
        payload = st.session_state.get(K.SEARCH_RESULTS)
        if payload and payload.get("session_id") == session_id:
            page = client.get_video_priorities(
                session_id,
                limit=payload["display_limit"],
                apply_boundary_refinement=payload["apply_refinement"],
                apply_tuple_ranking=payload.get("apply_tuple_ranking", False),
            )
            payload["items"] = _normalize_adaptive_items(page, payload["event_count"])
            payload["capability"] = page.get("boundary_refinement_capability") or {}
            st.session_state[K.SEARCH_RESULTS] = payload
    except ApiError as exc:
        st.session_state[K.SEARCH_ERROR] = ("api", exc)


def _render_frame_fixer(
    client: TemporalApiClient,
    *,
    session_id: str,
    event_id: str,
    video_id: str,
    anchor_seconds: float,
    flag_key: str,
) -> None:
    # The 31-frame window is centered on `center_key`, which starts at the
    # detected/default anchor but can be moved anywhere in the video with the
    # step buttons or the jump-to box below - not limited to browsing only
    # around whatever the pipeline originally found.
    center_key = f"center_{flag_key}"
    if center_key not in st.session_state:
        st.session_state[center_key] = float(anchor_seconds)

    nav = st.columns([1, 1, 1, 1, 2])
    if nav[0].button("- 10s", key=f"back10_{flag_key}"):
        st.session_state[center_key] = max(0.0, st.session_state[center_key] - 10.0)
    if nav[1].button("- 1s", key=f"back1_{flag_key}"):
        st.session_state[center_key] = max(0.0, st.session_state[center_key] - 1.0)
    if nav[2].button("+ 1s", key=f"fwd1_{flag_key}"):
        st.session_state[center_key] += 1.0
    if nav[3].button("+ 10s", key=f"fwd10_{flag_key}"):
        st.session_state[center_key] += 10.0
    with nav[4]:
        st.number_input(
            "Jump to (seconds)",
            min_value=0.0,
            step=1.0,
            key=center_key,
            label_visibility="collapsed",
        )
    center_seconds = float(st.session_state[center_key])

    try:
        preview = client.get_frame_preview(video_id, anchor_seconds=center_seconds, radius_frames=15)
    except ApiError as exc:
        st.error(f"Could not load frames: {exc.message}")
        return
    frames = preview.get("frames", [])
    if not frames:
        st.caption("No frames available for this video.")
        return
    st.caption(f"{len(frames)} frames around {format_timestamp(center_seconds)} - click one to pick it.")
    columns_per_row = 6
    for row_start in range(0, len(frames), columns_per_row):
        row = frames[row_start : row_start + columns_per_row]
        cols = st.columns(len(row))
        for col, frame in zip(cols, row):
            with col:
                image_bytes = base64.b64decode(frame["image_base64"])
                st.image(image_bytes, use_container_width=True)
                choose_label = format_timestamp(frame["timestamp_seconds"])
                if frame["offset"] == 0:
                    choose_label += " *"
                if st.button(choose_label, key=f"pick_{flag_key}_{frame['pts_ms']}"):
                    with st.spinner("Saving…"):
                        _confirm_fixed_frame(
                            client,
                            session_id=session_id,
                            event_id=event_id,
                            video_id=video_id,
                            frame=frame,
                        )
                    st.session_state[flag_key] = False
                    st.session_state.pop(center_key, None)
                    st.rerun()


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
                flag_key = f"show_frames_{index}_{video_id}_{moment['event_id']}"
                if row[1].button("Fix", key=f"fixbtn_{flag_key}"):
                    st.session_state[flag_key] = not st.session_state.get(flag_key, False)
                if st.session_state.get(flag_key):
                    _render_frame_fixer(
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
                flag_key = f"show_frames_{index}_{video_id}_{event_id}"
                if row[1].button("Fix", key=f"fixbtn_{flag_key}"):
                    st.session_state[flag_key] = not st.session_state.get(flag_key, False)
                if st.session_state.get(flag_key):
                    _render_frame_fixer(
                        client,
                        session_id=session_id,
                        event_id=event_id,
                        video_id=video_id,
                        anchor_seconds=default_anchor,
                        flag_key=flag_key,
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
                "Jump to", options=list(options.keys()), key=f"jump_{index}_{video_id}"
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
                reject_key = item.get("reject_key")
                if st.button("Not this one", key=f"reject_{index}_{reject_key}"):
                    with st.spinner("Updating…"):
                        _reject_item(client, reject_key)
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

with st.expander("Advanced options"):
    result_limit = st.slider("Number of results", min_value=5, max_value=50, value=20, step=5)
    apply_refinement = st.checkbox(
        "Refine timestamps (slower, experimental)",
        value=False,
        help=(
            "Runs a real GPU decode+embed scan around every matched moment of every "
            "result to snap onto the exact frame, instead of the coarse retrieval "
            "anchor. Cost scales with results x moments, no other cap - e.g. 50 "
            "results x 3 moments measured at ~65s. In practice the timestamp "
            "adjustment itself is usually small (a fraction of a second)."
        ),
    )
    if apply_refinement and result_limit > 20:
        st.caption(
            f"{result_limit} results with refinement on can take a while (roughly "
            f"1-1.5s per result per moment) - lower 'Number of results' for a faster search."
        )
    if pipeline == "adaptive_coarse":
        apply_tuple_ranking = st.checkbox(
            "Tuple-aware ranking (experimental)",
            value=False,
            help=(
                "Ranks videos by jointly considering all events' candidate regions together "
                "(rewarding a video whose events land in the right chronological order) instead "
                "of picking each event's best match independently. Benchmarked on real YouCook2 "
                "queries at +15% final_query_score over the default ranking (n=30 sample - see "
                "irrelevant_things/benchmarks/youcook2/region_tuple_ranking_results.md). "
                "Pure CPU, no extra latency; when combined with 'Refine timestamps', refinement "
                "also seeds from this joint choice instead of each event's independent match."
            ),
        )
    else:
        apply_tuple_ranking = False
        st.caption("Tuple-aware ranking is only available for Adaptive search.")

search_clicked = st.button("Search", type="primary")

if search_clicked:
    common_query, lines = _parse_query_block(query_text)
    st.session_state[K.SEARCH_QUERY_TEXT] = query_text
    st.session_state[K.SEARCH_PIPELINE] = pipeline
    st.session_state[K.SEARCH_APPLY_REFINEMENT] = apply_refinement
    st.session_state[K.SEARCH_APPLY_TUPLE_RANKING] = apply_tuple_ranking
    st.session_state[K.SEARCH_ERROR] = None
    st.session_state[K.SEARCH_REJECTED_IDS] = set()
    if not lines:
        st.session_state[K.SEARCH_ERROR] = ("input", "Describe at least one moment.")
        st.session_state[K.SEARCH_RESULTS] = None
    else:
        try:
            with st.spinner("Searching…"):
                if pipeline == "adaptive_coarse":
                    results = _run_adaptive_search(
                        client,
                        lines,
                        limit=result_limit,
                        apply_refinement=apply_refinement,
                        apply_tuple_ranking=apply_tuple_ranking,
                        common_query=common_query,
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
        show_connection_error(payload)
    elif kind == "api":
        st.error(f"{type(payload).__name__}: {payload.message}")
    else:
        st.error(str(payload))

results = st.session_state.get(K.SEARCH_RESULTS)
if results is not None:
    _render_results(results, media_resolver, client)
