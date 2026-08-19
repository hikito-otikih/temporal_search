"""Session-mutation actions for the Search page: reject/prioritize a video,
confirm a manually-picked frame. All three go through `patch_constraints`
or `commands/fix-frame` and then re-fetch + re-normalize video-priorities so
the caller's stored results reflect the server's post-mutation state.
"""

from __future__ import annotations

from typing import Any, Callable

from models.ui_models import normalize_adaptive_items
from services.api_client import ApiError, TemporalApiClient
from state import keys as K


def patch_constraints(
    client: TemporalApiClient,
    session_id: str,
    updater: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Fetch the session's *current* constraints, let `updater(constraints)`
    return the field updates to merge in, and PUT the whole merged object
    back. PUT .../constraints is a real REST replace, not a merge
    (AdaptiveSearchService.replace_constraints does `bundle.session.
    constraints = constraints.model_copy(deep=True)`) - a caller that PUTs
    only the one field it cares about silently wipes every other constraint
    already set (e.g. a prior commands/fix-frame's event_constraints),
    which is exactly what this function exists to avoid by always starting
    from the server's own current state. `updater` takes the current
    constraints dict rather than each caller fetching its own copy, so a
    caller that needs to read-then-modify a list field (see
    `prioritize_item`) does it against the one fetch this function already
    made, not a second, separately-racing one.
    """
    session = client.get_session(session_id)
    revision = int(session.session.get("revision", 0))
    constraints = dict(session.session.get("constraints") or {})
    constraints.update(updater(constraints))
    client.replace_constraints(session_id, expected_revision=revision, constraints=constraints)


def reject_item(store: Any, client: TemporalApiClient, reject_key: str) -> None:
    """Mark one result wrong. Adaptive: reject its video server-side and
    re-rank, so the next-best video (already computed, just outside the
    previous cutoff) takes its place. Legacy: no session to mutate, so this
    just hides it and reveals the next item already sitting in the
    over-fetched buffer.

    A video that is currently prioritized, allowlisted, or fix_frame'd
    cannot also be rejected (server-enforced - see EventConstraint/
    SearchConstraints validators), so the updater below clears all three
    before adding the rejection, instead of sending a partial PATCH that
    422s. The local hide (SEARCH_REJECTED_IDS) is only recorded after the
    server call succeeds, so a failed request doesn't hide a result the
    server never actually rejected.
    """
    payload = store.get(K.SEARCH_RESULTS)
    if not payload or payload.get("pipeline") != "adaptive_coarse" or not payload.get("session_id"):
        rejected: set[str] = store.setdefault(K.SEARCH_REJECTED_IDS, set())
        rejected.add(reject_key)
        return

    video_id = reject_key  # adaptive reject_key is always the bare video_id
    try:
        def updater(constraints: dict[str, Any]) -> dict[str, Any]:
            prioritized = [
                v for v in (constraints.get("prioritized_video_ids") or []) if v != video_id
            ]
            allowed = [v for v in (constraints.get("allowed_video_ids") or []) if v != video_id]
            event_constraints = dict(constraints.get("event_constraints") or {})
            for event_id, event_constraint in event_constraints.items():
                if (event_constraint or {}).get("fixed_video_id") == video_id:
                    event_constraints[event_id] = {
                        **event_constraint,
                        "fixed_video_id": None,
                        "fixed_region_id": None,
                        "fixed_frame_id": None,
                        "fixed_timestamp_seconds": None,
                    }
            already_rejected = set(constraints.get("rejected_video_ids") or [])
            return {
                "rejected_video_ids": sorted(already_rejected | {video_id}),
                "prioritized_video_ids": prioritized,
                "allowed_video_ids": allowed,
                "event_constraints": event_constraints,
            }

        patch_constraints(client, payload["session_id"], updater)
        page = client.get_video_priorities(
            payload["session_id"],
            limit=payload["display_limit"],
            apply_boundary_refinement=payload["apply_refinement"],
        )
        payload["items"] = normalize_adaptive_items(page, payload["event_count"])
        payload["capability"] = page.get("boundary_refinement_capability") or {}
        store[K.SEARCH_RESULTS] = payload
        rejected: set[str] = store.setdefault(K.SEARCH_REJECTED_IDS, set())
        rejected.add(reject_key)
    except ApiError as exc:
        store[K.SEARCH_ERROR] = ("api", exc)


def prioritize_item(store: Any, client: TemporalApiClient, video_id: str) -> None:
    """Pin one video to the top of the ranked list. In contrast with 'Not
    this one' (which changes set membership - excludes a video entirely),
    this only reorders the response: it does not touch priority_score, any
    video's own tuple/region choice, or fix_frame's per-event pins (which
    are deliberately scoped to their own video and never affect the
    ordering of videos against each other - see prioritized_video_ids'
    schema docstring).
    """
    payload = store.get(K.SEARCH_RESULTS)
    if not payload or payload.get("pipeline") != "adaptive_coarse" or not payload.get("session_id"):
        return
    try:
        def updater(constraints: dict[str, Any]) -> dict[str, Any]:
            current = list(constraints.get("prioritized_video_ids") or [])
            prioritized = [video_id] + [v for v in current if v != video_id]
            rejected = [v for v in (constraints.get("rejected_video_ids") or []) if v != video_id]
            return {"prioritized_video_ids": prioritized, "rejected_video_ids": rejected}

        patch_constraints(client, payload["session_id"], updater)
        page = client.get_video_priorities(
            payload["session_id"],
            limit=payload["display_limit"],
            apply_boundary_refinement=payload["apply_refinement"],
        )
        payload["items"] = normalize_adaptive_items(page, payload["event_count"])
        payload["capability"] = page.get("boundary_refinement_capability") or {}
        store[K.SEARCH_RESULTS] = payload
    except ApiError as exc:
        store[K.SEARCH_ERROR] = ("api", exc)


def confirm_fixed_frame(
    store: Any,
    client: TemporalApiClient,
    *,
    session_id: str,
    event_id: str,
    video_id: str,
    frame: dict[str, Any],
    region_id: str | None = None,
) -> None:
    """Persist a manually-picked frame, then re-fetch so the fix is
    reflected (video-priorities treats a fixed event as authoritative and
    skips re-refining it).

    `region_id` preserves *provenance* when the pick came from a real,
    scored retrieval candidate (the keyframe browser) rather than an
    arbitrary raw frame (the frame-preview fixer, which has no backing
    region at all) - without it, commands/fix-frame still works (it
    defaults to a synthetic placeholder id), but GET /proposals and
    /regions lose the link back to which actual candidate region the user
    confirmed.
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
            region_id=region_id,
        )
        payload = store.get(K.SEARCH_RESULTS)
        if payload and payload.get("session_id") == session_id:
            page = client.get_video_priorities(
                session_id,
                limit=payload["display_limit"],
                apply_boundary_refinement=payload["apply_refinement"],
            )
            payload["items"] = normalize_adaptive_items(page, payload["event_count"])
            payload["capability"] = page.get("boundary_refinement_capability") or {}
            store[K.SEARCH_RESULTS] = payload
    except ApiError as exc:
        store[K.SEARCH_ERROR] = ("api", exc)
