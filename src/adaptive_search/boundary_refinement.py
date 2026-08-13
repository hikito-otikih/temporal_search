"""The boundary-refinement pipeline stage: a cheap, per-event local sweep.

Given any pipeline's already-chosen anchor timestamp for one event in one
video, look at real native-fps neighbors (+/-`radius_frames`, spaced
`stride` native frames apart) and use the existing profile-aware
change-point detector (`generate_profiled_proposals`) to snap onto the true
onset/offset - instead of the expensive frontier-wide refine `adaptive_full`
used to run (removed; see `refinement_runtime.py`'s module docstring and
`docs/ADAPTIVE_PIPELINE_MIGRATION.md`).

Ported from `irrelevant_things/benchmarks/youcook2/boundary_refinement.py`,
validated there against real YouCook2 queries this session, with one
correctness fix: `pre_state`/`post_state` are optional here. The benchmark
version required them (only ever called for "onset"/"offset" moment-oriented
queries, which always had synthesized state text). Production callers -
`legacy_search` in particular - often have no state-semantics text for a
freeform query at all. Passing the same fallback text for both `pre_state`
and `post_state` would make the transition-profile contrast math degenerate
(`pairwise_softmax(x, x) == (0.5, 0.5)` for every frame, verified directly
against `algorithms.py`), so when `pre_state`/`post_state` are `None` this
module dispatches, via `generate_profiled_proposals()`, to the "state"
profile instead (anchor-persistence re-centering on `normalized_anchor_score`
alone) - a well-defined, non-degenerate behavior, at the cost of not doing
true onset/offset detection for queries with no state semantics. Pass
`boundary_type="onset"/"offset"` with real `pre_state`/`post_state` text when
you have it (true transition detection); pass `"unknown"` with both `None`
otherwise (state re-centering).

Same `radius_frames=10, stride=1` defaults as the benchmark version - a
hyperparameter sweep this session (n=60 real queries) found no combination
that beat them; don't invent new defaults here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embedding import TextImageEmbedder
from .providers import FrameProvider
from .proposal_profiles import generate_profiled_proposals
from .refinement_runtime import score_frames, state_queries
from .schemas import (
    BoundaryHyperparameters,
    BoundaryType,
    EventDefinition,
    EventProposal,
    FrameScoreSample,
)


@dataclass(frozen=True)
class RefinementOutcome:
    event_id: str
    video_id: str
    anchor_seconds: float
    refined_seconds: float  # == anchor_seconds when used_fallback
    used_fallback: bool
    sampled_frame_count: int
    top_proposal: EventProposal | None


def _native_window_options(fps: float, stride: int) -> tuple[float, ...]:
    # A 1-sample window can never satisfy min_samples_per_side=2 (there's at
    # most one neighbor on each side at that spacing), so it's dead weight -
    # start from 2 sample steps. Scaled by `stride`: consecutive *sampled*
    # frames are `stride` native frames (stride/fps seconds) apart, not
    # necessarily 1 native frame apart.
    sample_seconds = stride / fps
    return tuple(sorted({sample_seconds * samples for samples in (2, 4, 8)}))


def refine_event_boundary(
    *,
    provider: FrameProvider,
    embedder: TextImageEmbedder,
    video_id: str,
    event_id: str,
    anchor_query: str,
    pre_state: str | None,
    post_state: str | None,
    boundary_type: BoundaryType,
    anchor_seconds: float,
    fps: float,
    duration_ms: int | None = None,
    radius_frames: int = 10,
    stride: int = 1,
    image_batch_size: int | None = 32,
    session_id: str | None = None,
    region_id: str | None = None,
) -> RefinementOutcome:
    # Reuse EventDefinition's own validator instead of re-implementing the
    # onset/offset-requires-pre/post-state check.
    event_definition = EventDefinition(
        event_id=event_id,
        original_query=anchor_query,
        anchor_query=anchor_query,
        pre_state=pre_state,
        post_state=post_state,
        boundary_type=boundary_type,
    )
    if fps <= 0:
        raise ValueError("fps must be positive")
    if stride < 1:
        raise ValueError("stride must be a positive integer (in native frames)")

    session_id = session_id or f"boundary_refine_{video_id}"
    region_id = region_id or f"boundary_refine_region_{video_id}_{event_id}"
    anchor_frame = round(anchor_seconds * fps)

    def frame_to_ms(index: int) -> int:
        ms = max(0, round(index / fps * 1000))
        return ms if duration_ms is None else min(ms, duration_ms)

    target_pts_ms = sorted(
        {
            frame_to_ms(anchor_frame + step * stride)
            for step in range(-radius_frames, radius_frames + 1)
        }
    )
    frames = provider.get_frames(video_id, target_pts_ms)
    if not frames:
        return RefinementOutcome(event_id, video_id, anchor_seconds, anchor_seconds, True, 0, None)

    # Non-None text is required to embed at all - state_queries() falls back
    # to anchor_query when pre_state/post_state are None. This only affects
    # which text gets embedded into the (ignored, for the "state" profile)
    # raw_pre_score/raw_post_score columns; dispatch to "state" vs
    # "transition" is decided by event_definition.pre_state/post_state
    # (still None here when the caller passed None), not by this fallback.
    pre_text, post_text, _ = state_queries(event_definition)
    curves = score_frames(
        embedder,
        anchor_query=anchor_query,
        pre_state_query=pre_text,
        post_state_query=post_text,
        images=[frame.image for frame in frames],
        image_batch_size=image_batch_size,
    )
    samples = [
        FrameScoreSample(
            session_id=session_id,
            event_id=event_id,
            video_id=video_id,
            region_id=region_id,
            frame_id=frame.pts_ms,
            timestamp_seconds=frame.pts_ms / 1000.0,
            raw_anchor_score=curves.anchor[index],
            raw_pre_score=curves.pre[index],
            raw_post_score=curves.post[index],
            raw_motion_score=0.0,  # hardcoded repo-wide; no motion signal implemented
        )
        for index, frame in enumerate(frames)
    ]

    parameters = BoundaryHyperparameters(
        window_options_seconds=_native_window_options(fps, stride)
    )
    proposals = generate_profiled_proposals(
        [event_definition], samples, parameters, source="dense"
    )
    if not proposals:
        return RefinementOutcome(
            event_id, video_id, anchor_seconds, anchor_seconds, True, len(frames), None
        )

    top = max(proposals, key=lambda proposal: proposal.final_event_score)
    return RefinementOutcome(
        event_id, video_id, anchor_seconds, top.timestamp_seconds, False, len(frames), top
    )
