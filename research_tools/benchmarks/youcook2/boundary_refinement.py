"""The "final refinement boundary stage": a cheap, pipeline-agnostic local sweep.

Given any pipeline's already-chosen anchor timestamp for one event in one
video, look at real native-fps neighbors (+/-`radius_frames`) and use the
existing contrastive pre/post-state change-point detector to snap onto the
true onset/offset - instead of the expensive mid-pipeline frontier-wide
refine that only the now-removed `adaptive_full` pipeline used to run (see
docs/ADAPTIVE_PIPELINE_MIGRATION.md).

Reuses the existing dense-scoring/change-point machinery in
`adaptive_search` as-is (`score_event_frames`, `calibrate_frame_scores`,
`generate_boundary_proposals`) rather than inventing a new algorithm. The
only new code here is: (a) building the narrow, native-fps sweep window
instead of the frontier-wide dense sampling grid `refine_session()` uses,
and (b) fabricating schema-safe placeholder session/region ids, since this
bypasses the real session/region system entirely.

Deliberately single-anchor, not multi-seed: an earlier version accepted
several seed candidates from coarse_anchor.py and swept the union of their
neighborhoods, picking whichever seed's local proposal scored best. That
was wrong - `generate_boundary_proposals()`'s `final_event_score` is
`robust_sigmoid`-calibrated *within* each seed's own small local group, so
scores from different seeds' groups aren't on a comparable scale; a
mediocre frame that's merely locally-peaked in the wrong neighborhood can
outscore the genuinely correct frame in the right one. Verified directly:
it picked a proposal from a wrong seed's neighborhood even when the correct
seed (already ranked #1 by coarse_anchor.py) was in the pool and would have
produced a correct in-ground-truth answer on its own. Adjudicating *between*
candidate neighborhoods isn't what a change-point detector does - that's
coarse_anchor.py's job (region-based seed ranking); this function's job is
only to fine-tune within whichever single neighborhood it's given.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_search.algorithms import generate_boundary_proposals
from adaptive_search.config import runtime_from_environment
from adaptive_search.embedding import TextImageEmbedder, score_event_frames
from adaptive_search.providers import FrameProvider
from adaptive_search.schemas import (
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


def build_runtime() -> tuple[FrameProvider, TextImageEmbedder]:
    """Build the same (frame_provider, embedder) pair the real backend uses."""

    config = runtime_from_environment()
    if not (config.provider_configured and config.model_configured):
        raise RuntimeError(f"live refinement runtime unavailable: {config.reason}")
    return config.frame_provider, config.embedder


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
    pre_state: str,
    post_state: str,
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
    EventDefinition(
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

    session_id = session_id or f"boundary_exp_{video_id}"
    region_id = region_id or f"boundary_exp_region_{video_id}_{event_id}"
    anchor_frame = round(anchor_seconds * fps)

    def frame_to_ms(index: int) -> int:
        ms = max(0, round(index / fps * 1000))
        return ms if duration_ms is None else min(ms, duration_ms)

    # `radius_frames` samples on each side, `stride` native frames apart -
    # total coverage is +/-(radius_frames * stride) native frames, but only
    # 2*radius_frames+1 frames are actually decoded/scored (stride=1 is the
    # original dense behavior: every native frame within +/-radius_frames).
    target_pts_ms = sorted(
        {
            frame_to_ms(anchor_frame + step * stride)
            for step in range(-radius_frames, radius_frames + 1)
        }
    )
    frames = provider.get_frames(video_id, target_pts_ms)
    if not frames:
        return RefinementOutcome(event_id, video_id, anchor_seconds, anchor_seconds, True, 0, None)

    curves = score_event_frames(
        embedder,
        anchor_query=anchor_query,
        pre_state_query=pre_state,
        post_state_query=post_state,
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
            raw_motion_score=0.0,  # hardcoded repo-wide; out of scope here
        )
        for index, frame in enumerate(frames)
    ]

    parameters = BoundaryHyperparameters(window_options_seconds=_native_window_options(fps, stride))
    # generate_boundary_proposals() calibrates internally - don't pre-calibrate here.
    proposals = generate_boundary_proposals(samples, parameters, source="dense", apply_nms=True)
    if not proposals:
        return RefinementOutcome(
            event_id, video_id, anchor_seconds, anchor_seconds, True, len(frames), None
        )

    top = max(proposals, key=lambda proposal: proposal.final_event_score)
    return RefinementOutcome(
        event_id, video_id, anchor_seconds, top.timestamp_seconds, False, len(frames), top
    )
