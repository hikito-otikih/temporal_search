"""Live frame sampling and text-image scoring for adaptive search sessions.

The model-free service accepts precomputed ``FrameScoreSample`` artifacts. This
optional boundary creates them from injected decoder and embedder dependencies.
Importing this module never loads model weights or a video runtime, keeping the
orchestration testable offline.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, replace as _dataclass_replace
from math import ceil
from time import perf_counter
from typing import Iterable

from .embedding import (
    EmbeddingConfigurationError,
    SimilarityCurves,
    TextImageEmbedder,
    score_event_frames,
)
from .exceptions import (
    AdaptiveInputError,
    LiveRefinementConfigurationError,
    LiveRefinementDataError,
    LiveRefinementUnavailableError,
    RevisionConflictError,
)
from .schemas import EventDefinition, FrameScoreSample, ModelRuntimeSpec, TemporalRegion
from .providers import (
    FrameProviderCapabilities,
    FrameProviderError,
    FrameReference,
)
from .service import AdaptiveSearchService
from .session import SearchRun, SessionBundle


MetricValue = int | float | str | bool | None

# Nearest-frame decoding rarely lands exactly on a region boundary timestamp
# (video frames sit on the stream's own ~33ms/40ms grid, not on the second).
# A frame within this tolerance of the boundary is real, in-region data that
# just snapped a few ms past the edge; clamp it in rather than discarding it.
_BOUNDARY_SNAP_TOLERANCE_MS = 100


@dataclass(frozen=True)
class LiveRefinementCapabilities:
    """Combined decoder/model capability; provider availability is insufficient."""

    available: bool
    provider: FrameProviderCapabilities
    model_available: bool
    model_id: str | None
    model_revision: str | None
    runtime_model_spec: dict[str, MetricValue] | None
    reason: str | None = None


@dataclass(frozen=True)
class LiveRefinementResult:
    """Committed state plus end-to-end orchestration measurements."""

    bundle: SessionBundle
    run: SearchRun
    metrics: dict[str, MetricValue]


@dataclass(frozen=True)
class _SampleStats:
    duplicate_frames: int = 0
    out_of_region_frames: int = 0


@dataclass(frozen=True)
class _RegionOutcome:
    """One region's contribution to a video-group refinement pass."""

    samples: list[FrameScoreSample]
    medium_frame_count: int
    dense_frame_count: int
    duplicate_frames: int
    out_of_region_frames: int
    used_fallback: bool
    failed_reason: str | None


@dataclass(frozen=True)
class _VideoGroupResult:
    regions: dict[str, _RegionOutcome]
    sampling_ms: float
    embedding_ms: float


@dataclass
class _PlannedRegion:
    region: TemporalRegion
    dense_budget: int
    medium_targets: list[int]


class LiveRefinementOrchestrator:
    """Run deterministic medium-to-dense refinement under a frame budget.

    Revision is checked before decoding and by the atomic core commit. Concurrent
    edits may waste inference work, but stale frame scores can never commit.
    """

    def __init__(
        self,
        service: AdaptiveSearchService,
        embedder: TextImageEmbedder | None = None,
    ) -> None:
        self.service = service
        self.embedder = embedder

    def capabilities(
        self,
        runtime_spec: ModelRuntimeSpec | None = None,
    ) -> LiveRefinementCapabilities:
        """Report combined provider/model state without loading dependencies."""

        try:
            provider = self.service.frame_provider.capabilities()
        except Exception as exc:  # pragma: no cover - deployment boundary
            provider = FrameProviderCapabilities(
                available=False,
                supports_batch=False,
                supports_dense_sampling=False,
                supports_thumbnails=False,
                reason=f"frame provider capability check failed: {exc}",
            )
        model_id = getattr(self.embedder, "model_id", None)
        model_revision = getattr(self.embedder, "revision", None)
        runtime_model_spec: dict[str, MetricValue] | None = None
        reasons: list[str] = []
        if not provider.available:
            reasons.append(provider.reason or "frame provider is unavailable")
        elif not provider.supports_dense_sampling:
            reasons.append("frame provider does not support dense interval sampling")

        model_available = self.embedder is not None
        if not model_available:
            reasons.append("text-image embedder is not configured")
        else:
            try:
                configured_runtime_spec = self.configured_runtime_spec()
                if configured_runtime_spec is not None:
                    runtime_model_spec = configured_runtime_spec.model_dump(
                        mode="json"
                    )
                _validate_embedder_identity(self.embedder, runtime_spec)
            except LiveRefinementConfigurationError as exc:
                model_available = False
                reasons.append(str(exc))
        return LiveRefinementCapabilities(
            available=not reasons,
            provider=provider,
            model_available=model_available,
            model_id=model_id,
            model_revision=model_revision,
            runtime_model_spec=runtime_model_spec,
            reason="; ".join(reasons) if reasons else None,
        )

    def configured_runtime_spec(self) -> ModelRuntimeSpec | None:
        """Return the exact active model identity without loading its weights."""

        if self.embedder is None:
            return None
        _validate_embedder_identity(self.embedder, None)
        return ModelRuntimeSpec(
            model_id=self.embedder.model_id,
            revision=self.embedder.revision,
            dimension=self.embedder.output_dimension,
            preprocess=self.embedder.preprocess_version,
            instruction=self.embedder.instruction,
        )

    def refine_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        region_ids: Iterable[str] | None = None,
        max_frames: int | None = None,
    ) -> LiveRefinementResult:
        """Sample selected/frontier regions, score them, and atomically commit.

        ``max_frames`` may lower but never exceed the session hard limit. At
        least one frame is reserved per region, so low budgets fail explicitly.
        """

        total_started = perf_counter()
        snapshot = self.service.get_session(session_id)
        if snapshot.session.revision != expected_revision:
            raise RevisionConflictError(expected_revision, snapshot.session.revision)
        regions = _select_regions(snapshot, region_ids)
        parameters = snapshot.session.hyperparameters.refinement
        runtime_spec = _active_runtime_spec(parameters)
        capability = self.capabilities(runtime_spec)
        if not capability.available:
            raise LiveRefinementUnavailableError(
                capability.reason or "live refinement is unavailable"
            )
        assert self.embedder is not None

        hard_budget = parameters.max_frames_per_run
        selected_budget = hard_budget if max_frames is None else max_frames
        if (
            isinstance(selected_budget, bool)
            or not isinstance(selected_budget, int)
            or selected_budget <= 0
        ):
            raise AdaptiveInputError("max_frames must be a positive integer")
        if selected_budget > hard_budget:
            raise AdaptiveInputError(
                "max_frames cannot exceed refinement.max_frames_per_run"
            )
        if selected_budget < len(regions):
            raise AdaptiveInputError(
                "max_frames must reserve at least one frame per selected region"
            )

        event_by_id = {event.event_id: event for event in snapshot.session.events}
        region_budgets = _fair_budgets(selected_budget, len(regions))
        medium_interval_ms = _seconds_to_ms(
            parameters.medium_interval_seconds, "medium_interval_seconds"
        )
        dense_interval_ms = _seconds_to_ms(
            parameters.dense_interval_seconds, "dense_interval_seconds"
        )
        dense_radius_ms = _seconds_to_ms(
            parameters.dense_radius_seconds, "dense_radius_seconds"
        )

        video_groups: dict[str, list[tuple[TemporalRegion, int]]] = defaultdict(list)
        for region, region_budget in zip(regions, region_budgets):
            video_groups[region.video_id].append((region, region_budget))

        samples: list[FrameScoreSample] = []
        sampling_ms = embedding_ms = 0.0
        medium_frames = dense_frames = 0
        duplicate_frames = out_of_region_frames = 0
        fallback_state_queries = 0
        refined_region_ids: list[str] = []
        failed_region_ids: list[str] = []
        last_failure_reason: str | None = None

        # Grouped by video (not processed one region at a time) so every
        # region of a video shares one decode pass and one embedding batch
        # per event instead of paying its own container-open and GPU call.
        #
        # Measured, did not keep: running these groups on a thread pool
        # (PyAV decode releases the GIL, so this looked like a free win)
        # made a 6-video/48-region session ~2.3x *slower* in this
        # environment (18.7s -> ~44s), reproduced twice. PyAV's own
        # thread_type="AUTO" already parallelizes decode inside one
        # container with native threads; running several containers
        # concurrently multiplies that against 16 CPU cores and, combined
        # with concurrent reads off the WSL-mounted Windows video
        # directory, thrashes instead of overlapping. Sequential per-video
        # processing measured faster - see the timing harness this was
        # verified with.
        for video_id in sorted(video_groups):
            group_result = self._refine_video_group(
                session_id,
                video_id,
                video_groups[video_id],
                event_by_id,
                medium_interval_ms=medium_interval_ms,
                dense_interval_ms=dense_interval_ms,
                dense_radius_ms=dense_radius_ms,
                embedding_batch_size=parameters.embedding_batch_size,
            )
            sampling_ms += group_result.sampling_ms
            embedding_ms += group_result.embedding_ms
            for region, _ in video_groups[video_id]:
                outcome = group_result.regions[region.id]
                if outcome.failed_reason is not None:
                    # One region's (or one video's) frame data being unusable
                    # should not discard every other region's already-decoded
                    # work - same isolation guarantee as before, just decided
                    # per video-group instead of per single decode call.
                    failed_region_ids.append(region.id)
                    last_failure_reason = outcome.failed_reason
                    continue
                refined_region_ids.append(region.id)
                medium_frames += outcome.medium_frame_count
                dense_frames += outcome.dense_frame_count
                duplicate_frames += outcome.duplicate_frames
                out_of_region_frames += outcome.out_of_region_frames
                fallback_state_queries += int(outcome.used_fallback)
                samples.extend(outcome.samples)

        if not refined_region_ids:
            detail = f"; last error: {last_failure_reason}" if last_failure_reason else ""
            raise LiveRefinementDataError(
                f"live refinement failed for all {len(regions)} selected regions{detail}"
            )

        if len(samples) > selected_budget:
            raise LiveRefinementDataError(
                "live refinement exceeded the configured frame budget"
            )
        precommit_ms = (perf_counter() - total_started) * 1000.0
        run_metrics: dict[str, MetricValue] = {
            "live_refinement": True,
            "refinement_model_id": runtime_spec.model_id,
            "refinement_model_revision": runtime_spec.revision,
            "refinement_model_dimension": runtime_spec.dimension,
            "refinement_preprocess": runtime_spec.preprocess,
            "regions_requested": len(regions),
            "regions_refined": len(refined_region_ids),
            "regions_failed": len(failed_region_ids),
            "frame_budget": selected_budget,
            "embedding_batch_size": parameters.embedding_batch_size,
            "frames_sampled": len(samples),
            "medium_frames_sampled": medium_frames,
            "dense_frames_sampled": dense_frames,
            "duplicate_frames_dropped": duplicate_frames,
            "out_of_region_frames_dropped": out_of_region_frames,
            "fallback_state_queries": fallback_state_queries,
            "motion_scores_available": False,
            "sampling_ms": sampling_ms,
            "embedding_ms": embedding_ms,
            "precommit_elapsed_ms": precommit_ms,
        }
        core_started = perf_counter()
        bundle, run = self.service.replace_frame_scores(
            session_id,
            expected_revision=expected_revision,
            region_ids=refined_region_ids,
            samples=samples,
            run_metrics=run_metrics,
        )
        core_ms = (perf_counter() - core_started) * 1000.0
        result_metrics = {
            **run_metrics,
            **run.metrics,
            "core_commit_ms": core_ms,
            "total_elapsed_ms": (perf_counter() - total_started) * 1000.0,
        }
        return LiveRefinementResult(bundle=bundle, run=run, metrics=result_metrics)

    def _refine_video_group(
        self,
        session_id: str,
        video_id: str,
        region_budgets: list[tuple[TemporalRegion, int]],
        event_by_id: dict[str, EventDefinition],
        *,
        medium_interval_ms: int,
        dense_interval_ms: int,
        dense_radius_ms: int,
        embedding_batch_size: int,
    ) -> _VideoGroupResult:
        """Batch decode and score every region of one video together.

        One container-open per phase (medium, then dense) covers every
        region of this video instead of one per region, and one embedding
        call per (event, phase) covers every region of that event instead of
        one per region. The per-region science - bounds, budgets, boundary
        clamping, the dense peak search - is unchanged; only how many decode
        and GPU calls it takes to get there changes.
        """

        provider = self.service.frame_provider
        outcomes: dict[str, _RegionOutcome] = {}
        sampling_ms = embedding_ms = 0.0

        planned: dict[str, _PlannedRegion] = {}
        for region, region_budget in region_budgets:
            dense_budget = _dense_budget(
                region, region_budget, dense_interval_ms, dense_radius_ms
            )
            medium_budget = region_budget - dense_budget
            targets = (
                provider.plan_interval(
                    _region_start_ms(region),
                    _region_end_ms(region),
                    medium_interval_ms,
                    max_frames=medium_budget,
                )
                if medium_budget > 0
                else []
            )
            planned[region.id] = _PlannedRegion(
                region=region, dense_budget=dense_budget, medium_targets=targets
            )

        union_medium = sorted({pts for plan in planned.values() for pts in plan.medium_targets})
        started = perf_counter()
        try:
            medium_pool = provider.get_frames(video_id, union_medium) if union_medium else []
        except FrameProviderError as exc:
            sampling_ms += (perf_counter() - started) * 1000.0
            reason = f"frame provider failed for video {video_id}: {exc}"
            return _VideoGroupResult(
                regions={
                    region_id: _failed_outcome(reason) for region_id in planned
                },
                sampling_ms=sampling_ms,
                embedding_ms=0.0,
            )
        sampling_ms += (perf_counter() - started) * 1000.0

        region_medium: dict[str, tuple[list[FrameReference], _SampleStats]] = {}
        medium_by_event: dict[str, list[str]] = defaultdict(list)
        for region_id, plan in planned.items():
            if not plan.medium_targets:
                outcomes[region_id] = _failed_outcome(
                    f"provider returned no encodable frame for region {plan.region.id}"
                )
                continue
            extracted, stats = _extract_region_frames(
                medium_pool, plan.region, plan.medium_targets
            )
            if not extracted:
                outcomes[region_id] = _failed_outcome(
                    f"provider returned no encodable frame for region {plan.region.id}",
                    duplicate_frames=stats.duplicate_frames,
                    out_of_region_frames=stats.out_of_region_frames,
                )
                continue
            region_medium[region_id] = (extracted, stats)
            medium_by_event[plan.region.event_id].append(region_id)

        region_samples: dict[str, list[FrameScoreSample]] = defaultdict(list)
        region_medium_curves: dict[str, SimilarityCurves] = {}
        embedding_ms += self._score_grouped(
            session_id,
            medium_by_event,
            planned,
            region_medium,
            event_by_id,
            embedding_batch_size,
            region_samples,
            region_medium_curves,
        )

        dense_targets: dict[str, list[int]] = {}
        for region_id, (frames, _stats) in region_medium.items():
            plan = planned[region_id]
            if not plan.dense_budget:
                continue
            curve = region_medium_curves[region_id]
            peak_index = max(
                range(len(frames)), key=lambda index: (curve.anchor[index], -index)
            )
            peak_pts_ms = frames[peak_index].pts_ms
            region = plan.region
            dense_start = max(_region_start_ms(region), peak_pts_ms - dense_radius_ms)
            dense_end = min(_region_end_ms(region), peak_pts_ms + dense_radius_ms)
            dense_targets[region_id] = provider.plan_interval(
                dense_start, dense_end, dense_interval_ms, max_frames=plan.dense_budget
            )

        union_dense = sorted({pts for targets in dense_targets.values() for pts in targets})
        dense_pool: list[FrameReference] = []
        if union_dense:
            started = perf_counter()
            try:
                dense_pool = provider.get_frames(video_id, union_dense)
            except FrameProviderError as exc:
                sampling_ms += (perf_counter() - started) * 1000.0
                # Dense sampling failing for a region discards that region's
                # medium work too, matching the pre-batching behavior where a
                # dense decode failure raised before any samples were built.
                reason = f"frame provider failed for video {video_id}: {exc}"
                for region_id in dense_targets:
                    outcomes[region_id] = _failed_outcome(reason)
                dense_targets = {}
            else:
                sampling_ms += (perf_counter() - started) * 1000.0

        region_dense: dict[str, tuple[list[FrameReference], _SampleStats]] = {}
        region_extra_duplicates: dict[str, int] = defaultdict(int)
        dense_by_event: dict[str, list[str]] = defaultdict(list)
        for region_id, targets in dense_targets.items():
            if region_id in outcomes or not targets:
                continue
            region = planned[region_id].region
            extracted, stats = _extract_region_frames(dense_pool, region, targets)
            region_extra_duplicates[region_id] += stats.duplicate_frames
            medium_pts = {frame.pts_ms for frame in region_medium[region_id][0]}
            unique_dense = [frame for frame in extracted if frame.pts_ms not in medium_pts]
            region_extra_duplicates[region_id] += len(extracted) - len(unique_dense)
            if not unique_dense:
                continue
            region_dense[region_id] = (unique_dense, stats)
            dense_by_event[region.event_id].append(region_id)

        dense_frame_counts: dict[str, int] = defaultdict(int)
        embedding_ms += self._score_grouped(
            session_id,
            dense_by_event,
            planned,
            region_dense,
            event_by_id,
            embedding_batch_size,
            region_samples,
            None,
            frame_counts=dense_frame_counts,
        )

        for region_id, plan in planned.items():
            if region_id in outcomes:
                continue
            frames, medium_stats = region_medium[region_id]
            _, dense_stats = region_dense.get(region_id, ([], _SampleStats()))
            _, _, used_fallback = _state_queries(event_by_id[plan.region.event_id])
            outcomes[region_id] = _RegionOutcome(
                samples=sorted(
                    region_samples[region_id], key=lambda item: item.timestamp_seconds
                ),
                medium_frame_count=len(frames),
                dense_frame_count=dense_frame_counts.get(region_id, 0),
                duplicate_frames=(
                    medium_stats.duplicate_frames
                    + region_extra_duplicates.get(region_id, 0)
                ),
                out_of_region_frames=(
                    medium_stats.out_of_region_frames + dense_stats.out_of_region_frames
                ),
                used_fallback=used_fallback,
                failed_reason=None,
            )

        return _VideoGroupResult(
            regions=outcomes, sampling_ms=sampling_ms, embedding_ms=embedding_ms
        )

    def _score_grouped(
        self,
        session_id: str,
        region_ids_by_event: dict[str, list[str]],
        planned: dict[str, _PlannedRegion],
        region_frames: dict[str, tuple[list[FrameReference], _SampleStats]],
        event_by_id: dict[str, EventDefinition],
        embedding_batch_size: int,
        region_samples: dict[str, list[FrameScoreSample]],
        region_curves_out: dict[str, SimilarityCurves] | None,
        *,
        frame_counts: dict[str, int] | None = None,
    ) -> float:
        """Score every region of one event together in a single embed call."""

        total_embedding_ms = 0.0
        for event_id, region_ids in region_ids_by_event.items():
            event = event_by_id[event_id]
            pre_query, post_query, _ = _state_queries(event)
            images: list[object] = []
            offsets: dict[str, tuple[int, int]] = {}
            for region_id in region_ids:
                frames, _stats = region_frames[region_id]
                start = len(images)
                images.extend(frame.image for frame in frames)
                offsets[region_id] = (start, len(images))
            started = perf_counter()
            curves = _score_frames(
                self.embedder,
                anchor_query=event.anchor_query,
                pre_state_query=pre_query,
                post_state_query=post_query,
                images=images,
                image_batch_size=embedding_batch_size,
            )
            total_embedding_ms += (perf_counter() - started) * 1000.0
            for region_id in region_ids:
                frames, _stats = region_frames[region_id]
                start, end = offsets[region_id]
                curve = SimilarityCurves(
                    anchor=curves.anchor[start:end],
                    pre=curves.pre[start:end],
                    post=curves.post[start:end],
                )
                if region_curves_out is not None:
                    region_curves_out[region_id] = curve
                region = planned[region_id].region
                region_samples[region_id].extend(
                    _build_samples(
                        session_id, region, frames, curve.anchor, curve.pre, curve.post
                    )
                )
                if frame_counts is not None:
                    frame_counts[region_id] += len(frames)
        return total_embedding_ms


def _failed_outcome(
    reason: str, *, duplicate_frames: int = 0, out_of_region_frames: int = 0
) -> _RegionOutcome:
    return _RegionOutcome(
        samples=[],
        medium_frame_count=0,
        dense_frame_count=0,
        duplicate_frames=duplicate_frames,
        out_of_region_frames=out_of_region_frames,
        used_fallback=False,
        failed_reason=reason,
    )


def _nearest_pts(sorted_pts: list[int], target_ms: int) -> int:
    """Return the pool timestamp closest to ``target_ms`` (ties -> earlier)."""

    index = bisect_left(sorted_pts, target_ms)
    if index == 0:
        return sorted_pts[0]
    if index == len(sorted_pts):
        return sorted_pts[-1]
    before, after = sorted_pts[index - 1], sorted_pts[index]
    return before if target_ms - before <= after - target_ms else after


def _extract_region_frames(
    pool: list[FrameReference], region: TemporalRegion, targets: list[int]
) -> tuple[list[FrameReference], _SampleStats]:
    """Recover one region's own frames from a video-wide decoded pool.

    ``pool`` may hold frames decoded for every region of a video in one
    batch call. ``targets`` is this region's own requested grid (from
    ``plan_interval``); each target was part of the union that produced
    ``pool``, so the pool timestamp nearest to a target is exactly what a
    solo decode for this region alone would have produced - looking targets
    up this way (instead of taking whatever falls inside the region's time
    window) keeps a region from silently absorbing a neighboring region's
    frames and exceeding its own budget.
    """

    if not pool or not targets:
        return [], _SampleStats()
    sorted_pts = sorted(frame.pts_ms for frame in pool)
    by_pts = {frame.pts_ms: frame for frame in pool}
    region_start = _region_start_ms(region)
    region_end = _region_end_ms(region)
    unique: dict[int, FrameReference] = {}
    out_of_region = 0
    for target_ms in targets:
        frame = by_pts[_nearest_pts(sorted_pts, target_ms)]
        _validate_frame(frame, region)
        pts_ms = frame.pts_ms
        if pts_ms < region_start or pts_ms > region_end:
            snapped_ms = min(max(pts_ms, region_start), region_end)
            if abs(pts_ms - snapped_ms) > _BOUNDARY_SNAP_TOLERANCE_MS:
                out_of_region += 1
                continue
            # Nearest-frame decoding snapped just past the boundary because
            # no real frame lands exactly on it; clamp it back in rather
            # than discarding genuine in-region data.
            frame = _dataclass_replace(frame, pts_ms=snapped_ms)
            pts_ms = snapped_ms
        unique.setdefault(pts_ms, frame)
    ordered = [unique[pts] for pts in sorted(unique)]
    return ordered, _SampleStats(
        duplicate_frames=len(targets) - out_of_region - len(ordered),
        out_of_region_frames=out_of_region,
    )


def _select_regions(
    snapshot: SessionBundle, region_ids: Iterable[str] | None
) -> list[TemporalRegion]:
    selected_ids = (
        list(snapshot.artifacts.frontier_region_ids)
        if region_ids is None
        else list(region_ids)
    )
    if not selected_ids:
        raise AdaptiveInputError("no refinement regions were selected")
    if len(selected_ids) != len(set(selected_ids)):
        raise AdaptiveInputError("region_ids must be unique")
    region_by_id = {region.id: region for region in snapshot.artifacts.regions}
    missing = set(selected_ids) - set(region_by_id)
    if missing:
        raise AdaptiveInputError(f"unknown region_ids: {sorted(missing)}")

    rejected_by_event = {
        event_id: constraint.rejected_region_ids
        for event_id, constraint in snapshot.session.constraints.event_constraints.items()
    }
    selected = [region_by_id[region_id] for region_id in selected_ids]
    rejected = [
        region.id
        for region in selected
        if region.user_status == "rejected"
        or region.id in rejected_by_event.get(region.event_id, frozenset())
    ]
    if rejected:
        raise AdaptiveInputError(f"cannot refine rejected regions: {sorted(rejected)}")
    return selected


def _score_frames(
    embedder: TextImageEmbedder,
    *,
    anchor_query: str,
    pre_state_query: str,
    post_state_query: str,
    images: list[object],
    image_batch_size: int,
) -> SimilarityCurves:
    try:
        return score_event_frames(
            embedder,
            anchor_query=anchor_query,
            pre_state_query=pre_state_query,
            post_state_query=post_state_query,
            images=images,
            image_batch_size=image_batch_size,
        )
    except (EmbeddingConfigurationError, OSError, RuntimeError) as exc:
        raise LiveRefinementUnavailableError(
            "embedding runtime failed "
            f"({type(exc).__name__}); inspect server logs and runtime setup"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise LiveRefinementDataError(
            f"embedding output violated the scoring contract ({type(exc).__name__})"
        ) from exc


def _active_runtime_spec(parameters) -> ModelRuntimeSpec:
    if parameters.quality_profile_enabled:
        if parameters.quality_embedding_model is None:  # guarded by Pydantic
            raise LiveRefinementConfigurationError(
                "quality embedding model is not configured"
            )
        return parameters.quality_embedding_model
    return parameters.embedding_model


def _validate_embedder_identity(
    embedder: TextImageEmbedder, runtime_spec: ModelRuntimeSpec | None
) -> None:
    required = {
        "model_id": getattr(embedder, "model_id", None),
        "revision": getattr(embedder, "revision", None),
        "output_dimension": getattr(embedder, "output_dimension", None),
        "preprocess_version": getattr(embedder, "preprocess_version", None),
        "instruction": getattr(embedder, "instruction", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise LiveRefinementConfigurationError(
            f"embedder is missing runtime identity fields: {', '.join(missing)}"
        )
    if not required["revision"] or required["revision"] == "main":
        raise LiveRefinementConfigurationError(
            "embedder revision must be an immutable model commit"
        )
    if (
        isinstance(required["output_dimension"], bool)
        or not isinstance(required["output_dimension"], int)
        or required["output_dimension"] <= 0
    ):
        raise LiveRefinementConfigurationError(
            "embedder output_dimension must be a positive integer"
        )
    if runtime_spec is None:
        return
    if not runtime_spec.revision or runtime_spec.revision == "main":
        raise LiveRefinementConfigurationError(
            "session model revision must be pinned to an immutable commit"
        )
    mismatches: list[str] = []
    expected = {
        "model_id": runtime_spec.model_id,
        "revision": runtime_spec.revision,
        "preprocess_version": runtime_spec.preprocess,
        "instruction": runtime_spec.instruction,
    }
    for name, value in expected.items():
        if required[name] != value:
            mismatches.append(f"{name} expected {value!r}, got {required[name]!r}")
    if (
        runtime_spec.dimension is not None
        and required["output_dimension"] != runtime_spec.dimension
    ):
        mismatches.append(
            "output_dimension expected "
            f"{runtime_spec.dimension}, got {required['output_dimension']}"
        )
    if mismatches:
        raise LiveRefinementConfigurationError(
            "embedder does not match session runtime spec: " + "; ".join(mismatches)
        )


def _state_queries(event: EventDefinition) -> tuple[str, str, bool]:
    used_fallback = event.pre_state is None or event.post_state is None
    return (
        event.pre_state or event.anchor_query,
        event.post_state or event.anchor_query,
        used_fallback,
    )


def _fair_budgets(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def _dense_budget(
    region: TemporalRegion,
    region_budget: int,
    dense_interval_ms: int,
    dense_radius_ms: int,
) -> int:
    if region_budget < 2 or region.end_seconds <= region.start_seconds:
        return 0
    span_ms = min(
        _region_end_ms(region) - _region_start_ms(region), 2 * dense_radius_ms
    )
    natural_dense_frames = ceil(span_ms / dense_interval_ms) + 1
    return min(natural_dense_frames, region_budget // 2)


def _build_samples(
    session_id: str,
    region: TemporalRegion,
    frames: list[FrameReference],
    anchor: list[float],
    pre: list[float],
    post: list[float],
) -> list[FrameScoreSample]:
    if not (len(frames) == len(anchor) == len(pre) == len(post)):
        raise LiveRefinementDataError("embedding curves do not match sampled frames")
    return [
        FrameScoreSample(
            session_id=session_id,
            event_id=region.event_id,
            video_id=region.video_id,
            region_id=region.id,
            frame_id=frame.frame_index if frame.frame_index is not None else frame.pts_ms,
            timestamp_seconds=frame.timestamp_seconds,
            raw_anchor_score=float(anchor[index]),
            raw_pre_score=float(pre[index]),
            raw_post_score=float(post[index]),
            raw_motion_score=0.0,
        )
        for index, frame in enumerate(frames)
    ]


def _validate_frame(frame: FrameReference, region: TemporalRegion) -> None:
    if frame.video_id != region.video_id:
        raise LiveRefinementDataError(
            f"provider returned video {frame.video_id!r} for region {region.id}"
        )
    if isinstance(frame.pts_ms, bool) or not isinstance(frame.pts_ms, int):
        raise LiveRefinementDataError("provider pts_ms must be an integer")
    if frame.pts_ms < 0:
        raise LiveRefinementDataError("provider pts_ms must be non-negative")
    if frame.frame_index is not None and frame.frame_index < 0:
        raise LiveRefinementDataError("provider frame_index must be non-negative")
    if frame.image is None:
        raise LiveRefinementDataError(
            "provider returned metadata-only frame without pixels"
        )


def _seconds_to_ms(value: float, label: str) -> int:
    result = int(round(value * 1000.0))
    if result <= 0:
        raise LiveRefinementConfigurationError(
            f"{label} is below the provider one-millisecond resolution"
        )
    return result


def _region_start_ms(region: TemporalRegion) -> int:
    return int(round(region.start_seconds * 1000.0))


def _region_end_ms(region: TemporalRegion) -> int:
    return int(round(region.end_seconds * 1000.0))
