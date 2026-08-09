"""Live frame sampling and text-image scoring for adaptive search sessions.

The model-free service accepts precomputed ``FrameScoreSample`` artifacts. This
optional boundary creates them from injected decoder and embedder dependencies.
Importing this module never loads model weights or a video runtime, keeping the
orchestration testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Iterable

from .embedding import (
    EmbeddingConfigurationError,
    SimilarityCurves,
    TextImageEmbedder,
    score_event_frames,
)
from .models import EventDefinition, FrameScoreSample, ModelRuntimeSpec, TemporalRegion
from .providers import (
    FrameProviderCapabilities,
    FrameProviderError,
    FrameReference,
)
from .service import AdaptiveInputError, AdaptiveSearchService
from .session import RevisionConflictError, SearchRun, SessionBundle


MetricValue = int | float | str | bool | None


class LiveRefinementError(RuntimeError):
    """Base error for live refinement failures."""


class LiveRefinementUnavailableError(LiveRefinementError):
    """Decoder or model runtime is unavailable before any work starts."""


class LiveRefinementConfigurationError(LiveRefinementError):
    """Injected model identity differs from the pinned session spec."""


class LiveRefinementDataError(LiveRefinementError):
    """A provider response violates the canonical frame contract."""


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

        samples: list[FrameScoreSample] = []
        sampling_ms = embedding_ms = 0.0
        medium_frames = dense_frames = 0
        duplicate_frames = out_of_region_frames = 0
        fallback_state_queries = 0

        for region, region_budget in zip(regions, region_budgets):
            event = event_by_id[region.event_id]
            pre_query, post_query, used_fallback = _state_queries(event)
            fallback_state_queries += int(used_fallback)
            dense_budget = _dense_budget(
                region, region_budget, dense_interval_ms, dense_radius_ms
            )
            medium_budget = region_budget - dense_budget

            started = perf_counter()
            medium, stats = self._sample_region(
                region, interval_ms=medium_interval_ms, max_frames=medium_budget
            )
            sampling_ms += (perf_counter() - started) * 1000.0
            duplicate_frames += stats.duplicate_frames
            out_of_region_frames += stats.out_of_region_frames
            if not medium:
                raise LiveRefinementDataError(
                    f"provider returned no encodable frame for region {region.id}"
                )

            started = perf_counter()
            medium_curves = _score_frames(
                self.embedder,
                anchor_query=event.anchor_query,
                pre_state_query=pre_query,
                post_state_query=post_query,
                images=[frame.image for frame in medium],
                image_batch_size=parameters.embedding_batch_size,
            )
            embedding_ms += (perf_counter() - started) * 1000.0
            medium_frames += len(medium)
            region_samples = _build_samples(
                session_id,
                region,
                medium,
                medium_curves.anchor,
                medium_curves.pre,
                medium_curves.post,
            )

            if dense_budget:
                peak_index = max(
                    range(len(medium)),
                    key=lambda index: (medium_curves.anchor[index], -index),
                )
                peak_pts_ms = medium[peak_index].pts_ms
                dense_start = max(
                    _region_start_ms(region), peak_pts_ms - dense_radius_ms
                )
                dense_end = min(_region_end_ms(region), peak_pts_ms + dense_radius_ms)
                started = perf_counter()
                dense, stats = self._sample_region(
                    region,
                    interval_ms=dense_interval_ms,
                    max_frames=dense_budget,
                    start_pts_ms=dense_start,
                    end_pts_ms=dense_end,
                )
                sampling_ms += (perf_counter() - started) * 1000.0
                duplicate_frames += stats.duplicate_frames
                out_of_region_frames += stats.out_of_region_frames
                medium_pts = {frame.pts_ms for frame in medium}
                unique_dense = [
                    frame for frame in dense if frame.pts_ms not in medium_pts
                ]
                duplicate_frames += len(dense) - len(unique_dense)
                if unique_dense:
                    started = perf_counter()
                    dense_curves = _score_frames(
                        self.embedder,
                        anchor_query=event.anchor_query,
                        pre_state_query=pre_query,
                        post_state_query=post_query,
                        images=[frame.image for frame in unique_dense],
                        image_batch_size=parameters.embedding_batch_size,
                    )
                    embedding_ms += (perf_counter() - started) * 1000.0
                    dense_frames += len(unique_dense)
                    region_samples.extend(
                        _build_samples(
                            session_id,
                            region,
                            unique_dense,
                            dense_curves.anchor,
                            dense_curves.pre,
                            dense_curves.post,
                        )
                    )
            samples.extend(
                sorted(region_samples, key=lambda item: item.timestamp_seconds)
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
            "regions_refined": len(regions),
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
            region_ids=[region.id for region in regions],
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

    def _sample_region(
        self,
        region: TemporalRegion,
        *,
        interval_ms: int,
        max_frames: int,
        start_pts_ms: int | None = None,
        end_pts_ms: int | None = None,
    ) -> tuple[list[FrameReference], _SampleStats]:
        if max_frames <= 0:
            return [], _SampleStats()
        start = _region_start_ms(region) if start_pts_ms is None else start_pts_ms
        end = _region_end_ms(region) if end_pts_ms is None else end_pts_ms
        try:
            frames = self.service.frame_provider.sample_interval(
                region.video_id,
                start,
                end,
                interval_ms,
                max_frames=max_frames,
            )
        except FrameProviderError as exc:
            raise LiveRefinementDataError(
                f"frame provider failed for region {region.id}: {exc}"
            ) from exc
        if len(frames) > max_frames:
            raise LiveRefinementDataError(
                f"frame provider exceeded max_frames for region {region.id}"
            )
        unique: dict[int, FrameReference] = {}
        out_of_region = 0
        for frame in frames:
            _validate_frame(frame, region)
            if (
                frame.pts_ms < _region_start_ms(region)
                or frame.pts_ms > _region_end_ms(region)
            ):
                out_of_region += 1
                continue
            unique.setdefault(frame.pts_ms, frame)
        ordered = [unique[pts] for pts in sorted(unique)]
        return ordered, _SampleStats(
            duplicate_frames=len(frames) - out_of_region - len(ordered),
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
