"""Capability reporting and shared scoring helpers for the boundary-refinement stage.

Trimmed from the old live-refinement orchestrator (`refinement.py`): the
frontier-wide decode-and-embed orchestration (`refine_session`, region/video-group
sampling, budget planning) has been removed along with `commands/refine` and
`GET /tuples` - the coarse pipeline (`prioritize_videos`) already produces video
rankings without it, and the youcook2 benchmark (see
`irrelevant_things/benchmarks/youcook2/hyperparameter_sweep_results.md` and
`boundary_refinement_results.md`) established that stage never beat
`adaptive_coarse` on real data.

What remains here is what the much cheaper `boundary_refinement.py` stage (a
local per-event sweep, not a frontier-wide one) still needs: reporting whether
the SigLIP2 dense-scoring runtime is configured (`BoundaryRefinementRuntime`,
used by both `GET /searchers` and the new opt-in refinement endpoints), and the
pre/post-state-fallback + embedding-error-translation helpers shared by
`boundary_refinement.py` and `legacy_search`'s use of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embedding import (
    EmbeddingConfigurationError,
    SimilarityCurves,
    TextImageEmbedder,
    score_event_frames,
)
from .exceptions import (
    LiveRefinementConfigurationError,
    LiveRefinementDataError,
    LiveRefinementUnavailableError,
)
from .schemas import EventDefinition, ModelRuntimeSpec
from .providers import FrameProvider, FrameProviderCapabilities, UnavailableFrameProvider

MetricValue = int | float | str | bool | None


@dataclass(frozen=True)
class LiveRefinementCapabilities:
    """Combined decoder/model capability; provider availability alone is insufficient."""

    available: bool
    provider: FrameProviderCapabilities
    model_available: bool
    model_id: str | None
    model_revision: str | None
    runtime_model_spec: dict[str, MetricValue] | None
    reason: str | None = None


class BoundaryRefinementRuntime:
    """Holds the (frame_provider, embedder) pair boundary refinement needs and
    reports whether they're configured, without loading model weights.

    Decoupled from AdaptiveSearchService on purpose (the old orchestrator read
    `service.frame_provider`): boundary refinement is stateless and reused by
    legacy_search too, which has no session/service of its own.
    """

    def __init__(
        self,
        frame_provider: FrameProvider | None = None,
        embedder: TextImageEmbedder | None = None,
    ) -> None:
        self.frame_provider: FrameProvider = frame_provider or UnavailableFrameProvider()
        self.embedder = embedder

    def capabilities(
        self,
        runtime_spec: ModelRuntimeSpec | None = None,
    ) -> LiveRefinementCapabilities:
        """Report combined provider/model state without loading dependencies."""

        try:
            provider = self.frame_provider.capabilities()
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


def state_queries(event: EventDefinition) -> tuple[str, str, bool]:
    """pre/post-state text with anchor-query fallback, and whether it fell back."""

    used_fallback = event.pre_state is None or event.post_state is None
    return (
        event.pre_state or event.anchor_query,
        event.post_state or event.anchor_query,
        used_fallback,
    )


def score_frames(
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
