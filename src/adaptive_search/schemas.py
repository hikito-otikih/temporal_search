"""Strict domain models for the adaptive temporal-search core.

The module deliberately uses explicit ``*_seconds`` field names.  The legacy
code mixes frame indices and timestamp strings; keeping seconds canonical at
the adaptive-core boundary prevents unit ambiguity in API and cache payloads.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


NonEmptyStr = Annotated[str, Field(min_length=1)]
RawScore = Annotated[float, Field(allow_inf_nan=False)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Seconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


BoundaryType = Literal[
    "onset",
    "offset",
    "transition",
    "plateau_start",
    "symmetric_peak",
    "state",
    "unknown",
]
RefinementStatus = Literal[
    "pending",
    "medium",
    "dense",
    "completed",
    "failed",
]
RegionUserStatus = Literal["active", "fixed", "rejected"]
ProposalSource = Literal["sparse", "medium", "dense", "user"]
ProposalUserStatus = Literal["active", "fixed", "confirmed", "rejected"]
TemporalRelationType = Literal[
    "sequence_start",
    "after",
    "before",
    "during",
    "simultaneous",
    "independent",
    "unknown",
]


class StrictModel(BaseModel):
    """Base class shared by externally serialized adaptive-search models."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class ModelRuntimeSpec(StrictModel):
    """Every value that can change model output or an embedding cache key."""

    model_id: NonEmptyStr
    revision: NonEmptyStr = "main"
    dimension: PositiveInt | None = None
    preprocess: NonEmptyStr
    instruction: str = ""


def _runtime_embedding_model() -> ModelRuntimeSpec:
    return ModelRuntimeSpec(
        model_id="google/siglip2-base-patch16-224",
        revision="main",
        dimension=768,
        preprocess="siglip2-auto-processor-v1",
        instruction="",
    )


def _quality_embedding_model() -> ModelRuntimeSpec:
    return ModelRuntimeSpec(
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        revision="main",
        dimension=1024,
        preprocess="qwen3-vl-embedding:v1",
        instruction=(
            "Retrieve video frames that visually depict the described event."
        ),
    )


def _quality_reranker_model() -> ModelRuntimeSpec:
    return ModelRuntimeSpec(
        model_id="Qwen/Qwen3-VL-Reranker-2B",
        revision="main",
        dimension=1,
        preprocess="qwen3-vl-reranker:v1",
        instruction=(
            "Judge whether the ordered video frames depict the described "
            "events in the required chronological order."
        ),
    )


class RetrievalHyperparameters(StrictModel):
    # top_n_per_variant/top_n_fused: winner config from the youcook2 benchmark's
    # hyperparameter sweep (research_tools/benchmarks/youcook2/hyperparameter_sweep_results.md) -
    # wide retrieval measurably improved adaptive_coarse recall over the previous
    # narrower defaults with no downside found.
    top_n_per_variant: PositiveInt = 500
    top_n_fused: PositiveInt = 1000
    rrf_k: PositiveInt = 60
    query_variants_per_event: PositiveInt = 4


class ClusteringHyperparameters(StrictModel):
    gap_seconds: Seconds = 3.0
    margin_seconds: Seconds = 3.0
    max_region_seconds: PositiveFloat = 30.0

    @model_validator(mode="after")
    def validate_region_limit(self) -> "ClusteringHyperparameters":
        if self.max_region_seconds < 2.0 * self.margin_seconds:
            raise ValueError(
                "max_region_seconds must be >= 2 * margin_seconds"
            )
        return self


class RefinementHyperparameters(StrictModel):
    """Frontier budget, video-priority blend weights, and fully
    fingerprintable inference configuration. Dense/medium sampling-interval
    fields were removed with `commands/refine` (the dense orchestrator that
    was their only consumer) - see `docs/ADAPTIVE_PIPELINE_MIGRATION.md`."""

    embedding_model: ModelRuntimeSpec = Field(
        default_factory=_runtime_embedding_model
    )
    quality_embedding_model: ModelRuntimeSpec | None = Field(
        default_factory=_quality_embedding_model
    )
    reranker_model: ModelRuntimeSpec | None = Field(
        default_factory=_quality_reranker_model
    )
    quality_profile_enabled: bool = False
    use_reranker: bool = False

    max_initial_videos: PositiveInt = 20
    max_regions_per_event_per_video: PositiveInt = 5
    max_total_regions: PositiveInt = 400
    exploration_region_ratio: UnitScore = 0.15

    # Winner config from the same hyperparameter sweep: mean-only ranking beat
    # every blend that included coverage or min once retrieval is wide (see
    # hyperparameter_sweep_results.md finding #2).
    video_coverage_weight: NonNegativeFloat = 0.0
    video_mean_weight: NonNegativeFloat = 1.0
    video_min_weight: NonNegativeFloat = 0.0
    # Distinctness: penalizes a video whose covered events' chosen timestamps
    # collapse onto (or near) the same physical moment - a real, observed
    # failure mode of pure mean-of-independent-maxes ranking (one frame
    # superficially matching several events' text queries, not a genuine
    # multi-event match). 0.0 (off) pending the n=60 sweep this was added
    # for - see region_tuple_ranking_results.md. `distinctness_norm_seconds`
    # reuses the pipeline's existing 3.0s margin/gap scale rather than
    # introducing a new constant.
    video_distinctness_weight: NonNegativeFloat = 0.0
    distinctness_norm_seconds: PositiveFloat = 3.0

    @model_validator(mode="after")
    def validate_refinement_configuration(self) -> "RefinementHyperparameters":
        if self.quality_profile_enabled and self.quality_embedding_model is None:
            raise ValueError(
                "quality_embedding_model is required when quality_profile_enabled"
            )
        if self.use_reranker:
            raise ValueError(
                "use_reranker=true is not supported until the ordered-frame "
                "reranker runtime is implemented"
            )
        video_weight = (
            self.video_coverage_weight
            + self.video_mean_weight
            + self.video_min_weight
            + self.video_distinctness_weight
        )
        if video_weight <= 0.0:
            raise ValueError("at least one video-priority weight must be positive")
        return self


class BoundaryHyperparameters(StrictModel):
    window_options_seconds: tuple[PositiveFloat, ...] = (0.5, 1.0, 2.0, 3.0)
    min_samples_per_side: PositiveInt = 2
    pairwise_temperature: PositiveFloat = 1.0
    anchor_clip_z: PositiveFloat = 8.0

    post_contrast_weight: NonNegativeFloat = 0.5
    pre_contrast_weight: NonNegativeFloat = 0.5
    motion_contrast_weight: NonNegativeFloat = 0.1
    window_length_regularization: NonNegativeFloat = 0.01
    window_asymmetry_regularization: NonNegativeFloat = 0.01

    semantic_weight: NonNegativeFloat = 0.4
    boundary_weight: NonNegativeFloat = 0.3
    pre_weight: NonNegativeFloat = 0.1
    post_weight: NonNegativeFloat = 0.2

    nms_radius_seconds: Seconds = 0.5
    max_proposals_per_region: PositiveInt = 5

    @field_validator("window_options_seconds", mode="before")
    @classmethod
    def accept_json_window_array(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_boundary_configuration(self) -> "BoundaryHyperparameters":
        windows = self.window_options_seconds
        if not windows:
            raise ValueError("window_options_seconds must not be empty")
        if len(set(windows)) != len(windows):
            raise ValueError("window_options_seconds must contain unique values")
        if tuple(sorted(windows)) != windows:
            raise ValueError("window_options_seconds must be strictly increasing")
        contrast_weight = (
            self.post_contrast_weight
            + self.pre_contrast_weight
            + self.motion_contrast_weight
        )
        if contrast_weight <= 0.0:
            raise ValueError("at least one boundary contrast weight must be positive")
        proposal_weight = (
            self.semantic_weight
            + self.boundary_weight
            + self.pre_weight
            + self.post_weight
        )
        if proposal_weight <= 0.0:
            raise ValueError("at least one proposal-score weight must be positive")
        return self


class TupleRankingHyperparameters(StrictModel):
    """Multi-region pooling + order-aware tuple video ranking - the sole
    ranking algorithm behind `GET .../video-priorities`.
    `prioritize_videos()` (independent per-event max) still exists and is
    still tested, but nothing in this endpoint calls it any more.

    Validated against real YouCook2 queries, n=60
    (research_tools/benchmarks/youcook2/region_tuple_ranking_results.md):
    +12.4% final_query_score over prioritize_videos() (1.2133 -> 1.3633),
    concentrated on the queries prioritize_videos() gets wrong (hard-subset
    MRR +78%). Defaults below are that report's winning, seven-round-
    validated configuration. This has only been validated on one corpus
    (YouCook2) - re-check before assuming these defaults transfer to a
    materially different one.
    """

    absolute_threshold: UnitScore = 0.0
    """alpha: a region must score >= this on normalized_coarse_score."""

    relative_delta: NonNegativeFloat = 0.15
    """delta: a region must score >= (event's best region score - delta)."""

    max_regions_per_event: PositiveInt = 100_000
    """N: cap on pooled regions per (event, video), best-score-first.
    Effectively unbounded - a real, measured exhaustive search (no N cap,
    no max_combinations_per_video cap) over the full n=60 corpus, every
    candidate video per query, produced byte-identical ranks on 59/60
    videos and identical aggregate metrics to N=20, at +3.2% wall-clock
    cost (11.8s exhaustive vs. 11.5s capped, both dwarfed by ~170s of
    upstream fetch for the same run). `relative_delta` is the real
    relevance gate; N was already not truncating anything real on this
    corpus. Not a guarantee for a longer or denser corpus - re-check first
    if pool sizes grow.
    """

    order_weight: NonNegativeFloat = 0.8
    """Weight of the (confidence-gated) order-agreement term added to a
    tuple's mean region score."""

    pooling: Literal["max", "mean"] = "max"
    """How a video's kept tuples combine into one video score: "max" (best
    tuple wins) or "mean" (average of up to max_tuples_per_video). "mean"
    scored well alone in the benchmark but combined poorly with
    confidence_gate - not used together in the winning configuration."""

    confidence_gate: Literal["none", "linear", "threshold", "margin"] = "threshold"
    """How `order_weight` is scaled before being applied - "none" (fixed
    order_weight, caused the tail-recall regression), "linear" (order_weight
    * region_mean_score), "threshold" (order_weight if region_mean_score >=
    confidence_gate_threshold else 0 - the winning choice as of the n=30/60
    sweeps), or "margin" (same hard-cutoff shape as "threshold", but gated on
    each event's chosen-region-vs-best-alternative score gap instead of the
    tuple's mean region score - see `tuple_ranking._region_margin`). See
    region_tuple_ranking_results.md for the fine-grained tau sweep comparing
    "threshold" against "margin"."""

    confidence_gate_threshold: UnitScore = 0.5
    """Cutoff for confidence_gate="threshold" or "margin". See
    region_tuple_ranking_results.md for the fine-grained sweep this default
    was chosen from."""

    max_combinations_per_video: PositiveInt = 10_000_000
    """Hard cap on tuples explored per video during backtracking - a safety
    valve against the N^event_count combinatorial blowup pooling many
    regions across many events could otherwise cause. Search order is
    best-region-first per event, so this cap truncates from the
    least-promising end first. Raised alongside max_regions_per_event: on
    the real n=60 corpus the worst single video's true (uncapped) combination
    count was 19,008 and the worst full-query total (summed across every
    candidate video) was 21,467 - this cap is a safety net for pathological
    inputs, not a lever that was actually shaping results here."""

    max_tuples_per_video: PositiveInt = 20
    """How many top-scoring tuples per video to retain (only matters for
    pooling="mean")."""


class SearchHyperparameters(StrictModel):
    retrieval: RetrievalHyperparameters = Field(
        default_factory=RetrievalHyperparameters
    )
    clustering: ClusteringHyperparameters = Field(
        default_factory=ClusteringHyperparameters
    )
    refinement: RefinementHyperparameters = Field(
        default_factory=RefinementHyperparameters
    )
    boundary: BoundaryHyperparameters = Field(
        default_factory=BoundaryHyperparameters
    )
    tuple_ranking: TupleRankingHyperparameters = Field(
        default_factory=TupleRankingHyperparameters
    )


class EventDefinition(StrictModel):
    event_id: NonEmptyStr
    original_query: NonEmptyStr
    anchor_query: NonEmptyStr
    pre_state: NonEmptyStr | None = None
    post_state: NonEmptyStr | None = None
    boundary_type: BoundaryType = "unknown"
    temporal_relation: TemporalRelationType = "unknown"
    reference_event_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_transition_states(self) -> "EventDefinition":
        transition_profiles = {"onset", "offset", "transition", "plateau_start"}
        if self.boundary_type in transition_profiles and (
            self.pre_state is None or self.post_state is None
        ):
            raise ValueError(
                "transition boundary profiles require both pre_state and post_state"
            )
        return self

    @model_validator(mode="after")
    def validate_temporal_relation_reference(self) -> "EventDefinition":
        relation_requires_reference = self.temporal_relation in {
            "after", "before", "during", "simultaneous",
        }
        if relation_requires_reference and self.reference_event_id is None:
            raise ValueError(
                f"temporal_relation={self.temporal_relation!r} requires reference_event_id"
            )
        if not relation_requires_reference and self.reference_event_id is not None:
            raise ValueError(
                f"temporal_relation={self.temporal_relation!r} requires reference_event_id to be null"
            )
        if self.reference_event_id == self.event_id:
            raise ValueError("reference_event_id must not reference the event itself")
        return self


class SparseCandidate(StrictModel):
    id: NonEmptyStr
    session_id: NonEmptyStr
    event_id: NonEmptyStr
    video_id: NonEmptyStr
    frame_id: NonNegativeInt
    timestamp_seconds: Seconds
    raw_relevance_score: RawScore
    normalized_relevance_score: UnitScore | None = None
    query_variant: NonEmptyStr


class TemporalRegion(StrictModel):
    id: NonEmptyStr
    session_id: NonEmptyStr
    event_id: NonEmptyStr
    video_id: NonEmptyStr
    start_seconds: Seconds
    end_seconds: Seconds
    candidate_ids: tuple[NonEmptyStr, ...]
    raw_coarse_score: RawScore
    normalized_coarse_score: UnitScore | None = None
    refinement_status: RefinementStatus = "pending"
    user_status: RegionUserStatus = "active"

    @model_validator(mode="after")
    def validate_region(self) -> "TemporalRegion":
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be >= start_seconds")
        if not self.candidate_ids:
            raise ValueError("candidate_ids must not be empty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        return self


class FrameScoreSample(StrictModel):
    session_id: NonEmptyStr
    event_id: NonEmptyStr
    video_id: NonEmptyStr
    region_id: NonEmptyStr
    frame_id: NonNegativeInt
    timestamp_seconds: Seconds

    raw_anchor_score: RawScore
    raw_pre_score: RawScore
    raw_post_score: RawScore
    raw_motion_score: RawScore = 0.0

    normalized_anchor_score: UnitScore | None = None
    normalized_pre_score: UnitScore | None = None
    normalized_post_score: UnitScore | None = None
    normalized_motion_score: UnitScore | None = None


class EventProposal(StrictModel):
    id: NonEmptyStr
    session_id: NonEmptyStr
    event_id: NonEmptyStr
    video_id: NonEmptyStr
    region_id: NonEmptyStr
    timestamp_seconds: Seconds
    frame_id: NonNegativeInt

    raw_semantic_score: RawScore
    normalized_semantic_score: UnitScore
    raw_boundary_score: RawScore
    normalized_boundary_score: UnitScore
    raw_motion_score: RawScore = 0.0
    normalized_motion_score: UnitScore = 0.5
    pre_consistency_score: UnitScore
    post_persistence_score: UnitScore
    final_event_score: UnitScore

    left_window_seconds: PositiveFloat | None = None
    right_window_seconds: PositiveFloat | None = None
    source: ProposalSource = "dense"
    user_status: ProposalUserStatus = "active"


class EventConstraint(StrictModel):
    fixed_video_id: NonEmptyStr | None = None
    fixed_region_id: NonEmptyStr | None = None
    fixed_frame_id: NonNegativeInt | None = None
    fixed_timestamp_seconds: Seconds | None = None
    rejected_region_ids: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    rejected_proposal_ids: frozenset[NonEmptyStr] = Field(default_factory=frozenset)

    @field_validator("rejected_region_ids", "rejected_proposal_ids", mode="before")
    @classmethod
    def accept_json_rejection_arrays(cls, value):
        if isinstance(value, list):
            return frozenset(value)
        return value

    @model_validator(mode="after")
    def validate_fixed_region(self) -> "EventConstraint":
        # A fixed frame deliberately outranks a stale fixed-region selection.
        if (
            self.fixed_frame_id is None
            and self.fixed_region_id is not None
            and self.fixed_region_id in self.rejected_region_ids
        ):
            raise ValueError("fixed_region_id cannot also be rejected")
        return self


class AdjacentGapConstraint(StrictModel):
    before_event_id: NonEmptyStr
    after_event_id: NonEmptyStr
    min_gap_seconds: Seconds = 0.0
    max_gap_seconds: Seconds | None = None
    hinge_tau_seconds: Seconds | None = None
    gap_lambda: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_gap(self) -> "AdjacentGapConstraint":
        if self.before_event_id == self.after_event_id:
            raise ValueError("adjacent gap must reference two different events")
        if (
            self.max_gap_seconds is not None
            and self.max_gap_seconds < self.min_gap_seconds
        ):
            raise ValueError("max_gap_seconds must be >= min_gap_seconds")
        return self


class SearchConstraints(StrictModel):
    allowed_video_ids: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    rejected_video_ids: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    event_constraints: dict[NonEmptyStr, EventConstraint] = Field(
        default_factory=dict
    )
    adjacent_gap_constraints: tuple[AdjacentGapConstraint, ...] = ()
    max_tuple_span_seconds: Seconds | None = None

    @field_validator("allowed_video_ids", "rejected_video_ids", mode="before")
    @classmethod
    def accept_json_video_arrays(cls, value):
        if isinstance(value, list):
            return frozenset(value)
        return value

    @field_validator("adjacent_gap_constraints", mode="before")
    @classmethod
    def accept_json_gap_array(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_constraints(self) -> "SearchConstraints":
        overlap = self.allowed_video_ids & self.rejected_video_ids
        if overlap:
            raise ValueError("a video cannot be both allowed and rejected")

        gap_pairs: set[tuple[str, str]] = set()
        for gap in self.adjacent_gap_constraints:
            pair = (gap.before_event_id, gap.after_event_id)
            if pair in gap_pairs:
                raise ValueError("adjacent gap constraints must have unique pairs")
            gap_pairs.add(pair)

        for event_constraint in self.event_constraints.values():
            if event_constraint.fixed_video_id in self.rejected_video_ids:
                raise ValueError("a fixed video cannot also be rejected")
        return self


class VideoPriority(StrictModel):
    video_id: NonEmptyStr
    event_coverage: NonNegativeInt
    normalized_coverage: UnitScore
    mean_best_event_score: UnitScore
    min_best_event_score: UnitScore
    distinctness: UnitScore
    priority_score: UnitScore


