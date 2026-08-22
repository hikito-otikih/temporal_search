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


class VideoPriorityHyperparameters(StrictModel):
    """Video-priority blend weights. Dense/medium sampling-interval fields
    were removed with `commands/refine` (the dense orchestrator that was
    their only consumer). The refinement-frontier budget fields
    (max_initial_videos, max_regions_per_event_per_video, max_total_regions,
    exploration_region_ratio) were removed alongside `/artifacts/frame-scores`
    and `select_refinement_frontier` - all three existed only to support an
    external client precomputing frame-level refinement scores and
    submitting them back into a session, a workflow this product does not
    support. See `docs/ADAPTIVE_PIPELINE_MIGRATION.md`."""

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
    def validate_video_priority_configuration(self) -> "VideoPriorityHyperparameters":
        video_weight = (
            self.video_coverage_weight
            + self.video_mean_weight
            + self.video_min_weight
            + self.video_distinctness_weight
        )
        if video_weight <= 0.0:
            raise ValueError("at least one video-priority weight must be positive")
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

    order_weight: NonNegativeFloat = 1.2
    """Weight of the (confidence-gated) order-agreement term added to a
    tuple's mean region score. Raised from the benchmark-chosen 0.8 (see
    class docstring) as a deliberate manual change, not a re-validated
    number: real contest use showed a noticeably-higher-relevance but
    wrongly-ordered tuple could still occasionally outrank a correctly-
    ordered one at 0.8. At 1.2 the term's swing (+/-1.2) exceeds the
    typical region-relevance range ([0,1]), so order-correctness dominates
    almost every time once confidence_gate passes its threshold. Re-run the
    YouCook2 benchmark to re-validate before assuming this transfers."""

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

    default_adjacent_gap_tau_seconds: Seconds = 25.0
    """Free zone (in seconds) for the automatic, default-on adjacent-gap
    penalty applied to every positionally-adjacent event pair a caller
    hasn't explicitly named in SearchConstraints.adjacent_gap_constraints -
    that opt-in mechanism (including its own hard min/max bounds) always
    takes full precedence for any pair it does name; this default only
    fills in the gaps. Below this many seconds between two consecutive
    events' timestamps costs nothing. Sanity-checked against three real
    4-event contest answers (L24_V018, L26_V072, L26_V194); across the
    corpus's ~25-30fps range their 9 consecutive-event frame gaps convert to
    ~2.2-25.3s, worst case (25.3s) at the low (25fps) end of that range -
    this tau comfortably covers it, so a genuinely tight, correctly-clustered
    answer like those costs ~0 penalty regardless of a given video's exact
    fps within the known range. Re-check if real gaps start running higher
    than this on a wider sample; set to a very large value to effectively
    disable."""

    default_adjacent_gap_lambda: NonNegativeFloat = 0.05
    """Slope (score lost per second) of the default-on adjacent-gap penalty
    above default_adjacent_gap_tau_seconds, reusing
    algorithms.scoring.adjacent_hinge_penalty() exactly as the opt-in
    SearchConstraints.adjacent_gap_constraints mechanism does. Unlike that
    mechanism this is soft-only (never rejects a combination) and applies
    automatically with no per-session configuration needed - rewards tight,
    coherent event sequences (like a real TRAKE answer) over tuples whose
    covered events are scattered far apart in the video. Set to 0.0 to
    fully restore pre-this-field behavior (no default gap penalty at all)."""


class SearchHyperparameters(StrictModel):
    tuple_ranking: TupleRankingHyperparameters = Field(
        default_factory=TupleRankingHyperparameters
    )


class EventDefinition(StrictModel):
    event_id: NonEmptyStr
    anchor_query: NonEmptyStr
    boundary_type: BoundaryType = "unknown"
    temporal_relation: TemporalRelationType = "unknown"
    reference_event_id: NonEmptyStr | None = None

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
    # A video-level property (same for every candidate/region of a given
    # video_id), carried per-candidate since that's the only place upstream
    # metadata enters the pipeline - see video_priorities.py, which looks
    # this up per winning video_id from the session's candidates.
    watch_url: NonEmptyStr | None = None


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


class EventConstraint(StrictModel):
    fixed_video_id: NonEmptyStr | None = None
    fixed_region_id: NonEmptyStr | None = None
    fixed_frame_id: NonNegativeInt | None = None
    fixed_timestamp_seconds: Seconds | None = None
    rejected_region_ids: frozenset[NonEmptyStr] = Field(default_factory=frozenset)

    @field_validator("rejected_region_ids", mode="before")
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
        # fixed_video_id is the pin's scope: _region_allowed only narrows a
        # video's own region choice once it knows *which* video that is
        # (fix_frame() always stamps both together). A standalone
        # fixed_region_id/fixed_frame_id/fixed_timestamp_seconds with no
        # fixed_video_id is silently inert rather than a real independent
        # pin - reject it instead of accepting a constraint with no effect.
        if self.fixed_video_id is None and (
            self.fixed_region_id is not None
            or self.fixed_frame_id is not None
            or self.fixed_timestamp_seconds is not None
        ):
            raise ValueError(
                "fixed_region_id/fixed_frame_id/fixed_timestamp_seconds require fixed_video_id"
            )
        # The reverse gap: fixed_video_id alone (or with only fixed_frame_id,
        # no timestamp) is accepted but inert. _region_allowed only narrows
        # a video's region choice via fixed_timestamp_seconds (the exact-pin
        # window) or via fixed_region_id with fixed_frame_id absent (a
        # genuine "pin this existing region" reference) - fixed_frame_id's
        # own value is never read anywhere in ranking, only its nullness as
        # a marker alongside those two, so fixed_video_id + fixed_frame_id
        # with no fixed_timestamp_seconds narrows nothing either. fix_frame()
        # (the only real caller) always sets fixed_timestamp_seconds
        # alongside fixed_video_id, so this only rejects raw PUT payloads
        # that could never have come from it.
        if (
            self.fixed_video_id is not None
            and self.fixed_timestamp_seconds is None
            and self.fixed_region_id is None
        ):
            raise ValueError(
                "fixed_video_id alone has no effect - set fixed_timestamp_seconds "
                "(an exact-frame pin) or fixed_region_id (a region pin)"
            )
        # fixed_frame_id + fixed_region_id without fixed_timestamp_seconds:
        # accepted but inert. _region_allowed's region-id-only narrowing
        # ("a genuine 'pin this existing region' constraint") only applies
        # when fixed_frame_id is None; its timestamp-window narrowing only
        # applies when fixed_timestamp_seconds is set. With fixed_frame_id
        # set and no timestamp, neither condition in _region_allowed,
        # _effective_region_timestamp, or _synthetic_fixed_regions ever
        # fires, so this pin silently filters nothing - a 200 that does
        # exactly what an unset constraint would have done.
        if (
            self.fixed_video_id is not None
            and self.fixed_frame_id is not None
            and self.fixed_region_id is not None
            and self.fixed_timestamp_seconds is None
        ):
            raise ValueError(
                "fixed_frame_id + fixed_region_id without fixed_timestamp_seconds "
                "has no effect - set fixed_timestamp_seconds too, or drop "
                "fixed_frame_id to make this a plain region pin"
            )
        # fixed_region_id + fixed_timestamp_seconds without fixed_frame_id:
        # accepted, but two different consumers of this same constraint
        # disagree about what it means. _region_allowed/
        # _effective_region_timestamp treat fixed_frame_id being absent
        # alongside fixed_region_id as exactly the "genuine region pin"
        # carve-out - they narrow to that region and report its own best
        # candidate's timestamp, never reading fixed_timestamp_seconds at
        # all. video_priorities.py's boundary-refinement skip-path checks
        # only fixed_video_id + fixed_timestamp_seconds, with no awareness
        # of fixed_region_id - it reports fixed_timestamp_seconds as the
        # event's refined_seconds regardless. The API could then report a
        # refined_seconds that does not match the timestamp the winning
        # tuple actually used for that event.
        if (
            self.fixed_video_id is not None
            and self.fixed_region_id is not None
            and self.fixed_timestamp_seconds is not None
            and self.fixed_frame_id is None
        ):
            raise ValueError(
                "fixed_region_id + fixed_timestamp_seconds without "
                "fixed_frame_id is ambiguous - set fixed_frame_id too (an "
                "exact-frame pin, timestamp wins), or drop "
                "fixed_timestamp_seconds to make this a plain region pin"
            )
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
    # Ordered, first = highest priority - distinct from allowed/rejected
    # (which only control set membership in the ranked list): this reorders
    # GET .../video-priorities' output directly, floating these video_ids to
    # the front (in this order) ahead of every video ranked purely by score.
    # Deliberately does not change any video's own priority_score or a
    # fixed event's per-video region choice (see fix_frame) - fixing a
    # frame only affects ranking *within* its own video, never the ordering
    # of videos against each other; this field is the explicit, separate
    # mechanism for "put this video first" a user actually asks for.
    prioritized_video_ids: tuple[NonEmptyStr, ...] = ()
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

    @field_validator("prioritized_video_ids", mode="before")
    @classmethod
    def accept_json_prioritized_array(cls, value):
        if isinstance(value, list):
            return tuple(value)
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

        if len(set(self.prioritized_video_ids)) != len(self.prioritized_video_ids):
            raise ValueError("prioritized_video_ids must not contain duplicates")
        prioritized_and_rejected = set(self.prioritized_video_ids) & self.rejected_video_ids
        if prioritized_and_rejected:
            raise ValueError("a video cannot be both prioritized and rejected")

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
    # Order-checked pairs, drawn from `TupleRankingResult.order_score` -
    # [-1, 1], not a UnitScore: an uncovered event drops any constraint
    # touching it (see tuple_ranking._order_score), so this reflects only
    # the pairs that were actually checkable for this video's winning tuple,
    # not an absolute measure of how "in order" the full event set is.
    order_score: RawScore
    # Pre-normalization tuple score (region_mean_score + order_weight *
    # order_score - gap_penalty) - unbounded, not a UnitScore. Exists so
    # ties in priority_score (possible when min_max_normalize() has a zero-
    # spread population, or on a genuine raw_score tie - see
    # min_max_normalize()'s own docstring) can still be broken by a caller,
    # and so the order/coverage tradeoff baked into priority_score is
    # inspectable instead of opaque.
    raw_score: RawScore
    priority_score: UnitScore
    # From the winning tuple's candidates (SparseCandidate.watch_url) -
    # None when the session's candidates never carried it (e.g. an upstream
    # response that didn't include watch_url, or a synthetic fixed region
    # with no backing candidate).
    watch_url: NonEmptyStr | None = None
    # One entry per session event, same order as the session's events - the
    # winning tuple's frame_id for that event, "?" where the event is
    # uncovered or its frame can't be resolved. A commands/fix-frame pin for
    # (event, this video) always wins here, live - not a snapshot from when
    # the tuple was ranked, so this list changes on the next call after a fix.
    matched_frame_ids: tuple[NonEmptyStr, ...] = ()
    # Parallel to matched_frame_ids (same length, same event order, same
    # "uncovered" case) - the winning tuple's own timestamp_seconds for that
    # event, `None` where matched_frame_ids reports "?", or in the rare
    # accepted-but-inert case of a fix-frame constraint that pins
    # fixed_frame_id without fixed_timestamp_seconds (see
    # EventConstraint.validate_fixed_region's own comments). Saves a caller
    # from re-deriving seconds from frame_id + fps itself.
    matched_timestamps: tuple[Seconds | None, ...] = ()


