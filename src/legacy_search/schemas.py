from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]


BoundaryRefinementStatus = Literal[
    "applied",
    "skipped_runtime_unavailable",
    "skipped_no_metadata",
    "skipped_video_not_in_catalog",
    "not_requested",
]


class ClusteredCandidate(BaseModel):
    frame_index: int
    timestamp: str
    score: float
    query_id: Optional[int] = None
    satisfiedObjects: Optional[bool] = None
    refined_timestamp_seconds: Optional[float] = None
    boundary_refinement_status: Optional[BoundaryRefinementStatus] = None


class Videos(BaseModel):
    video_name: str
    results: list[ClusteredCandidate]


class Candidate(ClusteredCandidate):
    video_name: str
    video_title: str
    author: str
    watch_url: str


class QueryResponse(BaseModel):
    query: str
    results: list[Candidate]


class TemporalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: list[NonEmptyQuery] = Field(min_length=1, max_length=32)
    top_k_tuple: int = Field(default=100, ge=1, le=10_000)
    top_k_each_query: int = Field(default=100, ge=1, le=10_000)
    gamma: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)
    searcher_type: Literal["TemporalSearcher", "AmbiguousSearcher"] = (
        "TemporalSearcher"
    )
    objectFilterMode: bool = False
    object_name_list: Optional[list[str]] = None
    objectThreshold: float = Field(default=0.5, ge=0.0, le=1.0)
    apply_boundary_refinement: bool = False

    @model_validator(mode="after")
    def validate_object_filter(self) -> "TemporalSearchRequest":
        # check_object_satisfaction() builds required_object_names from
        # object_name_list, and an empty set is trivially a subset of
        # anything - with no labels, objectFilterMode=True doesn't filter by
        # content at all, it accidentally becomes a "does this video have
        # object-detection metadata at all" filter (satisfiedObjects only
        # goes False when the detection JSON is missing/unparseable).
        if self.objectFilterMode and not self.object_name_list:
            raise ValueError(
                "object_name_list must be non-empty when objectFilterMode is true"
            )
        return self


class BoundaryRefinementCapability(BaseModel):
    requested: bool
    available: bool
    reason: Optional[str] = None


class TemporalSearchResultItem(BaseModel):
    score: float
    video_name: str
    tuple: list[ClusteredCandidate]


class TemporalSearchResponse(BaseModel):
    query: list[str]
    results: list[TemporalSearchResultItem]
    boundary_refinement_capability: BoundaryRefinementCapability
    search_truncated: bool
