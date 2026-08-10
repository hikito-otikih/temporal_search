from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]


class ClusteredCandidate(BaseModel):
    frame_index: int
    timestamp: str
    score: float
    query_id: Optional[int] = None
    satisfiedObjects: Optional[bool] = None


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
