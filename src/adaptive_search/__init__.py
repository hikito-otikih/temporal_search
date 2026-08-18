"""Public API for the model-free adaptive temporal-search core."""

from .algorithms import (
    generate_boundary_proposals,
    pairwise_softmax,
    prioritize_videos,
    robust_sigmoid,
    temporal_nms,
)
from .proposal_profiles import generate_profiled_proposals
from .retrieval import fuse_candidates_rrf
from .schemas import (
    AdjacentGapConstraint,
    BoundaryHyperparameters,
    EventConstraint,
    EventDefinition,
    EventProposal,
    FrameScoreSample,
    ModelRuntimeSpec,
    RefinementHyperparameters,
    RetrievalHyperparameters,
    SearchConstraints,
    SearchHyperparameters,
    SparseCandidate,
    TemporalRegion,
    VideoPriority,
)

__all__ = [
    "AdjacentGapConstraint",
    "BoundaryHyperparameters",
    "EventConstraint",
    "EventDefinition",
    "EventProposal",
    "FrameScoreSample",
    "ModelRuntimeSpec",
    "RefinementHyperparameters",
    "RetrievalHyperparameters",
    "SearchConstraints",
    "SearchHyperparameters",
    "SparseCandidate",
    "TemporalRegion",
    "VideoPriority",
    "generate_boundary_proposals",
    "generate_profiled_proposals",
    "fuse_candidates_rrf",
    "pairwise_softmax",
    "prioritize_videos",
    "robust_sigmoid",
    "temporal_nms",
]
