"""Deterministic, model-free algorithms for adaptive temporal search.

Embedding and video decoding intentionally live outside this package.  Every
function here consumes validated domain objects, has deterministic tie
breaks, and can be tested without a GPU, a model download, or network
access.

Split by concern: scoring.py (robust normalization, the population-relative
score maps everything else builds on), constraints.py (Stage 6 filtering,
shared with tuple_ranking.py), regions.py (temporal region formation),
proposals.py (boundary proposal generation + temporal NMS), ranking.py
(independent per-event video ranking). This module re-exports everything so
`from adaptive_search.
algorithms import X` (or `from .algorithms import X` within the package)
keeps working unchanged - existing callers and tests reference the flat
module path, some down to private helpers like `_region_allowed` and
`_stable_id`.
"""

from __future__ import annotations

from .constraints import _event_constraint, _region_allowed
from .proposals import (
    _proposal_output_key,
    _proposal_rank_key,
    generate_boundary_proposals,
    temporal_nms,
)
from .ranking import (
    _min_pairwise_gap_seconds,
    distinctness_from_timestamps,
    prioritize_videos,
)
from .regions import atomic_regions
from .scoring import (
    _candidate_normalized_scores,
    _candidate_normalized_scores_by_event,
    _mean,
    _region_score_map,
    _sigmoid,
    _stable_id,
    adjacent_hinge_penalty,
    calibrate_frame_scores,
    pairwise_softmax,
    robust_sigmoid,
)

__all__ = [
    "adjacent_hinge_penalty",
    "atomic_regions",
    "calibrate_frame_scores",
    "distinctness_from_timestamps",
    "generate_boundary_proposals",
    "pairwise_softmax",
    "prioritize_videos",
    "robust_sigmoid",
    "temporal_nms",
]
