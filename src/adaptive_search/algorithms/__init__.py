"""Deterministic, model-free algorithms for adaptive temporal search.

Split by concern: scoring.py (robust normalization, the population-relative
score maps everything else builds on), constraints.py (Stage 6 filtering,
shared with tuple_ranking.py), regions.py (temporal region formation),
ranking.py (independent per-event video ranking). This module re-exports
everything so `from adaptive_search.algorithms import X` (or `from
.algorithms import X` within the package) keeps working unchanged - existing
callers and tests reference the flat module path, some down to private
helpers like `_region_allowed`.
"""

from __future__ import annotations

from .constraints import _event_constraint, _region_allowed
from .ranking import (
    _min_pairwise_gap_seconds,
    distinctness_from_timestamps,
    prioritize_videos,
)
from .regions import atomic_regions
from .scoring import (
    _candidate_normalized_scores,
    _candidate_normalized_scores_by_event,
    _region_score_map,
    _sigmoid,
    adjacent_hinge_penalty,
    robust_sigmoid,
)

__all__ = [
    "adjacent_hinge_penalty",
    "atomic_regions",
    "distinctness_from_timestamps",
    "prioritize_videos",
    "robust_sigmoid",
]
