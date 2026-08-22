from __future__ import annotations

import unittest
from statistics import fmean

from adaptive_search.algorithms import prioritize_videos
from adaptive_search.schemas import (
    AdjacentGapConstraint,
    EventConstraint,
    EventDefinition,
    SearchConstraints,
    SparseCandidate,
    TemporalRegion,
    TupleRankingHyperparameters,
    VideoPriorityHyperparameters,
)
from adaptive_search.tuple_ranking import (
    _effective_order_weight,
    _order_score,
    _region_margin,
    assemble_region_tuples_for_video,
    build_order_constraints,
    pool_event_regions,
    rank_videos_by_region_tuples,
)


def _region(region_id, *, event_id, video_id, candidate_ids, normalized_coarse_score,
            start_seconds=0.0, end_seconds=6.0, raw_coarse_score=None):
    return TemporalRegion(
        id=region_id,
        session_id="s1",
        event_id=event_id,
        video_id=video_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        candidate_ids=tuple(candidate_ids),
        raw_coarse_score=raw_coarse_score if raw_coarse_score is not None else normalized_coarse_score,
        normalized_coarse_score=normalized_coarse_score,
    )


def _candidate(candidate_id, *, event_id, video_id, timestamp_seconds, raw_relevance_score):
    return SparseCandidate(
        id=candidate_id,
        session_id="s1",
        event_id=event_id,
        video_id=video_id,
        frame_id=int(timestamp_seconds * 30),
        timestamp_seconds=timestamp_seconds,
        raw_relevance_score=raw_relevance_score,
    )


class PoolEventRegionsTests(unittest.TestCase):
    def test_relative_delta_excludes_far_below_best(self) -> None:
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.90),
            _region("r2", event_id="e1", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.60),
        ]
        params = TupleRankingHyperparameters(relative_delta=0.15, max_regions_per_event=20)
        pooled = pool_event_regions(regions, event_id="e1", video_id="v1", params=params)
        self.assertEqual([r.id for r in pooled], ["r1"])

    def test_absolute_threshold_excludes_even_if_within_delta(self) -> None:
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.20),
            _region("r2", event_id="e1", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.15),
        ]
        params = TupleRankingHyperparameters(absolute_threshold=0.18, relative_delta=0.5, max_regions_per_event=20)
        pooled = pool_event_regions(regions, event_id="e1", video_id="v1", params=params)
        self.assertEqual([r.id for r in pooled], ["r1"])

    def test_max_regions_per_event_caps_even_within_delta(self) -> None:
        regions = [
            _region(f"r{i}", event_id="e1", video_id="v1", candidate_ids=[f"c{i}"], normalized_coarse_score=1.0 - i * 0.01)
            for i in range(10)
        ]
        params = TupleRankingHyperparameters(relative_delta=1.0, max_regions_per_event=3)
        pooled = pool_event_regions(regions, event_id="e1", video_id="v1", params=params)
        self.assertEqual(len(pooled), 3)
        self.assertEqual([r.id for r in pooled], ["r0", "r1", "r2"])

    def test_no_matching_regions_returns_empty(self) -> None:
        params = TupleRankingHyperparameters()
        self.assertEqual(pool_event_regions([], event_id="e1", video_id="v1", params=params), [])


class OrderScoreTests(unittest.TestCase):
    def test_single_event_is_neutral(self) -> None:
        self.assertEqual(_order_score([5.0]), 0.0)

    def test_perfectly_increasing_is_max(self) -> None:
        self.assertEqual(_order_score([1.0, 2.0, 3.0, 4.0]), 1.0)

    def test_perfectly_decreasing_is_min(self) -> None:
        self.assertEqual(_order_score([4.0, 3.0, 2.0, 1.0]), -1.0)

    def test_mixed_order_is_between(self) -> None:
        # pairs: (1,3) correct, (3,2) violated, (2,4) correct -> (2*2-3)/3
        self.assertAlmostEqual(_order_score([1.0, 3.0, 2.0, 4.0]), 1.0 / 3.0)

    def test_tie_counts_as_violation(self) -> None:
        self.assertEqual(_order_score([1.0, 1.0]), -1.0)


class OrderConstraintsTests(unittest.TestCase):
    def test_none_defaults_to_adjacent_list_position_chain(self) -> None:
        self.assertEqual(_order_score([1.0, 3.0, 2.0]), _order_score([1.0, 3.0, 2.0], [(0, 1), (1, 2)]))

    def test_dropping_a_constraint_removes_it_from_scoring(self) -> None:
        self.assertEqual(_order_score([1.0, 3.0, 2.0], [(0, 1), (1, 2)]), 0.0)
        self.assertEqual(_order_score([1.0, 3.0, 2.0], [(0, 1)]), 1.0)

    def test_no_constraints_is_neutral(self) -> None:
        self.assertEqual(_order_score([3.0, 1.0, 2.0], []), 0.0)

    def test_constraint_direction_is_independent_of_list_position(self) -> None:
        # If temporal_relation says event 0 is *after* event 1 - i.e. the
        # expected constraint is predecessor=1, successor=0 - then timestamps
        # [1.0, 4.0, 3.0, 2.0] (event 0's timestamp is smaller than event 1's)
        # VIOLATE that constraint and must be penalized, regardless of the
        # events' positions in the array. A naive adjacent-list-position
        # check would instead score the (0,1) gap as +1 (1.0 < 4.0), which
        # is backwards relative to what the relation actually claims.
        timestamps = [1.0, 4.0, 3.0, 2.0]
        constraint_event0_after_event1 = [(1, 0)]
        self.assertEqual(_order_score(timestamps, constraint_event0_after_event1), -1.0)
        self.assertEqual(_order_score(timestamps, [(0, 1)]), 1.0)

    def test_dropped_constraint_removes_penalty_at_tuple_level(self) -> None:
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.8),
            _region("r2_early", event_id="e2", video_id="v1", candidate_ids=["c2e"], normalized_coarse_score=0.8),
            _region("r2_late", event_id="e2", video_id="v1", candidate_ids=["c2l"], normalized_coarse_score=0.8),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c2e": _candidate("c2e", event_id="e2", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
            "c2l": _candidate("c2l", event_id="e2", video_id="v1", timestamp_seconds=20.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.2)
        tuples = assemble_region_tuples_for_video(
            "v1", ["e1", "e2"], regions, candidates_by_id, params, order_constraints=[],
        )
        self.assertEqual(len(tuples), 2)
        self.assertAlmostEqual(tuples[0].score, tuples[1].score)


class AssembleRegionTuplesTests(unittest.TestCase):
    def test_all_events_uncovered_yields_no_tuples(self) -> None:
        params = TupleRankingHyperparameters()
        tuples = assemble_region_tuples_for_video("v1", ["e1", "e2"], [], {}, params)
        self.assertEqual(tuples, [])

    def test_partial_coverage_zero_fills_missing_event_like_prioritize_videos(self) -> None:
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.8),
        ]
        candidates_by_id = {"c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0)}
        params = TupleRankingHyperparameters()
        tuples = assemble_region_tuples_for_video("v1", ["e1", "e2"], regions, candidates_by_id, params)
        self.assertEqual(len(tuples), 1)
        winner = tuples[0]
        self.assertAlmostEqual(winner.region_mean_score, 0.8 / 2.0)
        self.assertEqual(winner.region_ids, ("r1", None))
        self.assertEqual(winner.timestamps, (10.0, None))
        self.assertEqual(winner.order_score, 0.0)

    def test_order_weight_zero_score_equals_mean_of_best_regions(self) -> None:
        regions = [
            _region("r1a", event_id="e1", video_id="v1", candidate_ids=["c1a"], normalized_coarse_score=0.9),
            _region("r1b", event_id="e1", video_id="v1", candidate_ids=["c1b"], normalized_coarse_score=0.8),
            _region("r2a", event_id="e2", video_id="v1", candidate_ids=["c2a"], normalized_coarse_score=0.7),
        ]
        candidates_by_id = {
            "c1a": _candidate("c1a", event_id="e1", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
            "c1b": _candidate("c1b", event_id="e1", video_id="v1", timestamp_seconds=50.0, raw_relevance_score=1.0),
            "c2a": _candidate("c2a", event_id="e2", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.0)
        tuples = assemble_region_tuples_for_video("v1", ["e1", "e2"], regions, candidates_by_id, params)
        self.assertEqual(len(tuples), 2)
        best = tuples[0]
        self.assertAlmostEqual(best.region_mean_score, (0.9 + 0.7) / 2.0)
        self.assertEqual(best.region_ids, ("r1a", "r2a"))

    def test_order_weight_prefers_chronologically_consistent_tuple_on_tie(self) -> None:
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.8),
            _region("r2_early", event_id="e2", video_id="v1", candidate_ids=["c2e"], normalized_coarse_score=0.8),
            _region("r2_late", event_id="e2", video_id="v1", candidate_ids=["c2l"], normalized_coarse_score=0.8),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c2e": _candidate("c2e", event_id="e2", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
            "c2l": _candidate("c2l", event_id="e2", video_id="v1", timestamp_seconds=20.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.2)
        tuples = assemble_region_tuples_for_video("v1", ["e1", "e2"], regions, candidates_by_id, params)
        self.assertEqual(tuples[0].region_ids, ("r1", "r2_late"))
        self.assertGreater(tuples[0].score, tuples[1].score)

    def test_combinations_capped_and_truncated_to_max_tuples(self) -> None:
        e1_regions = [
            _region(f"r1_{i}", event_id="e1", video_id="v1", candidate_ids=[f"c1_{i}"], normalized_coarse_score=0.9 - i * 0.001)
            for i in range(50)
        ]
        e2_regions = [
            _region(f"r2_{i}", event_id="e2", video_id="v1", candidate_ids=[f"c2_{i}"], normalized_coarse_score=0.9 - i * 0.001)
            for i in range(50)
        ]
        candidates_by_id = {}
        for i in range(50):
            candidates_by_id[f"c1_{i}"] = _candidate(f"c1_{i}", event_id="e1", video_id="v1", timestamp_seconds=1.0 + i, raw_relevance_score=1.0)
            candidates_by_id[f"c2_{i}"] = _candidate(f"c2_{i}", event_id="e2", video_id="v1", timestamp_seconds=100.0 + i, raw_relevance_score=1.0)
        params = TupleRankingHyperparameters(
            relative_delta=1.0, max_regions_per_event=50,
            max_combinations_per_video=200, max_tuples_per_video=5,
        )
        tuples = assemble_region_tuples_for_video("v1", ["e1", "e2"], e1_regions + e2_regions, candidates_by_id, params)
        self.assertLessEqual(len(tuples), 5)

    def test_low_scoring_but_order_satisfying_region_is_found_under_a_tight_cap(self) -> None:
        """Regression test for a real reported bug: naive lexicographic
        enumeration explores (r1_0, e2), (r1_1, e2), (r1_2, e2) in exactly
        that order (pools[1] has one member, nothing to interleave) - a
        leaf-count cap of 2 would find only the two order-violating,
        higher-raw-score leaves and never reach the third, correctly-
        ordered, truly-best one, silently returning score -0.30 as "the
        best" when the actual best is 1.55. Best-first search must still
        find the true best despite an equally tight combinations budget
        (10, well under the 3x1=3 leaves x several expansions this
        structure needs but far below what an exhaustive scan of a wider
        pool would require), because it explores by achievable score, not
        enumeration position."""

        e1_regions = [
            _region("r1_0", event_id="e1", video_id="v1", candidate_ids=["c1_0"], normalized_coarse_score=0.90, start_seconds=100.0, end_seconds=100.0),
            _region("r1_1", event_id="e1", video_id="v1", candidate_ids=["c1_1"], normalized_coarse_score=0.89, start_seconds=100.0, end_seconds=100.0),
            _region("r1_2", event_id="e1", video_id="v1", candidate_ids=["c1_2"], normalized_coarse_score=0.60, start_seconds=1.0, end_seconds=1.0),
        ]
        e2_region = _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.50, start_seconds=50.0, end_seconds=50.0)
        candidates_by_id = {
            "c1_0": _candidate("c1_0", event_id="e1", video_id="v1", timestamp_seconds=100.0, raw_relevance_score=1.0),
            "c1_1": _candidate("c1_1", event_id="e1", video_id="v1", timestamp_seconds=100.0, raw_relevance_score=1.0),
            "c1_2": _candidate("c1_2", event_id="e1", video_id="v1", timestamp_seconds=1.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=50.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(
            relative_delta=1.0, order_weight=1.0, confidence_gate="none",
            max_combinations_per_video=10, max_tuples_per_video=1,
            default_adjacent_gap_lambda=0.0,  # isolate this from the new default gap penalty - not what this regression test is about
        )
        tuples = assemble_region_tuples_for_video(
            "v1", ["e1", "e2"], e1_regions + [e2_region], candidates_by_id, params,
            order_constraints=[(0, 1)],
        )
        self.assertEqual(len(tuples), 1)
        self.assertEqual(tuples[0].region_ids, ("r1_2", "r2"))
        self.assertAlmostEqual(tuples[0].score, (0.60 + 0.50) / 2 + 1.0)

    def test_a_cap_below_event_count_plus_one_still_returns_a_result(self) -> None:
        """Regression test for a real reported bug: reaching even one
        complete result costs a minimum of len(event_ids)+1 heap pops (one
        per event to walk root-to-leaf, plus one more to pop the finalized
        leaf itself) - a caller-supplied max_combinations_per_video below
        that floor made it structurally impossible to ever return a single
        result, even with a region plainly available (reported: cap=1
        returned zero video priorities despite an existing region)."""

        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.8),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(max_combinations_per_video=1, max_tuples_per_video=1)
        tuples = assemble_region_tuples_for_video("v1", ["e1"], regions, candidates_by_id, params)
        self.assertEqual(len(tuples), 1)
        self.assertEqual(tuples[0].region_ids, ("r1",))

    def test_heavily_tied_pool_still_returns_results_under_a_modest_cap(self) -> None:
        """Regression test for a real reported bug: heapq's FIFO tie-break
        among equal-priority entries caused near-breadth-first expansion
        across every sibling at a shallow level before any path reached a
        complete leaf - reported as 400 valid tied leaves and cap 10
        returning zero tuples while the heap grew to 191 entries. Depth is
        now a secondary sort key, so the search commits to finishing one
        path before fanning out into the next."""

        e1_regions = [
            _region(f"r1_{i}", event_id="e1", video_id="v1", candidate_ids=[f"c1_{i}"], normalized_coarse_score=0.5)
            for i in range(20)
        ]
        e2_regions = [
            _region(f"r2_{i}", event_id="e2", video_id="v1", candidate_ids=[f"c2_{i}"], normalized_coarse_score=0.5)
            for i in range(20)
        ]
        candidates_by_id = {}
        for i in range(20):
            candidates_by_id[f"c1_{i}"] = _candidate(f"c1_{i}", event_id="e1", video_id="v1", timestamp_seconds=1.0 + i, raw_relevance_score=1.0)
            candidates_by_id[f"c2_{i}"] = _candidate(f"c2_{i}", event_id="e2", video_id="v1", timestamp_seconds=100.0 + i, raw_relevance_score=1.0)
        params = TupleRankingHyperparameters(
            relative_delta=1.0, max_regions_per_event=20,
            max_combinations_per_video=10, max_tuples_per_video=5,
            # This test's e1/e2 timestamps are ~80-118s apart by construction
            # (irrelevant to what it's actually testing - heap/tie-break
            # mechanics under a tight cap) - upper_bound() deliberately never
            # accounts for gap penalty (admissible-but-not-tight, see its own
            # comment), so a large default penalty here can make every
            # completed leaf's real score come in worse than still-optimistic
            # partials, and get trimmed out of the heap before ever being
            # popped - a real interaction, but with this test's artificial
            # cap=10 for ~400 possible leaves, not with production's actual
            # defaults (cap=10_000_000). Isolate this test from that.
            default_adjacent_gap_lambda=0.0,
        )
        tuples = assemble_region_tuples_for_video(
            "v1", ["e1", "e2"], e1_regions + e2_regions, candidates_by_id, params,
        )
        self.assertGreater(len(tuples), 0)

    def test_tight_cap_does_not_discard_the_correctly_ordered_winner(self) -> None:
        """Regression test for a real reported bug (Blocker 1): an earlier
        attempt at bounding heap memory truncated each expansion's children
        to the top few by raw coarse score, before order/gap scoring ran.
        That is unsound for the *last* event in a tuple, whose children go
        straight to finalize() - which computes the REAL order-agreement
        term from the actual chosen timestamp, not the optimistic bound. A
        lower-scoring, correctly-ordered region can (and, with order_weight
        defaulting to 0.8, easily does) beat a higher-scoring, wrong-order
        one once finalized. e2's pool below has four higher-scoring but
        wrong-order decoys ranked above the correct-order winner - with
        max_combinations_per_video=3 (this session's exact reported repro),
        a top-3-by-raw-score truncation would keep only the decoys and
        never even consider the real winner.
        """
        e1_region = _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.9)
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
        }
        e2_regions = []
        for name, score, seconds in [
            ("wrong1", 0.95, 1.0),
            ("wrong2", 0.90, 2.0),
            ("wrong3", 0.85, 3.0),
            ("wrong4", 0.80, 4.0),
            ("right", 0.30, 60.0),
        ]:
            e2_regions.append(
                _region(name, event_id="e2", video_id="v1", candidate_ids=[f"c_{name}"], normalized_coarse_score=score)
            )
            candidates_by_id[f"c_{name}"] = _candidate(
                f"c_{name}", event_id="e2", video_id="v1", timestamp_seconds=seconds, raw_relevance_score=1.0
            )
        params = TupleRankingHyperparameters(
            relative_delta=1.0, order_weight=0.8, confidence_gate="none",
            max_combinations_per_video=3, max_tuples_per_video=1,
            default_adjacent_gap_lambda=0.0,  # isolate this from the new default gap penalty - not what this regression test is about
        )
        tuples = assemble_region_tuples_for_video(
            "v1", ["e1", "e2"], [e1_region] + e2_regions, candidates_by_id, params,
        )
        self.assertEqual(len(tuples), 1)
        self.assertEqual(tuples[0].region_ids, ("r1", "right"))
        self.assertGreater(tuples[0].score, 1.0)

    def test_heap_trim_bounds_growth_across_many_expansions(self) -> None:
        """Regression test for a real reported bug: max_combinations_per_video
        only bounded heap *pops*, not pushes - without any trim, two large
        pools (one per event) could compound across expansions toward
        pool_size x pool_size entries. A single expansion must still be free
        to push every one of its own children (see the test above - that's
        what correctness requires), but the heap must not be allowed to
        accumulate anywhere near a full pool's worth even from one
        expansion's own fan-out: it must be trimmed after every individual
        push, not once per expansion. Regression test for a real reported
        bug: an earlier version trimmed only once per expansion, after the
        entire fan-out loop over one event's pool had already pushed every
        child - with cap=1 and a 5,000-region pool, the heap still reached
        5,000 entries before that trim ever ran once."""
        import heapq as heapq_module
        from unittest.mock import patch

        pool_size = 3000
        e1_regions = [
            _region(f"a{i}", event_id="e1", video_id="v1", candidate_ids=[f"ca{i}"], normalized_coarse_score=i / pool_size)
            for i in range(pool_size)
        ]
        e2_regions = [
            _region(f"b{i}", event_id="e2", video_id="v1", candidate_ids=[f"cb{i}"], normalized_coarse_score=i / pool_size)
            for i in range(pool_size)
        ]
        candidates_by_id = {}
        for i in range(pool_size):
            candidates_by_id[f"ca{i}"] = _candidate(f"ca{i}", event_id="e1", video_id="v1", timestamp_seconds=float(i), raw_relevance_score=1.0)
            candidates_by_id[f"cb{i}"] = _candidate(f"cb{i}", event_id="e2", video_id="v1", timestamp_seconds=1000.0 + i, raw_relevance_score=1.0)
        params = TupleRankingHyperparameters(
            relative_delta=1.0, max_combinations_per_video=5, max_tuples_per_video=1,
            max_regions_per_event=pool_size,
            # See the same note in test_heavily_tied_pool_still_returns_
            # results_under_a_modest_cap: this test's ~1000s e1/e2 gap is
            # irrelevant to what it tests (heap growth bounding), and a
            # large default gap penalty under this artificial cap=5 can
            # starve every completed leaf out of the heap before it's ever
            # popped - isolate this test from that.
            default_adjacent_gap_lambda=0.0,
        )

        peak_heap_size = 0
        real_heappush = heapq_module.heappush

        def _tracking_heappush(heap, item):
            nonlocal peak_heap_size
            real_heappush(heap, item)
            peak_heap_size = max(peak_heap_size, len(heap))

        with patch("heapq.heappush", side_effect=_tracking_heappush):
            tuples = assemble_region_tuples_for_video(
                "v1", ["e1", "e2"], e1_regions + e2_regions, candidates_by_id, params,
            )

        self.assertEqual(len(tuples), 1)
        # Bounded by the cap itself (n=2 events -> floor of 3, so
        # effective_max_combinations = max(5, 3) = 5), not by pool_size or
        # pool_size * pool_size - only +1 slack for the one push that
        # transiently exceeds the cap before push()'s own trim clips it back
        # down, which is exactly when _tracking_heappush's wrapper samples
        # len(heap) (right after heapq.heappush, before push()'s trim runs).
        effective_max_combinations = max(params.max_combinations_per_video, len(["e1", "e2"]) + 1)
        self.assertLessEqual(peak_heap_size, effective_max_combinations + 1)

    def test_heap_trim_bounds_a_single_expansions_own_fan_out(self) -> None:
        """Regression test for the exact reported repro: cap=1 (floored to
        2 - the minimum for one event to ever produce a result) against a
        single 5,000-region pool. The old trim sat after the whole fan-out
        loop, so the very first expansion alone could push all 5,000
        children before the trim ever ran once. Trimming per-push instead
        must keep the heap close to the cap even mid-fan-out, not just
        between expansions."""
        import heapq as heapq_module
        from unittest.mock import patch

        pool_size = 5000
        e1_regions = [
            _region(f"a{i}", event_id="e1", video_id="v1", candidate_ids=[f"ca{i}"], normalized_coarse_score=i / pool_size)
            for i in range(pool_size)
        ]
        candidates_by_id = {
            f"ca{i}": _candidate(f"ca{i}", event_id="e1", video_id="v1", timestamp_seconds=float(i), raw_relevance_score=1.0)
            for i in range(pool_size)
        }
        params = TupleRankingHyperparameters(
            relative_delta=1.0, max_combinations_per_video=1, max_tuples_per_video=1,
            max_regions_per_event=pool_size,
        )

        peak_heap_size = 0
        real_heappush = heapq_module.heappush

        def _tracking_heappush(heap, item):
            nonlocal peak_heap_size
            real_heappush(heap, item)
            peak_heap_size = max(peak_heap_size, len(heap))

        with patch("heapq.heappush", side_effect=_tracking_heappush):
            tuples = assemble_region_tuples_for_video(
                "v1", ["e1"], e1_regions, candidates_by_id, params,
            )

        self.assertEqual(len(tuples), 1)
        effective_max_combinations = max(params.max_combinations_per_video, len(["e1"]) + 1)
        self.assertLessEqual(peak_heap_size, effective_max_combinations + 1)

    def test_adjacent_gap_constraint_hard_rejects_violating_combination(self) -> None:
        regions = [
            _region("r1_close", event_id="e1", video_id="v1", candidate_ids=["c1c"], normalized_coarse_score=0.9, start_seconds=10.0, end_seconds=10.0),
            _region("r1_far", event_id="e1", video_id="v1", candidate_ids=["c1f"], normalized_coarse_score=0.5, start_seconds=1.0, end_seconds=1.0),
            _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.9, start_seconds=12.0, end_seconds=12.0),
        ]
        candidates_by_id = {
            "c1c": _candidate("c1c", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c1f": _candidate("c1f", event_id="e1", video_id="v1", timestamp_seconds=1.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=12.0, raw_relevance_score=1.0),
        }
        constraints = SearchConstraints(
            adjacent_gap_constraints=(
                AdjacentGapConstraint(before_event_id="e1", after_event_id="e2", min_gap_seconds=5.0),
            )
        )
        params = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.0)
        tuples = assemble_region_tuples_for_video(
            "v1", ["e1", "e2"], regions, candidates_by_id, params,
            order_constraints=None, constraints=constraints,
        )
        # (r1_close=10.0, r2=12.0): gap=2.0 < min_gap_seconds=5.0 -> rejected.
        # (r1_far=1.0, r2=12.0): gap=11.0 >= 5.0 -> the only survivor, despite
        # r1_close scoring higher on its own.
        self.assertEqual(len(tuples), 1)
        self.assertEqual(tuples[0].region_ids, ("r1_far", "r2"))

    def test_max_tuple_span_seconds_hard_rejects_wide_combination(self) -> None:
        regions = [
            _region("r1_close", event_id="e1", video_id="v1", candidate_ids=["c1c"], normalized_coarse_score=0.9, start_seconds=10.0, end_seconds=10.0),
            _region("r1_far", event_id="e1", video_id="v1", candidate_ids=["c1f"], normalized_coarse_score=0.5, start_seconds=1.0, end_seconds=1.0),
            _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.9, start_seconds=11.0, end_seconds=11.0),
        ]
        candidates_by_id = {
            "c1c": _candidate("c1c", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c1f": _candidate("c1f", event_id="e1", video_id="v1", timestamp_seconds=1.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=11.0, raw_relevance_score=1.0),
        }
        constraints = SearchConstraints(max_tuple_span_seconds=5.0)
        params = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.0)
        tuples = assemble_region_tuples_for_video(
            "v1", ["e1", "e2"], regions, candidates_by_id, params,
            order_constraints=None, constraints=constraints,
        )
        # (r1_close=10.0, r2=11.0): span=1.0 <= 5.0 -> allowed.
        # (r1_far=1.0, r2=11.0): span=10.0 > 5.0 -> rejected, despite r1_far
        # being a legal (delta-surviving) pool member on its own.
        self.assertEqual(len(tuples), 1)
        self.assertEqual(tuples[0].region_ids, ("r1_close", "r2"))


class DefaultAdjacentGapPenaltyTests(unittest.TestCase):
    def test_tight_gap_outranks_scattered_gap_with_no_constraints_configured(self) -> None:
        # Same region_mean_score and order_score either way (both events
        # cover, both timestamps increasing) - only the gap between them
        # differs. No SearchConstraints at all: this is purely the new
        # default-on penalty at work.
        regions = [
            _region("e1_tight", event_id="e1", video_id="tight", candidate_ids=["c1t"], normalized_coarse_score=0.8),
            _region("e2_tight", event_id="e2", video_id="tight", candidate_ids=["c2t"], normalized_coarse_score=0.8),
            _region("e1_wide", event_id="e1", video_id="wide", candidate_ids=["c1w"], normalized_coarse_score=0.8),
            _region("e2_wide", event_id="e2", video_id="wide", candidate_ids=["c2w"], normalized_coarse_score=0.8),
        ]
        candidates_by_id = {
            "c1t": _candidate("c1t", event_id="e1", video_id="tight", timestamp_seconds=100.0, raw_relevance_score=1.0),
            "c2t": _candidate("c2t", event_id="e2", video_id="tight", timestamp_seconds=110.0, raw_relevance_score=1.0),
            "c1w": _candidate("c1w", event_id="e1", video_id="wide", timestamp_seconds=100.0, raw_relevance_score=1.0),
            "c2w": _candidate("c2w", event_id="e2", video_id="wide", timestamp_seconds=400.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(relative_delta=1.0)
        ranking, _ = rank_videos_by_region_tuples(regions, candidates_by_id, ["e1", "e2"], params)
        self.assertEqual([video_id for video_id, _ in ranking], ["tight", "wide"])

    def test_explicit_constraint_takes_precedence_over_the_default_for_that_pair(self) -> None:
        # (e1,e2) is explicitly configured (its own, different gap_lambda) -
        # the default must not also apply to it. (e2,e3) is unconfigured -
        # it must still get the new default penalty.
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.8, start_seconds=0.0, end_seconds=0.0),
            _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.8, start_seconds=100.0, end_seconds=100.0),
            _region("r3", event_id="e3", video_id="v1", candidate_ids=["c3"], normalized_coarse_score=0.8, start_seconds=200.0, end_seconds=200.0),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=0.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=100.0, raw_relevance_score=1.0),
            "c3": _candidate("c3", event_id="e3", video_id="v1", timestamp_seconds=200.0, raw_relevance_score=1.0),
        }
        # Explicit (e1,e2): huge tau (200s) -> its own 100s gap costs nothing,
        # overriding what the default (tau=20) would have charged it.
        constraints = SearchConstraints(
            adjacent_gap_constraints=(
                AdjacentGapConstraint(before_event_id="e1", after_event_id="e2", hinge_tau_seconds=200.0, gap_lambda=1.0),
            )
        )
        params = TupleRankingHyperparameters(
            relative_delta=1.0, order_weight=0.0,
            default_adjacent_gap_tau_seconds=20.0, default_adjacent_gap_lambda=0.05,
        )
        tuples = assemble_region_tuples_for_video(
            "v1", ["e1", "e2", "e3"], regions, candidates_by_id, params, constraints=constraints,
        )
        # (e1,e2) explicit: gap=100, tau=200 -> penalty 0.
        # (e2,e3) default: gap=100, tau=20, lambda=0.05 -> penalty 0.05*80=4.0.
        # gap_penalty is the MEAN across every checked pair, explicit or
        # default alike - not just the unconfigured one.
        expected = fmean([0.8, 0.8, 0.8]) - fmean([0.0, 4.0])
        self.assertAlmostEqual(tuples[0].score, expected)

    def test_default_lambda_zero_exactly_reproduces_pre_change_behavior(self) -> None:
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.8),
            _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.6),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=0.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=500.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.0, default_adjacent_gap_lambda=0.0)
        tuples = assemble_region_tuples_for_video("v1", ["e1", "e2"], regions, candidates_by_id, params)
        self.assertAlmostEqual(tuples[0].score, (0.8 + 0.6) / 2.0)


class ConfidenceGateTests(unittest.TestCase):
    def test_none_gate_is_unchanged(self) -> None:
        params = TupleRankingHyperparameters(order_weight=0.3, confidence_gate="none")
        self.assertEqual(_effective_order_weight(params, 0.1), 0.3)
        self.assertEqual(_effective_order_weight(params, 0.9), 0.3)

    def test_linear_gate_scales_with_confidence(self) -> None:
        params = TupleRankingHyperparameters(order_weight=0.3, confidence_gate="linear")
        self.assertAlmostEqual(_effective_order_weight(params, 0.5), 0.15)
        self.assertAlmostEqual(_effective_order_weight(params, 1.0), 0.3)
        self.assertAlmostEqual(_effective_order_weight(params, 0.0), 0.0)

    def test_threshold_gate_cuts_off_below_threshold(self) -> None:
        params = TupleRankingHyperparameters(order_weight=0.3, confidence_gate="threshold", confidence_gate_threshold=0.5)
        self.assertEqual(_effective_order_weight(params, 0.49), 0.0)
        self.assertEqual(_effective_order_weight(params, 0.5), 0.3)
        self.assertEqual(_effective_order_weight(params, 0.9), 0.3)

    def test_threshold_gate_is_the_production_default(self) -> None:
        # TupleRankingHyperparameters()'s defaults ARE the benchmark's winning
        # configuration (region_tuple_ranking_results.md, Round 2) - this
        # pins that fact so a future default change is a deliberate edit,
        # not a silent drift away from the validated configuration.
        # order_weight and default_adjacent_gap_tau_seconds are the two
        # deliberate manual exceptions (0.8 -> 1.2, and 20.0 -> 25.0 once
        # real contest answers showed a 25fps video's gaps run higher - see
        # each field's own docstring) - neither is benchmark-re-validated.
        params = TupleRankingHyperparameters()
        self.assertEqual(params.confidence_gate, "threshold")
        self.assertEqual(params.confidence_gate_threshold, 0.5)
        self.assertEqual(params.order_weight, 1.2)
        self.assertEqual(params.default_adjacent_gap_tau_seconds, 25.0)
        self.assertEqual(params.pooling, "max")
        # Round 8: N and the combinations cap are effectively unbounded -
        # a real exhaustive search (n=60) found the old N=20/cap=20000
        # never actually truncated anything on this corpus (byte-identical
        # ranks on 59/60 videos, identical aggregate metrics, +3.2% time).
        self.assertEqual(params.max_regions_per_event, 100_000)
        self.assertEqual(params.max_combinations_per_video, 10_000_000)

    def test_linear_gate_suppresses_order_bonus_for_weak_tuple(self) -> None:
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.1),
            _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.1),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
        }
        params_none = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.8, confidence_gate="none")
        params_linear = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.8, confidence_gate="linear")
        tuple_none = assemble_region_tuples_for_video("v1", ["e1", "e2"], regions, candidates_by_id, params_none)[0]
        tuple_linear = assemble_region_tuples_for_video("v1", ["e1", "e2"], regions, candidates_by_id, params_linear)[0]
        self.assertAlmostEqual(tuple_none.score, 0.1 - 0.8)
        self.assertAlmostEqual(tuple_linear.score, 0.1 - 0.8 * 0.1)
        self.assertGreater(tuple_linear.score, tuple_none.score)


class RegionMarginTests(unittest.TestCase):
    def test_top_region_margin_is_gap_to_runner_up(self) -> None:
        pool = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.9),
            _region("r2", event_id="e1", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.6),
            _region("r3", event_id="e1", video_id="v1", candidate_ids=["c3"], normalized_coarse_score=0.5),
        ]
        self.assertAlmostEqual(_region_margin(pool[0], pool), 0.9 - 0.6)

    def test_non_top_region_margin_is_negative(self) -> None:
        pool = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.9),
            _region("r2", event_id="e1", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.6),
        ]
        # Choosing the worse-scoring region: margin is how much WORSE this
        # specific pick is than the pool's best alternative, not the pool's
        # own top-vs-runner-up gap - a tuple reaching past the top region
        # (to satisfy the order term) should read as less confident, not
        # equally confident as if it had picked the top region.
        self.assertAlmostEqual(_region_margin(pool[1], pool), 0.6 - 0.9)

    def test_single_member_pool_margin_is_its_own_score(self) -> None:
        pool = [_region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.7)]
        self.assertAlmostEqual(_region_margin(pool[0], pool), 0.7)

    def test_margin_gate_uses_margin_not_mean_score(self) -> None:
        # Two regions per event, both events tied at the same absolute score
        # (0.8) - region_mean_score is identical for every combination, so
        # only a margin-aware gate can distinguish "the obvious best pick,
        # confidently" from "reaching past a close second choice."
        regions = [
            _region("r1a", event_id="e1", video_id="v1", candidate_ids=["c1a"], normalized_coarse_score=0.8),
            _region("r1b", event_id="e1", video_id="v1", candidate_ids=["c1b"], normalized_coarse_score=0.3),
            _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.8),
        ]
        candidates_by_id = {
            "c1a": _candidate("c1a", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c1b": _candidate("c1b", event_id="e1", video_id="v1", timestamp_seconds=1.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(
            relative_delta=1.0, order_weight=0.5,
            confidence_gate="margin", confidence_gate_threshold=0.4,
        )
        tuples = assemble_region_tuples_for_video("v1", ["e1", "e2"], regions, candidates_by_id, params)
        by_ids = {t.region_ids: t for t in tuples}
        top_pick = by_ids[("r1a", "r2")]
        reach_pick = by_ids[("r1b", "r2")]
        # Mean margin: top_pick = mean(0.8-0.3, 0.8) = 0.65 (e1's margin as
        # the pool's top scorer, plus e2's single-candidate "nothing to
        # compare against" margin of its own score) -> above the 0.4 gate,
        # order bonus applied. reach_pick = mean(0.3-0.8, 0.8) = 0.15 (e1's
        # margin as a worse-than-top pick is NEGATIVE) -> below the gate,
        # order bonus suppressed. Both branches have very different
        # region_mean_score (0.8 vs 0.55) too, so this specifically checks
        # the GATE reads margin_score, not that scores differ at all.
        self.assertGreaterEqual(top_pick.margin_score, 0.4)
        self.assertLess(reach_pick.margin_score, 0.4)
        self.assertAlmostEqual(top_pick.score, top_pick.region_mean_score + 0.5 * top_pick.order_score)
        self.assertAlmostEqual(reach_pick.score, reach_pick.region_mean_score)


class RankVideosByRegionTuplesTests(unittest.TestCase):
    def test_order_weight_zero_matches_prioritize_videos_ranking(self) -> None:
        regions = [
            _region("v1_e1", event_id="e1", video_id="v1", candidate_ids=["v1c1"], normalized_coarse_score=0.9),
            _region("v1_e2", event_id="e2", video_id="v1", candidate_ids=["v1c2"], normalized_coarse_score=0.5),
            _region("v2_e1", event_id="e1", video_id="v2", candidate_ids=["v2c1"], normalized_coarse_score=0.7),
            _region("v2_e2", event_id="e2", video_id="v2", candidate_ids=["v2c2"], normalized_coarse_score=0.7),
        ]
        candidates_by_id = {
            "v1c1": _candidate("v1c1", event_id="e1", video_id="v1", timestamp_seconds=50.0, raw_relevance_score=1.0),
            "v1c2": _candidate("v1c2", event_id="e2", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
            "v2c1": _candidate("v2c1", event_id="e1", video_id="v2", timestamp_seconds=50.0, raw_relevance_score=1.0),
            "v2c2": _candidate("v2c2", event_id="e2", video_id="v2", timestamp_seconds=5.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(
            relative_delta=1.0, order_weight=0.0, pooling="max",
            default_adjacent_gap_lambda=0.0,  # prioritize_videos() has no gap-penalty concept to compare against
        )
        ranking, _ = rank_videos_by_region_tuples(regions, candidates_by_id, ["e1", "e2"], params)

        baseline = prioritize_videos(
            regions, ["e1", "e2"],
            VideoPriorityHyperparameters(video_coverage_weight=0.0, video_mean_weight=1.0, video_min_weight=0.0),
        )
        baseline_order = [p.video_id for p in baseline]
        new_order = [video_id for video_id, _ in ranking]
        self.assertEqual(new_order, baseline_order)
        baseline_by_video = {p.video_id: p.priority_score for p in baseline}
        for video_id, score in ranking:
            self.assertAlmostEqual(score, baseline_by_video[video_id])

    def test_winning_tuple_anchors_are_available_per_event(self) -> None:
        # This is the exact shape video_priorities.py's
        # _matched_frame_ids_and_timestamps() reads to resolve each event's
        # frame_id/timestamp from the winning tuple.
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.8),
            _region("r2", event_id="e2", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.7),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e2", video_id="v1", timestamp_seconds=20.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters()
        _, tuples_by_video = rank_videos_by_region_tuples(regions, candidates_by_id, ["e1", "e2"], params)
        winner = tuples_by_video["v1"][0]
        anchors = {
            event_id: timestamp
            for event_id, timestamp in zip(["e1", "e2"], winner.timestamps)
            if timestamp is not None
        }
        self.assertEqual(anchors, {"e1": 10.0, "e2": 20.0})

    def test_rejected_region_is_never_selected(self) -> None:
        # r1 outscores r2 (0.9 vs 0.5) and would win with no constraints -
        # rejecting r1 must force the winning tuple onto r2, not just lower
        # its rank within an unchanged pool.
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.9),
            _region("r2", event_id="e1", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.5),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e1", video_id="v1", timestamp_seconds=20.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(relative_delta=1.0)
        constraints = SearchConstraints(
            event_constraints={"e1": EventConstraint(rejected_region_ids=frozenset({"r1"}))}
        )
        _, tuples_by_video = rank_videos_by_region_tuples(
            regions, candidates_by_id, ["e1"], params, None, constraints,
        )
        winner = tuples_by_video["v1"][0]
        self.assertEqual(winner.region_ids, ("r2",))

    def test_fixed_region_id_is_always_selected(self) -> None:
        # Same pool, opposite mechanism: pinning r2 (the lower scorer) must
        # force it to win even though r1 would otherwise be preferred.
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.9),
            _region("r2", event_id="e1", video_id="v1", candidate_ids=["c2"], normalized_coarse_score=0.5),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e1", video_id="v1", timestamp_seconds=20.0, raw_relevance_score=1.0),
        }
        params = TupleRankingHyperparameters(relative_delta=1.0)
        # fix_frame() always stamps fixed_video_id alongside fixed_region_id
        # (service.py) - a region pin is scoped to "narrow this video's own
        # choice", not "exclude every other video from this event", so the
        # narrowing check only fires once region.video_id matches.
        constraints = SearchConstraints(
            event_constraints={
                "e1": EventConstraint(fixed_video_id="v1", fixed_region_id="r2")
            }
        )
        _, tuples_by_video = rank_videos_by_region_tuples(
            regions, candidates_by_id, ["e1"], params, None, constraints,
        )
        winner = tuples_by_video["v1"][0]
        self.assertEqual(winner.region_ids, ("r2",))

    def test_fixed_timestamp_without_existing_region_is_respected(self) -> None:
        # commands/fix-frame can set an arbitrary timestamp that doesn't
        # correspond to any real region - the ranker must still honor it,
        # not silently fall back to whichever real candidate scored best.
        regions = [
            _region("r1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.95),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
        }
        constraints = SearchConstraints(
            event_constraints={"e1": EventConstraint(fixed_video_id="v1", fixed_timestamp_seconds=42.0)}
        )
        params = TupleRankingHyperparameters()
        _, tuples_by_video = rank_videos_by_region_tuples(
            regions, candidates_by_id, ["e1"], params, None, constraints,
        )
        winner = tuples_by_video["v1"][0]
        self.assertEqual(winner.timestamps, (42.0,))

    def test_fixed_timestamp_inside_a_broad_region_is_still_reported_exactly(self) -> None:
        # Regression test for a real reported bug: when the fixed timestamp
        # already falls inside an existing region's span,
        # _synthetic_fixed_regions deliberately skips synthesizing a
        # placeholder (already_covered) and defers to that real region -
        # but a *broad* region (start != end, several real candidates
        # inside it) reported its own best-*scoring* candidate's timestamp,
        # not the user's actual pin, whenever those differ. Here the
        # region's best-scoring candidate sits at 10s but the user pinned
        # 42s (still within the region's [0, 50] span) - the reported
        # timestamp must be exactly 42.0, not 10.0.
        regions = [
            _region(
                "r1", event_id="e1", video_id="v1", candidate_ids=["c_best", "c_pinned"],
                normalized_coarse_score=0.9, start_seconds=0.0, end_seconds=50.0,
            ),
        ]
        candidates_by_id = {
            "c_best": _candidate("c_best", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=0.99),
            "c_pinned": _candidate("c_pinned", event_id="e1", video_id="v1", timestamp_seconds=42.0, raw_relevance_score=0.5),
        }
        constraints = SearchConstraints(
            event_constraints={"e1": EventConstraint(fixed_video_id="v1", fixed_timestamp_seconds=42.0)}
        )
        params = TupleRankingHyperparameters()
        _, tuples_by_video = rank_videos_by_region_tuples(
            regions, candidates_by_id, ["e1"], params, None, constraints,
        )
        winner = tuples_by_video["v1"][0]
        self.assertEqual(winner.region_ids, ("r1",))
        self.assertEqual(winner.timestamps, (42.0,))

    def test_fixed_timestamp_does_not_exclude_other_videos_for_that_event(self) -> None:
        # A fix_frame pin on v1 must narrow only v1's own region choice for
        # e1 - it must not remove v2's region from the pool, or v2 would
        # lose all coverage of e1 and vanish from ranking entirely as a side
        # effect of a pin that has nothing to do with it. Cross-video
        # ordering is a separate, explicit mechanism (prioritized_video_ids).
        regions = [
            _region("v1_e1", event_id="e1", video_id="v1", candidate_ids=["c1"], normalized_coarse_score=0.5),
            _region("v2_e1", event_id="e1", video_id="v2", candidate_ids=["c2"], normalized_coarse_score=0.95),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=5.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e1", video_id="v2", timestamp_seconds=6.0, raw_relevance_score=1.0),
        }
        constraints = SearchConstraints(
            event_constraints={"e1": EventConstraint(fixed_video_id="v1", fixed_timestamp_seconds=42.0)}
        )
        params = TupleRankingHyperparameters(relative_delta=1.0)
        ranking, tuples_by_video = rank_videos_by_region_tuples(
            regions, candidates_by_id, ["e1"], params, None, constraints,
        )
        self.assertIn("v2", tuples_by_video)
        self.assertEqual(tuples_by_video["v2"][0].region_ids, ("v2_e1",))
        self.assertEqual(tuples_by_video["v1"][0].timestamps, (42.0,))

    def test_fixed_timestamp_wins_even_when_a_real_region_already_covers_it(self):
        # Distinct from test_fixed_timestamp_without_existing_region_is_
        # respected (there, no region covers the fixed timestamp, so
        # _synthetic_fixed_regions always has to synthesize one). Here the
        # fixed timestamp exactly matches r2's own span, so synthesis has
        # nothing to add (already_covered) - _region_allowed's video-only
        # filter used to leave r1 (higher-scoring, wrong timestamp) equally
        # eligible, and pooling picked it over the user's explicit pick.
        regions = [
            _region(
                "r1", event_id="e1", video_id="v1", candidate_ids=["c1"],
                normalized_coarse_score=0.9, start_seconds=10.0, end_seconds=10.0,
            ),
            _region(
                "r2", event_id="e1", video_id="v1", candidate_ids=["c2"],
                normalized_coarse_score=0.3, start_seconds=20.0, end_seconds=20.0,
            ),
        ]
        candidates_by_id = {
            "c1": _candidate("c1", event_id="e1", video_id="v1", timestamp_seconds=10.0, raw_relevance_score=1.0),
            "c2": _candidate("c2", event_id="e1", video_id="v1", timestamp_seconds=20.0, raw_relevance_score=1.0),
        }
        constraints = SearchConstraints(
            event_constraints={
                "e1": EventConstraint(
                    fixed_video_id="v1", fixed_frame_id=600, fixed_timestamp_seconds=20.0,
                ),
            }
        )
        params = TupleRankingHyperparameters(relative_delta=1.0)
        _, tuples_by_video = rank_videos_by_region_tuples(
            regions, candidates_by_id, ["e1"], params, None, constraints,
        )
        winner = tuples_by_video["v1"][0]
        self.assertEqual(winner.region_ids, ("r2",))
        self.assertEqual(winner.timestamps, (20.0,))


def _event(event_id, *, relation="unknown", reference=None):
    return EventDefinition(
        event_id=event_id,
        anchor_query=event_id,
        temporal_relation=relation,
        reference_event_id=reference,
    )


class BuildOrderConstraintsTests(unittest.TestCase):
    def test_no_relation_data_anywhere_returns_none(self) -> None:
        events = [_event("e1"), _event("e2"), _event("e3")]
        self.assertIsNone(build_order_constraints(events))

    def test_after_produces_reference_then_event_edge(self) -> None:
        events = [_event("e1"), _event("e2", relation="after", reference="e1")]
        self.assertEqual(build_order_constraints(events), [(0, 1)])

    def test_before_produces_event_then_reference_edge(self) -> None:
        events = [_event("e1"), _event("e2", relation="before", reference="e1")]
        # e2 before e1 -> e2 (index 1) precedes e1 (index 0).
        self.assertEqual(build_order_constraints(events), [(1, 0)])

    def test_during_and_simultaneous_produce_no_edge(self) -> None:
        events = [
            _event("e1"),
            _event("e2", relation="during", reference="e1"),
            _event("e3", relation="simultaneous", reference="e1"),
        ]
        self.assertEqual(build_order_constraints(events), [])

    def test_independent_and_sequence_start_produce_no_edge(self) -> None:
        events = [
            _event("e1", relation="sequence_start"),
            _event("e2", relation="independent"),
        ]
        self.assertEqual(build_order_constraints(events), [])

    def test_transitive_closure_adds_the_implied_non_adjacent_pair(self) -> None:
        # e2 after e1, e3 after e2 -> direct edges (0,1) and (1,2), but also
        # the implied (0,2): e1 must precede e3 even though no event
        # references e1 and e3 directly against each other.
        events = [
            _event("e1"),
            _event("e2", relation="after", reference="e1"),
            _event("e3", relation="after", reference="e2"),
        ]
        constraints = build_order_constraints(events)
        self.assertEqual(set(constraints), {(0, 1), (1, 2), (0, 2)})

    def test_this_is_the_scenario_a_naive_list_position_check_gets_backwards(self) -> None:
        # e1 (index 0) is listed first but is explicitly "after" e4 (index 3)
        # - the transitive-closure-aware builder must derive (3, 0), the
        # opposite direction a list-position default would assume.
        events = [
            _event("e1", relation="after", reference="e4"),
            _event("e2"),
            _event("e3"),
            _event("e4", relation="sequence_start"),
        ]
        self.assertIn((3, 0), build_order_constraints(events))
        self.assertNotIn((0, 3), build_order_constraints(events))

    def test_dangling_reference_to_unknown_event_id_is_ignored(self) -> None:
        events = [_event("e1", relation="after", reference="does_not_exist")]
        self.assertEqual(build_order_constraints(events), [])

    def test_end_to_end_with_router_style_order_score_direction(self) -> None:
        # Reproduces the exact scenario a reviewer flagged for _order_score
        # directly, but built from real EventDefinition/build_order_constraints
        # end to end instead of a hand-written constraint list.
        events = [
            _event("e1", relation="after", reference="e2"),
            _event("e2", relation="sequence_start"),
        ]
        constraints = build_order_constraints(events)
        timestamps = [1.0, 4.0]  # e1 precedes e2 in time - violates "e1 after e2"
        self.assertEqual(_order_score(timestamps, constraints), -1.0)


if __name__ == "__main__":
    unittest.main()
