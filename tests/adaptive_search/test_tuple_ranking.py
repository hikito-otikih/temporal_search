from __future__ import annotations

import unittest

from adaptive_search.algorithms import prioritize_videos
from adaptive_search.schemas import (
    EventDefinition,
    RefinementHyperparameters,
    SparseCandidate,
    TemporalRegion,
    TupleRankingHyperparameters,
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
        query_variant="v0",
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
        params = TupleRankingHyperparameters()
        self.assertEqual(params.confidence_gate, "threshold")
        self.assertEqual(params.confidence_gate_threshold, 0.5)
        self.assertEqual(params.order_weight, 0.8)
        self.assertEqual(params.pooling, "max")

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
        params = TupleRankingHyperparameters(relative_delta=1.0, order_weight=0.0, pooling="max")
        ranking, _ = rank_videos_by_region_tuples(regions, candidates_by_id, ["e1", "e2"], params)

        baseline = prioritize_videos(
            regions, ["e1", "e2"],
            RefinementHyperparameters(video_coverage_weight=0.0, video_mean_weight=1.0, video_min_weight=0.0),
        )
        baseline_order = [p.video_id for p in baseline]
        new_order = [video_id for video_id, _ in ranking]
        self.assertEqual(new_order, baseline_order)
        baseline_by_video = {p.video_id: p.priority_score for p in baseline}
        for video_id, score in ranking:
            self.assertAlmostEqual(score, baseline_by_video[video_id])

    def test_winning_tuple_anchors_are_available_for_seeding_refinement(self) -> None:
        # This is the exact shape router.py's get_video_priorities() reads
        # to seed boundary refinement from the winning tuple instead of an
        # independent select_event_seeds() call.
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


def _event(event_id, *, relation="unknown", reference=None):
    return EventDefinition(
        event_id=event_id,
        original_query=event_id,
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
