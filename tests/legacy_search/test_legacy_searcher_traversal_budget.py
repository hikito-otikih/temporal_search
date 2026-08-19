"""Regression test for a real reported bug: AmbiguousSearcher/TemporalSearcher's
backtracking() had no bound on total recursive work - top_k_tuple only
capped how many results were *kept*, not the search cost to find them. A
video with many candidate frames could tie up the server for an unbounded
amount of time. These tests confirm the new node-visit budget actually
bounds recursion (not just that it exists) and that results already found
before the budget runs out are still kept, not discarded.
"""

import unittest
from itertools import count
from unittest.mock import patch

from legacy_search.schemas import ClusteredCandidate
from legacy_search.searchers.ambiguous import AmbiguousSearcher
from legacy_search.searchers.temporal import TemporalSearcher


def _counted(cls, method_name: str):
    """Wrap cls.method_name to count every real invocation, including
    recursive self-calls (which resolve through the class attribute at call
    time, so they go through the same wrapper) - the same measurement
    technique used to catch the real bug this guards against: nodes_visited
    silently undercounting real backtracking() calls."""
    original = getattr(cls, method_name)
    calls = {"count": 0}

    def wrapper(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    return calls, patch.object(cls, method_name, wrapper)


def _candidates(number_of_queries: int, per_query: int) -> list[ClusteredCandidate]:
    candidates = []
    for query_id in range(number_of_queries):
        for i in range(per_query):
            candidates.append(
                ClusteredCandidate(
                    frame_index=query_id * 1000 + i,
                    timestamp=f"0:{i:02d}",
                    score=1.0 - i * 0.001,
                    query_id=query_id,
                )
            )
    return candidates


class AmbiguousSearcherTraversalBudgetTests(unittest.TestCase):
    def test_recursion_stops_once_the_node_budget_is_exhausted(self) -> None:
        # 4 query_ids x 15 candidates each, unordered subset search - without
        # a budget this explores on the order of 15^4 combinations.
        results = _candidates(number_of_queries=4, per_query=15)
        query_results: list = []
        searcher = AmbiguousSearcher(
            number_of_queries=4,
            results=results,
            top_k_tuple=5,
            query_results=query_results,
            gamma=0.05,
            video_name="video",
            c=count(),
        )
        searcher.MAX_TRAVERSAL_NODES = 500
        truncated = searcher.start_from_last_element()

        self.assertLessEqual(searcher.nodes_visited, 501)
        self.assertGreater(len(query_results), 0)
        # Regression test for a real reported bug: hitting the budget used
        # to be swallowed silently (bare `except: pass`) - nothing told the
        # caller (and eventually the HTTP response) that results might be
        # incomplete rather than exhaustive.
        self.assertTrue(truncated)

    def test_default_budget_still_finds_results_for_a_small_pool(self) -> None:
        results = _candidates(number_of_queries=3, per_query=4)
        query_results: list = []
        searcher = AmbiguousSearcher(
            number_of_queries=3,
            results=results,
            top_k_tuple=5,
            query_results=query_results,
            gamma=0.05,
            video_name="video",
            c=count(),
        )
        truncated = searcher.start_from_last_element()
        self.assertGreater(len(query_results), 0)
        self.assertFalse(truncated)

    def test_object_filter_rejected_tuples_still_count_toward_the_budget(self) -> None:
        # Regression test for a real reported bug (Blocker 2): a complete
        # tuple rejected by objectFilterMode used to `return` before
        # incrementing nodes_visited, so with objectFilterMode on and most
        # tuples failing the filter, real recursive work vastly exceeded the
        # configured budget while nodes_visited kept reporting a small
        # number - measured 1,011 real backtracking() calls against a
        # reported nodes_visited of 11 for MAX_TRAVERSAL_NODES=10.
        results = _candidates(number_of_queries=4, per_query=10)
        for candidate in results:
            candidate.satisfiedObjects = False  # every complete tuple gets rejected
        query_results: list = []
        searcher = AmbiguousSearcher(
            number_of_queries=4,
            results=results,
            top_k_tuple=5,
            query_results=query_results,
            gamma=0.05,
            video_name="video",
            c=count(),
            objectFilterMode=True,
        )
        searcher.MAX_TRAVERSAL_NODES = 10

        calls, patcher = _counted(AmbiguousSearcher, "backtracking")
        with patcher:
            searcher.start_from_last_element()

        self.assertLessEqual(searcher.nodes_visited, 11)
        self.assertEqual(calls["count"], searcher.nodes_visited)
        self.assertEqual(len(query_results), 0)  # everything was rejected


class TemporalSearcherTraversalBudgetTests(unittest.TestCase):
    def test_recursion_stops_once_the_node_budget_is_exhausted(self) -> None:
        number_of_queries = 4
        per_query = 20
        results = _candidates(number_of_queries, per_query)
        list_indices: list[list[int]] = [[] for _ in range(number_of_queries)]
        list_prev_indices = [-1] * len(results)
        list_nearest_indices = [-1] * number_of_queries
        list_endable = [0] * len(results)
        for idx, result in enumerate(results):
            list_indices[result.query_id].append(idx)
            if result.query_id > 0:
                list_prev_indices[idx] = list_nearest_indices[result.query_id - 1]
            list_endable[idx] = result.query_id == 0 or list_prev_indices[idx] != -1
            if list_endable[idx]:
                list_nearest_indices[result.query_id] = len(list_indices[result.query_id]) - 1

        query_results: list = []
        searcher = TemporalSearcher(
            number_of_queries=number_of_queries,
            results=results,
            top_k_tuple=5,
            query_results=query_results,
            list_indices=list_indices,
            list_prev_indices=list_prev_indices,
            list_endable=list_endable,
            gamma=0.05,
            video_name="video",
            c=count(),
        )
        searcher.MAX_TRAVERSAL_NODES = 500
        truncated = searcher.start_from_last_query()

        self.assertLessEqual(searcher.nodes_visited, 501)
        self.assertGreater(len(query_results), 0)
        self.assertTrue(truncated)

    def test_object_filter_rejected_tuples_still_count_toward_the_budget(self) -> None:
        # Regression test for a real reported bug (Blocker 2): same as
        # AmbiguousSearcher's version - a complete tuple rejected by
        # objectFilterMode used to `return` before incrementing
        # nodes_visited, so real recursive work vastly exceeded the
        # configured budget while nodes_visited kept reporting a small
        # number - measured 911 real backtracking() calls against a
        # reported nodes_visited of 11 for MAX_TRAVERSAL_NODES=10.
        number_of_queries = 4
        per_query = 10
        results = _candidates(number_of_queries, per_query)
        for candidate in results:
            candidate.satisfiedObjects = False  # every complete tuple gets rejected
        list_indices: list[list[int]] = [[] for _ in range(number_of_queries)]
        list_prev_indices = [-1] * len(results)
        list_nearest_indices = [-1] * number_of_queries
        list_endable = [0] * len(results)
        for idx, result in enumerate(results):
            list_indices[result.query_id].append(idx)
            if result.query_id > 0:
                list_prev_indices[idx] = list_nearest_indices[result.query_id - 1]
            list_endable[idx] = result.query_id == 0 or list_prev_indices[idx] != -1
            if list_endable[idx]:
                list_nearest_indices[result.query_id] = len(list_indices[result.query_id]) - 1

        query_results: list = []
        searcher = TemporalSearcher(
            number_of_queries=number_of_queries,
            results=results,
            top_k_tuple=5,
            query_results=query_results,
            list_indices=list_indices,
            list_prev_indices=list_prev_indices,
            list_endable=list_endable,
            gamma=0.05,
            video_name="video",
            c=count(),
            objectFilterMode=True,
        )
        searcher.MAX_TRAVERSAL_NODES = 10

        calls, patcher = _counted(TemporalSearcher, "backtracking")
        with patcher:
            searcher.start_from_last_query()

        self.assertLessEqual(searcher.nodes_visited, 11)
        self.assertEqual(calls["count"], searcher.nodes_visited)
        self.assertEqual(len(query_results), 0)  # everything was rejected


if __name__ == "__main__":
    unittest.main()
