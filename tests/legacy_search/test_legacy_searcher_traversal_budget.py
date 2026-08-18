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

from legacy_search.schemas import ClusteredCandidate
from legacy_search.searchers.ambiguous import AmbiguousSearcher
from legacy_search.searchers.temporal import TemporalSearcher


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
        searcher.start_from_last_element()

        self.assertLessEqual(searcher.nodes_visited, 501)
        self.assertGreater(len(query_results), 0)

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
        searcher.start_from_last_element()
        self.assertGreater(len(query_results), 0)


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
        searcher.start_from_last_query()

        self.assertLessEqual(searcher.nodes_visited, 501)
        self.assertGreater(len(query_results), 0)


if __name__ == "__main__":
    unittest.main()
