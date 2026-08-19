from ..schemas import ClusteredCandidate
import heapq
from itertools import count

from .ambiguous import TraversalBudget, _TraversalBudgetExceeded


class TemporalSearcher :
    # Same safety valve as AmbiguousSearcher.MAX_TRAVERSAL_NODES - this
    # searcher's recursion is more structurally constrained (it only walks
    # DAG-shaped chains via list_prev_indices/list_endable, not arbitrary
    # subsets), but nothing formally bounds it either, so it gets the same
    # defense in depth. Defaults used only when no shared `budget` is
    # passed in; temporal_search() shares one TraversalBudget across every
    # video in a request instead of resetting it per video.
    MAX_TRAVERSAL_NODES = 200_000
    MAX_TRAVERSAL_SECONDS = 20.0

    def __init__(self, number_of_queries: int, results: list[ClusteredCandidate], top_k_tuple: int
                 , query_results: list[tuple[float, int, str, list[ClusteredCandidate]]]
                 , list_indices: list[list[int]], list_prev_indices: list[int]
                 , list_endable: list[int], gamma: float, video_name: str, c: count, objectFilterMode: bool = False
                 , budget: TraversalBudget | None = None) -> None:
        self.results = results
        self.top_k_tuple = top_k_tuple
        self.number_of_queries = number_of_queries
        self.query_results = query_results
        self.list_indices = list_indices
        self.list_prev_indices = list_prev_indices
        self.list_endable = list_endable
        self.gamma = gamma
        self.cur_List_Frame = []
        self.video_name = video_name
        self.objectFilterMode = objectFilterMode
        self.c = c
        # See AmbiguousSearcher's identical comment: built lazily in
        # start_from_last_query(), not here, so a post-construction
        # MAX_TRAVERSAL_NODES/MAX_TRAVERSAL_SECONDS override (existing test
        # pattern) still takes effect when no external `budget` is given.
        self._external_budget = budget
        self.budget: TraversalBudget | None = None

    @property
    def nodes_visited(self) -> int:
        return self.budget.nodes_visited if self.budget is not None else 0

    def backtracking(self, curScore = 0, curDistance = 0, sumFrameId = 0, curId = 0, cur_query_id = None, satisfiedObjects = False) -> None:
        # Counted before anything else - including both early returns below
        # (the objectFilterMode rejection and the dead-end "no valid
        # predecessor" case) - so every call spends its share of the
        # budget. It used to increment after those returns, so with
        # objectFilterMode on and most complete tuples failing the filter,
        # the majority of recursive calls went completely uncounted:
        # measured 911 real calls against a reported nodes_visited of only
        # 11 for MAX_TRAVERSAL_NODES=10.
        self.budget.spend_one()
        if self.list_prev_indices[curId] == -1:
            if cur_query_id == -1: ## reach the end of the query list
                avgScore = curScore / self.number_of_queries
                finalScore = avgScore / (1 + curDistance * self.gamma)
                result_tuple = self.cur_List_Frame[::-1]
                if self.objectFilterMode and not satisfiedObjects:
                    return
                if len(self.query_results) < self.top_k_tuple:
                    heapq.heappush(self.query_results, (finalScore, next(self.c), self.video_name, result_tuple))
                else:
                    if finalScore > self.query_results[0][0]:
                        heapq.heappop(self.query_results)
                        heapq.heappush(self.query_results, (finalScore, next(self.c), self.video_name, result_tuple))
            else:
                return
        for i in reversed(range(0, self.list_prev_indices[curId] + 1)):
            newId = self.list_indices[cur_query_id][i]
            if not self.list_endable[newId]:
                return
            self.cur_List_Frame.append(self.results[newId])
            self.backtracking(curScore + self.results[newId].score,
                              curDistance + sumFrameId - self.results[newId].frame_index * (self.number_of_queries - cur_query_id - 1),
                              sumFrameId + self.results[newId].frame_index, newId, cur_query_id - 1, satisfiedObjects or bool(self.results[newId].satisfiedObjects))
            self.cur_List_Frame.pop()
    def start_from_last_query(self) -> bool:
        """Returns True if the node budget was hit before the search
        finished exploring - the caller keeps whatever query_results were
        already found, but the result set may be incomplete rather than
        exhaustive."""
        self.budget = self._external_budget or TraversalBudget(
            max_nodes=self.MAX_TRAVERSAL_NODES, max_seconds=self.MAX_TRAVERSAL_SECONDS
        )
        try:
            for i in reversed(range(0, len(self.list_indices[self.number_of_queries - 1]))):
                if not self.list_endable[self.list_indices[self.number_of_queries - 1][i]]:
                    return False
                self.cur_List_Frame.append(self.results[self.list_indices[self.number_of_queries - 1][i]])
                self.backtracking(self.results[self.list_indices[self.number_of_queries - 1][i]].score, 0, self.results[self.list_indices[self.number_of_queries - 1][i]].frame_index, self.list_indices[self.number_of_queries - 1][i], self.number_of_queries - 2, bool(self.results[self.list_indices[self.number_of_queries - 1][i]].satisfiedObjects))
                self.cur_List_Frame.pop()
        except _TraversalBudgetExceeded:
            return True  # keep whatever was already found in query_results, stop expanding further
        return False
