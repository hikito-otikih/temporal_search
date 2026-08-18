from ..schemas import ClusteredCandidate
import heapq
from itertools import count


class _TraversalBudgetExceeded(Exception):
    """Internal signal to unwind every pending backtracking() frame (both
    recursive calls and the top-level starting loop) the moment the node
    budget trips - a plain `return` only stops the current frame from
    recursing deeper, but the caller's own loop over its remaining siblings
    would keep spawning new (now instantly-terminating) calls, letting
    nodes_visited overshoot the budget by roughly the combined width of
    every still-active loop above the trip point."""


class AmbiguousSearcher :
    # Safety valve against backtracking()'s unbounded subset/ordering search
    # (branching factor up to len(results), depth number_of_queries, no
    # structural pruning like TemporalSearcher's DAG-shaped chains) -
    # top_k_tuple only bounds how many results are kept, not how much
    # recursive work is done to find them. Without this, a single request
    # against a video with many candidate frames could tie up the server for
    # an unbounded amount of time.
    MAX_TRAVERSAL_NODES = 200_000

    def __init__(self, number_of_queries: int, results: list[ClusteredCandidate], top_k_tuple: int
                 , query_results: list[tuple[float, int, str, list[ClusteredCandidate]]]
                 , gamma: float, video_name: str, c: count, objectFilterMode: bool = False) -> None:
        self.results = results
        self.top_k_tuple = top_k_tuple
        self.number_of_queries = number_of_queries
        self.query_results = query_results
        self.gamma = gamma
        self.cur_List_Frame = []
        self.video_name = video_name
        self.c = c
        self.objectFilterMode = objectFilterMode
        self.nodes_visited = 0
    def backtracking(self, curScore = 0, curDistance = 0, sumFrameId = 0, curId = 0, curMaskFrame = 0, cur_query_id = None, satisfiedObjects  = False) -> None:
        if cur_query_id == -1:
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

        self.nodes_visited += 1
        if self.nodes_visited > self.MAX_TRAVERSAL_NODES:
            raise _TraversalBudgetExceeded()
        for newId in reversed(range(0, curId)):
            if (curMaskFrame >> self.results[newId].query_id) & 1 == 1:
                continue
            self.cur_List_Frame.append(self.results[newId])
            self.backtracking(curScore + self.results[newId].score,
                              curDistance + sumFrameId - self.results[newId].frame_index * (self.number_of_queries - cur_query_id - 1),
                              sumFrameId + self.results[newId].frame_index, newId, curMaskFrame | (1 << self.results[newId].query_id), cur_query_id - 1, satisfiedObjects or bool(self.results[newId].satisfiedObjects))
            self.cur_List_Frame.pop()
    def start_from_last_element(self) -> None:
        try:
            for i in reversed(range(0, len(self.results))):
                self.cur_List_Frame.append(self.results[i])
                self.backtracking(self.results[i].score, 0, self.results[i].frame_index, i, 1 << self.results[i].query_id, self.number_of_queries - 2, bool(self.results[i].satisfiedObjects))
                self.cur_List_Frame.pop()
        except _TraversalBudgetExceeded:
            pass  # keep whatever was already found in query_results, stop expanding further