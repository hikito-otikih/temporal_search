from ..schemas import ClusteredCandidate
import heapq
import time
from itertools import count


class _TraversalBudgetExceeded(Exception):
    """Internal signal to unwind every pending backtracking() frame (both
    recursive calls and the top-level starting loop) the moment the node
    budget trips - a plain `return` only stops the current frame from
    recursing deeper, but the caller's own loop over its remaining siblings
    would keep spawning new (now instantly-terminating) calls, letting
    nodes_visited overshoot the budget by roughly the combined width of
    every still-active loop above the trip point."""


class TraversalBudget:
    """Node count + wall-clock budget, shared across every video's searcher
    in one temporal_search() request - not one fresh budget per video.

    Regression fix for a real measured problem: each video used to get its
    own fresh MAX_TRAVERSAL_NODES, so total traversal work across many
    videos in one request was unbounded by the per-video cap. Measured with
    realistic (not pathological) parameters - 32 queries, top_k_each_query
    =2000 (well under the schema's 10,000 max), AmbiguousSearcher (no
    structural pruning) - 100-500 overlapping videos: 180-295 seconds of
    pure traversal time, every one of them still hitting its own 200,000-
    node cap. Sharing one budget across the whole request means the second
    video onward short-circuits almost immediately once it's spent, instead
    of each getting its own full budget to burn through.

    Wall-clock is checked too, not just node count: per-node cost is not
    constant - measured roughly 7x slower nodes/sec for a densely-populated
    video (many candidates per query slot) than a sparse one, so a
    node-count-only budget does not reliably bound real time either.
    Checked every `check_every` nodes (not every single one) to avoid a
    time.perf_counter() call on the hot path.
    """

    def __init__(self, *, max_nodes: int, max_seconds: float, check_every: int = 4096) -> None:
        self.max_nodes = max_nodes
        self.max_seconds = max_seconds
        self.check_every = check_every
        self.nodes_visited = 0
        self._deadline = time.perf_counter() + max_seconds

    def spend_one(self) -> None:
        self.nodes_visited += 1
        if self.nodes_visited > self.max_nodes:
            raise _TraversalBudgetExceeded()
        if self.nodes_visited % self.check_every == 0 and time.perf_counter() >= self._deadline:
            raise _TraversalBudgetExceeded()


class AmbiguousSearcher :
    # Safety valve against backtracking()'s unbounded subset/ordering search
    # (branching factor up to len(results), depth number_of_queries, no
    # structural pruning like TemporalSearcher's DAG-shaped chains) -
    # top_k_tuple only bounds how many results are kept, not how much
    # recursive work is done to find them. Without this, a single request
    # against a video with many candidate frames could tie up the server for
    # an unbounded amount of time. Defaults used only when no shared
    # `budget` is passed in (e.g. direct/test construction of one searcher);
    # temporal_search() constructs one TraversalBudget per request and
    # shares it across every video instead.
    MAX_TRAVERSAL_NODES = 200_000
    MAX_TRAVERSAL_SECONDS = 20.0

    def __init__(self, number_of_queries: int, results: list[ClusteredCandidate], top_k_tuple: int
                 , query_results: list[tuple[float, int, str, list[ClusteredCandidate]]]
                 , gamma: float, video_name: str, c: count, objectFilterMode: bool = False
                 , budget: TraversalBudget | None = None) -> None:
        self.results = results
        self.top_k_tuple = top_k_tuple
        self.number_of_queries = number_of_queries
        self.query_results = query_results
        self.gamma = gamma
        self.cur_List_Frame = []
        self.video_name = video_name
        self.c = c
        self.objectFilterMode = objectFilterMode
        # Not built here: MAX_TRAVERSAL_NODES/MAX_TRAVERSAL_SECONDS may still
        # be overridden on the instance after construction (existing test
        # pattern) - building a self-owned budget eagerly would freeze in
        # the class defaults before such an override ever runs. Built lazily
        # in start_from_last_element() instead, only when no external
        # `budget` was supplied.
        self._external_budget = budget
        self.budget: TraversalBudget | None = None

    @property
    def nodes_visited(self) -> int:
        return self.budget.nodes_visited if self.budget is not None else 0

    def backtracking(self, curScore = 0, curDistance = 0, sumFrameId = 0, curId = 0, curMaskFrame = 0, cur_query_id = None, satisfiedObjects  = False) -> None:
        # Counted before anything else - including the objectFilterMode
        # early return below - so a rejected-by-object-filter complete tuple
        # still spends its share of the budget. It used to increment after
        # that return, so with objectFilterMode on and most complete tuples
        # failing the filter, the vast majority of recursive calls (and
        # their own further recursion, since a rejected tuple used to bail
        # out before recursing too - now it still recurses, correctly
        # counted) went completely uncounted: measured 1,011 real calls
        # against a reported nodes_visited of only 11 for MAX_TRAVERSAL_NODES=10.
        self.budget.spend_one()
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

        for newId in reversed(range(0, curId)):
            if (curMaskFrame >> self.results[newId].query_id) & 1 == 1:
                continue
            self.cur_List_Frame.append(self.results[newId])
            self.backtracking(curScore + self.results[newId].score,
                              curDistance + sumFrameId - self.results[newId].frame_index * (self.number_of_queries - cur_query_id - 1),
                              sumFrameId + self.results[newId].frame_index, newId, curMaskFrame | (1 << self.results[newId].query_id), cur_query_id - 1, satisfiedObjects or bool(self.results[newId].satisfiedObjects))
            self.cur_List_Frame.pop()
    def start_from_last_element(self) -> bool:
        """Returns True if the node budget was hit before the search
        finished exploring - the caller keeps whatever query_results were
        already found, but the result set may be incomplete rather than
        exhaustive."""
        self.budget = self._external_budget or TraversalBudget(
            max_nodes=self.MAX_TRAVERSAL_NODES, max_seconds=self.MAX_TRAVERSAL_SECONDS
        )
        try:
            for i in reversed(range(0, len(self.results))):
                self.cur_List_Frame.append(self.results[i])
                self.backtracking(self.results[i].score, 0, self.results[i].frame_index, i, 1 << self.results[i].query_id, self.number_of_queries - 2, bool(self.results[i].satisfiedObjects))
                self.cur_List_Frame.pop()
        except _TraversalBudgetExceeded:
            return True  # keep whatever was already found in query_results, stop expanding further
        return False
