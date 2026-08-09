import unittest

from adaptive_search.invalidation import (
    artifact_fingerprint,
    changed_paths,
    invalidated_for_event,
    invalidated_for_parameters,
)


class AdaptiveInvalidationTests(unittest.TestCase):
    def test_ranking_change_invalidates_tuple_only(self):
        before = {"ranking": {"top_k": 10}, "clustering": {"gap_seconds": 3.0}}
        after = {"ranking": {"top_k": 20}, "clustering": {"gap_seconds": 3.0}}
        paths = changed_paths(before, after)

        self.assertEqual(paths, {"ranking.top_k"})
        self.assertEqual(invalidated_for_parameters(paths), ["tuple"])

    def test_frontier_budget_change_starts_at_frontier(self):
        paths = {"refinement.max_total_regions"}
        self.assertEqual(
            invalidated_for_parameters(paths),
            ["frontier", "refinement", "proposal", "tuple"],
        )

    def test_pre_state_event_change_reuses_sparse_retrieval(self):
        self.assertEqual(
            invalidated_for_event({"event.pre_state_query"}),
            ["refinement", "proposal", "tuple"],
        )

    def test_fingerprint_is_order_independent_but_code_sensitive(self):
        first = artifact_fingerprint(
            artifact_type="region",
            inputs={"b": 2, "a": 1},
            code_version="v1",
        )
        reordered = artifact_fingerprint(
            artifact_type="region",
            inputs={"a": 1, "b": 2},
            code_version="v1",
        )
        changed_code = artifact_fingerprint(
            artifact_type="region",
            inputs={"a": 1, "b": 2},
            code_version="v2",
        )

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed_code)


if __name__ == "__main__":
    unittest.main()
