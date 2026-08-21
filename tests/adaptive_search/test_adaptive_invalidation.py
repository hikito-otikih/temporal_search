import unittest

from adaptive_search.invalidation import (
    artifact_fingerprint,
    invalidated_for_event,
    invalidated_for_parameters,
)


class AdaptiveInvalidationTests(unittest.TestCase):
    def test_tuple_ranking_change_invalidates_tuple_only(self):
        # A real, valid patch to tuple_ranking (the sole ranking algorithm's
        # own hyperparameter section) once had no matching prefix at all,
        # raising an uncaught "no invalidation rule for path" ValueError
        # that surfaced as a 422 for a semantically valid config change.
        paths = {"tuple_ranking.order_weight"}
        self.assertEqual(invalidated_for_parameters(paths), ["tuple"])

    def test_boundary_type_event_change_starts_at_tuple(self):
        self.assertEqual(
            invalidated_for_event({"event.boundary_type"}),
            ["tuple"],
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
