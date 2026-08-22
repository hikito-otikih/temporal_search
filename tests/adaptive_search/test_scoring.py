import unittest

from adaptive_search.algorithms import min_max_normalize


class MinMaxNormalizeTests(unittest.TestCase):
    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(min_max_normalize([]), [])

    def test_single_value_maps_to_midpoint(self) -> None:
        self.assertEqual(min_max_normalize([3.7]), [0.5])

    def test_all_values_equal_maps_every_entry_to_midpoint(self) -> None:
        self.assertEqual(min_max_normalize([2.0, 2.0, 2.0]), [0.5, 0.5, 0.5])

    def test_min_and_max_land_exactly_on_the_unit_bounds(self) -> None:
        normalized = min_max_normalize([1.0, 3.0, 5.0])
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[2], 1.0)
        self.assertAlmostEqual(normalized[1], 0.5)

    def test_preserves_relative_spacing_not_just_rank(self) -> None:
        # Two values close together near the top and one far below - a rank-
        # or z-score-based transform would treat "close together" as "hard
        # to tell apart"; min-max keeps them close together but still
        # clearly above the outlier, exactly reflecting the real gap.
        normalized = min_max_normalize([10.0, 9.9, 0.0])
        self.assertAlmostEqual(normalized[2], 0.0)
        self.assertAlmostEqual(normalized[0], 1.0)
        self.assertAlmostEqual(normalized[1], 0.99)

    def test_a_tight_top_cluster_spreads_across_the_full_range_not_just_near_one(
        self,
    ) -> None:
        # The exact failure mode this function replaces robust_sigmoid() to
        # avoid: a handful of raw scores clustered close together, far above
        # a wide, low-scoring tail. robust_sigmoid() z-scores the cluster
        # against the whole population and can compress it into a narrow
        # band near its ceiling (or literally tie once clip_z is cleared);
        # min-max always uses the cluster's own min/max as the [0,1]
        # anchors, so the top of the cluster reads as meaningfully ahead of
        # everything else, including the rest of the cluster.
        strong_cluster = [2.20, 2.19, 2.18, 2.17]
        weak_tail = [0.5, 0.2, -0.3, -0.9, -1.4]
        normalized = min_max_normalize(strong_cluster + weak_tail)
        strong_normalized = normalized[: len(strong_cluster)]
        self.assertEqual(len(set(round(value, 6) for value in strong_normalized)), 4)
        self.assertTrue(all(value > 0.9 for value in strong_normalized))
        self.assertAlmostEqual(max(strong_normalized), 1.0)

    def test_output_is_unaffected_by_input_order(self) -> None:
        values = [4.0, -2.0, 9.0, 0.0]
        normalized = min_max_normalize(values)
        shuffled = [values[2], values[0], values[3], values[1]]
        normalized_shuffled = min_max_normalize(shuffled)
        self.assertAlmostEqual(normalized[2], normalized_shuffled[0])
        self.assertAlmostEqual(normalized[0], normalized_shuffled[1])
        self.assertAlmostEqual(normalized[3], normalized_shuffled[2])
        self.assertAlmostEqual(normalized[1], normalized_shuffled[3])


if __name__ == "__main__":
    unittest.main()
