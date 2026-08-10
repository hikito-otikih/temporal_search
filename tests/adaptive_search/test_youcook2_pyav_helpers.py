import unittest
from fractions import Fraction

from adaptive_search.providers import _frame_pts_ms, _pyav_duration_ms


class _Frame:
    pts = 3


class _Stream:
    duration = 31
    time_base = Fraction(1, 30)


class _Container:
    duration = None


class YouCook2PyavHelperTests(unittest.TestCase):
    def test_fraction_conversion_uses_complete_denominator_attribute(self):
        self.assertEqual(_frame_pts_ms(_Frame(), Fraction(1, 30)), 100)

    def test_frame_pts_is_normalized_to_stream_start_time(self):
        frame = _Frame()
        frame.pts = 103
        self.assertEqual(
            _frame_pts_ms(frame, Fraction(1, 30), start_time=100),
            100,
        )
        self.assertEqual(_pyav_duration_ms(_Container(), _Stream()), 1034)


if __name__ == "__main__":
    unittest.main()
