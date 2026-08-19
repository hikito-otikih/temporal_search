import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ui_models import normalize_adaptive_items


class NormalizeAdaptiveItemsTests(unittest.TestCase):
    def test_maps_applied_boundary_refinement_into_moments(self):
        page = {
            "items": [
                {
                    "video_id": "video_a",
                    "priority_score": 0.8,
                    "event_coverage": 2,
                    "boundary_refinement": {
                        "status": "applied",
                        "events": [
                            {"event_id": "e1", "refined_seconds": 5.0, "used_fallback": False, "source": "tuple_ranking"},
                            {"event_id": "e2", "refined_seconds": 12.0, "used_fallback": True, "source": "auto"},
                        ],
                    },
                }
            ]
        }
        items = normalize_adaptive_items(page, event_count=2)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["video_id"], "video_a")
        self.assertEqual(item["reject_key"], "video_a")
        self.assertEqual(item["coverage_label"], "Matched 2/2 moments")
        self.assertEqual(len(item["moments"]), 2)
        self.assertEqual(item["moments"][0]["seconds"], 5.0)
        self.assertTrue(item["moments"][0]["refined"])
        self.assertFalse(item["moments"][1]["refined"])  # used_fallback=True

    def test_not_requested_boundary_refinement_yields_no_moments(self):
        page = {
            "items": [
                {"video_id": "video_a", "priority_score": 0.5, "event_coverage": 1,
                 "boundary_refinement": {"status": "not_requested", "events": None}},
            ]
        }
        items = normalize_adaptive_items(page, event_count=1)
        self.assertEqual(items[0]["moments"], [])

    def test_empty_items_returns_empty_list(self):
        self.assertEqual(normalize_adaptive_items({"items": []}, event_count=3), [])


if __name__ == "__main__":
    unittest.main()
