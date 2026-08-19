"""check_object_satisfaction() against real shipped fixtures
(data/map-keyframes/L21_V002.csv, data/objects/L21_V002/*.json).

Regression test for a real off-by-one: the detection JSON filename must be
keyed by the CSV's own 1-indexed `n` column, not pandas' 0-indexed
DataFrame row position. For L21_V002, frame_idx=90 is row n=2 (001.json is
a valid, empty-detections fixture; 002.json holds the real detections for
that row) - the bug opened 001.json for every row of every video, not just
this one frame, silently returning "no objects satisfied" regardless of
what the correct row's file actually contained.
"""

import unittest

from legacy_search.schemas import ClusteredCandidate, Videos
from legacy_search.service import check_object_satisfaction


class CheckObjectSatisfactionTests(unittest.TestCase):
    def test_frame_90_reads_the_row_2_detection_file_not_row_1(self):
        candidate = ClusteredCandidate(frame_index=90, timestamp="00:00:03", score=1.0)
        videos = [Videos(video_name="L21_V002.mp4", results=[candidate])]

        # check_object_satisfaction matches against detection_class_entities
        # (the human-readable names, e.g. "Lantern"), not
        # detection_class_names (the raw "/m/01jfsr"-style class-ID strings
        # a real caller never types - object_name_list=["cat", "dog"] is the
        # documented example usage). "Lantern" is 002.json's real top
        # detection at score 0.797. If the code were still opening 001.json
        # (valid JSON, but zero detections), this would come back False
        # instead.
        check_object_satisfaction(videos, ["Lantern"], objectThreshold=0.5)

        self.assertTrue(candidate.satisfiedObjects)

    def test_matching_is_case_insensitive(self):
        # Regression test for a real reported bug: check_object_satisfaction
        # used to match against detection_class_names (machine IDs like
        # "/m/01jfsr") instead of detection_class_entities (human-readable
        # labels like "Lantern") - object_name_list is written in plain,
        # often lowercase words ("cat", "dog"), so matching must also be
        # case-insensitive against the capitalized Open Images entities.
        candidate = ClusteredCandidate(frame_index=90, timestamp="00:00:03", score=1.0)
        videos = [Videos(video_name="L21_V002.mp4", results=[candidate])]

        check_object_satisfaction(videos, ["lantern"], objectThreshold=0.5)

        self.assertTrue(candidate.satisfiedObjects)

    def test_unmet_object_threshold_is_not_satisfied(self):
        candidate = ClusteredCandidate(frame_index=90, timestamp="00:00:03", score=1.0)
        videos = [Videos(video_name="L21_V002.mp4", results=[candidate])]

        check_object_satisfaction(videos, ["a object that does not exist"], objectThreshold=0.5)

        self.assertFalse(candidate.satisfiedObjects)


if __name__ == "__main__":
    unittest.main()
