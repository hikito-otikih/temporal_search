import unittest

from adaptive_search.algorithms import prioritize_videos
from adaptive_search.schemas import TemporalRegion, VideoPriorityHyperparameters


class AdaptiveAlgorithmTests(unittest.TestCase):
    def test_prioritize_videos_distinctness_defaults_to_inert(self):
        """video_distinctness_weight=0.0 (default) must not change
        priority_score at all - distinctness is still computed and exposed,
        just not weighted into the ranking."""
        regions = [
            TemporalRegion(
                id="r1", session_id="session", event_id="e1", video_id="video",
                start_seconds=9.0, end_seconds=9.0, candidate_ids=("c1",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
            TemporalRegion(
                id="r2", session_id="session", event_id="e2", video_id="video",
                start_seconds=9.0, end_seconds=9.0, candidate_ids=("c2",),
                raw_coarse_score=0.7, normalized_coarse_score=0.7,
            ),
        ]
        priorities = prioritize_videos(regions, ["e1", "e2"])
        self.assertEqual(len(priorities), 1)
        self.assertEqual(priorities[0].distinctness, 0.0)
        self.assertAlmostEqual(priorities[0].priority_score, priorities[0].mean_best_event_score)

    def test_prioritize_videos_distinctness_penalizes_collapsed_timestamps(self):
        """With video_distinctness_weight > 0, a video whose covered events'
        timestamps collapse onto the same moment ranks below an
        equal-mean-score video whose events are genuinely spread out - the
        false-positive pattern this field exists to catch (region_tuple_
        ranking_results.md: 25% of rank-1 videos on the real n=60 corpus had
        an exact collision, 14/15 of them wrong answers)."""
        collapsed = [
            TemporalRegion(
                id="r1", session_id="s", event_id="e1", video_id="collapsed",
                start_seconds=9.0, end_seconds=9.0, candidate_ids=("c1",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
            TemporalRegion(
                id="r2", session_id="s", event_id="e2", video_id="collapsed",
                start_seconds=9.0, end_seconds=9.0, candidate_ids=("c2",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
        ]
        spread = [
            TemporalRegion(
                id="r3", session_id="s", event_id="e1", video_id="spread",
                start_seconds=9.0, end_seconds=9.0, candidate_ids=("c3",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
            TemporalRegion(
                id="r4", session_id="s", event_id="e2", video_id="spread",
                start_seconds=50.0, end_seconds=50.0, candidate_ids=("c4",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
        ]
        parameters = VideoPriorityHyperparameters(
            video_mean_weight=1.0, video_distinctness_weight=1.0,
            distinctness_norm_seconds=3.0,
        )
        priorities = prioritize_videos(collapsed + spread, ["e1", "e2"], parameters)
        by_video = {item.video_id: item for item in priorities}
        self.assertEqual(
            by_video["collapsed"].mean_best_event_score,
            by_video["spread"].mean_best_event_score,
        )
        self.assertEqual(by_video["collapsed"].distinctness, 0.0)
        self.assertEqual(by_video["spread"].distinctness, 1.0)
        self.assertLess(by_video["collapsed"].priority_score, by_video["spread"].priority_score)
        self.assertEqual(priorities[0].video_id, "spread")

    def test_prioritize_videos_distinctness_ramps_linearly_with_norm(self):
        regions = [
            TemporalRegion(
                id="r1", session_id="s", event_id="e1", video_id="video",
                start_seconds=0.0, end_seconds=0.0, candidate_ids=("c1",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
            TemporalRegion(
                id="r2", session_id="s", event_id="e2", video_id="video",
                start_seconds=1.5, end_seconds=1.5, candidate_ids=("c2",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
        ]
        parameters = VideoPriorityHyperparameters(distinctness_norm_seconds=3.0)
        priorities = prioritize_videos(regions, ["e1", "e2"], parameters)
        self.assertAlmostEqual(priorities[0].distinctness, 0.5)

    def test_prioritize_videos_distinctness_single_covered_event_is_fully_distinct(self):
        """Nothing to compare with one covered event - trivially distinct,
        not penalized for missing coverage the (separate) coverage term
        already accounts for."""
        regions = [
            TemporalRegion(
                id="r1", session_id="s", event_id="e1", video_id="video",
                start_seconds=9.0, end_seconds=9.0, candidate_ids=("c1",),
                raw_coarse_score=0.9, normalized_coarse_score=0.9,
            ),
        ]
        priorities = prioritize_videos(regions, ["e1", "e2"])
        self.assertEqual(priorities[0].distinctness, 1.0)


if __name__ == "__main__":
    unittest.main()
