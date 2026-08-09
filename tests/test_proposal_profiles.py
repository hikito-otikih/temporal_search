import unittest

from adaptive_search.models import (
    BoundaryHyperparameters,
    EventDefinition,
    FrameScoreSample,
)
from adaptive_search.proposal_profiles import generate_profiled_proposals


def sample(event_id: str, timestamp: float, anchor: float) -> FrameScoreSample:
    return FrameScoreSample(
        session_id="session",
        event_id=event_id,
        video_id="video",
        region_id=f"region-{event_id}",
        frame_id=int(timestamp * 10),
        timestamp_seconds=timestamp,
        raw_anchor_score=anchor,
        raw_pre_score=0.0,
        raw_post_score=0.0,
        raw_motion_score=0.0,
    )


class ProposalProfileTests(unittest.TestCase):
    def setUp(self):
        self.parameters = BoundaryHyperparameters(
            window_options_seconds=(1.0,),
            min_samples_per_side=1,
            nms_radius_seconds=0.0,
            max_proposals_per_region=10,
        )

    def test_state_profile_prefers_center_of_persistent_anchor_plateau(self):
        event = EventDefinition(
            event_id="state",
            original_query="person is seated",
            anchor_query="a seated person",
            boundary_type="state",
        )
        anchors = [0.0, 3.0, 3.0, 3.0, 0.0]
        proposals = generate_profiled_proposals(
            [event],
            [sample("state", float(index), value) for index, value in enumerate(anchors)],
            self.parameters,
        )

        best = max(proposals, key=lambda item: item.final_event_score)
        self.assertEqual(best.timestamp_seconds, 2.0)
        self.assertEqual(best.left_window_seconds, 1.0)
        self.assertAlmostEqual(
            best.pre_consistency_score,
            best.post_persistence_score,
        )

    def test_symmetric_peak_profile_prefers_isolated_center_spike(self):
        event = EventDefinition(
            event_id="peak",
            original_query="camera flash",
            anchor_query="a bright camera flash",
            boundary_type="symmetric_peak",
        )
        anchors = [0.0, 0.0, 5.0, 0.0, 0.0]
        proposals = generate_profiled_proposals(
            [event],
            [sample("peak", float(index), value) for index, value in enumerate(anchors)],
            self.parameters,
        )

        best = max(proposals, key=lambda item: item.final_event_score)
        self.assertEqual(best.timestamp_seconds, 2.0)
        self.assertGreater(best.raw_boundary_score, 0.0)
        self.assertGreater(best.pre_consistency_score, 0.0)
        self.assertGreater(best.post_persistence_score, 0.0)

    def test_unknown_profile_falls_back_to_transition_with_pre_and_post(self):
        event = EventDefinition(
            event_id="transition",
            original_query="door opens",
            anchor_query="an opening door",
            pre_state="door is closed",
            post_state="door is open",
            boundary_type="unknown",
        )
        samples = [
            FrameScoreSample(
                session_id="session",
                event_id="transition",
                video_id="video",
                region_id="region-transition",
                frame_id=index * 10,
                timestamp_seconds=float(index),
                raw_anchor_score=3.0 if index == 2 else 0.0,
                raw_pre_score=3.0 if index < 2 else -3.0,
                raw_post_score=-3.0 if index < 2 else 3.0,
            )
            for index in range(5)
        ]

        proposals = generate_profiled_proposals(
            [event], samples, self.parameters
        )
        best = max(proposals, key=lambda item: item.final_event_score)
        self.assertEqual(best.timestamp_seconds, 2.0)
        self.assertGreater(best.pre_consistency_score, 0.9)
        self.assertGreater(best.post_persistence_score, 0.9)


if __name__ == "__main__":
    unittest.main()
