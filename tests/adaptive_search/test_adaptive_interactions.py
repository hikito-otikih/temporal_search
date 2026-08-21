import unittest

from adaptive_search.schemas import EventDefinition
from adaptive_search.service import AdaptiveSearchService


class AdaptiveInteractionTests(unittest.TestCase):
    def test_refixing_and_clearing_frame_updates_event_constraints(self):
        service = AdaptiveSearchService()
        bundle = service.create_session(
            events=[
                EventDefinition(
                    event_id="e1",
                    anchor_query="the target state",
                )
            ]
        )
        session_id = bundle.session.id

        bundle, _ = service.fix_frame(
            session_id,
            expected_revision=0,
            event_id="e1",
            video_id="video",
            frame_id=10,
            timestamp_seconds=1.0,
        )
        self.assertEqual(
            bundle.session.constraints.event_constraints["e1"].fixed_frame_id, 10
        )

        bundle, _ = service.fix_frame(
            session_id,
            expected_revision=1,
            event_id="e1",
            video_id="video",
            frame_id=20,
            timestamp_seconds=2.0,
        )
        self.assertEqual(
            bundle.session.constraints.event_constraints["e1"].fixed_frame_id, 20
        )

        bundle, _ = service.clear_event_constraint(
            session_id,
            expected_revision=2,
            event_id="e1",
        )
        self.assertNotIn("e1", bundle.session.constraints.event_constraints)


if __name__ == "__main__":
    unittest.main()
