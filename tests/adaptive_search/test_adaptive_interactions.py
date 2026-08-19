import unittest

from adaptive_search.schemas import EventConstraint, EventDefinition, SearchConstraints
from adaptive_search.service import AdaptiveSearchService


class AdaptiveInteractionTests(unittest.TestCase):
    def test_refixing_and_clearing_frame_keeps_proposal_status_consistent(self):
        service = AdaptiveSearchService()
        bundle = service.create_session(
            events=[
                EventDefinition(
                    event_id="e1",
                    original_query="a target state",
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
            [item.frame_id for item in bundle.artifacts.proposals if item.user_status == "fixed"],
            [10],
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
            [item.frame_id for item in bundle.artifacts.proposals if item.user_status == "fixed"],
            [20],
        )

        bundle, _ = service.clear_event_constraint(
            session_id,
            expected_revision=2,
            event_id="e1",
        )
        self.assertFalse(
            any(item.user_status == "fixed" for item in bundle.artifacts.proposals)
        )
        self.assertNotIn("e1", bundle.session.constraints.event_constraints)

    def test_clearing_a_fix_via_raw_constraint_replace_also_resets_proposal_status(self):
        # Regression test for a real reported bug: replace_constraints()
        # (the generic PUT .../constraints path) bypassed
        # _clear_fixed_proposal_status entirely - only fix_frame() and
        # clear_event_constraint() (the dedicated command endpoints) called
        # it. A caller that cleared an event's fixed_video_id via a raw PUT
        # instead of commands/clear-constraint left GET /proposals reporting
        # a stale user_status="fixed" for a pin that no longer existed.
        service = AdaptiveSearchService()
        bundle = service.create_session(
            events=[
                EventDefinition(
                    event_id="e1",
                    original_query="a target state",
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
            [item.frame_id for item in bundle.artifacts.proposals if item.user_status == "fixed"],
            [10],
        )

        bundle, _ = service.replace_constraints(
            session_id,
            expected_revision=1,
            constraints=SearchConstraints(),
        )
        self.assertFalse(
            any(item.user_status == "fixed" for item in bundle.artifacts.proposals)
        )
        self.assertNotIn("e1", bundle.session.constraints.event_constraints)

    def test_replacing_a_fix_with_a_different_one_via_raw_constraint_replace_also_resets_proposal_status(self):
        # Regression test for a real reported bug: the fix above only
        # cleared _clear_fixed_proposal_status when the pin was removed
        # entirely (updated fixed_video_id is None) - but a raw PUT can
        # also *replace* a pin with a different one (different video,
        # region, frame, or timestamp) without ever removing it. That case
        # compared only "is there still some pin" rather than "is it still
        # the same pin", so the stale proposal from the old pin kept
        # reporting user_status="fixed" even though the constraint no
        # longer named it.
        service = AdaptiveSearchService()
        bundle = service.create_session(
            events=[
                EventDefinition(
                    event_id="e1",
                    original_query="a target state",
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
            [item.frame_id for item in bundle.artifacts.proposals if item.user_status == "fixed"],
            [10],
        )

        bundle, _ = service.replace_constraints(
            session_id,
            expected_revision=1,
            constraints=SearchConstraints(
                event_constraints={
                    "e1": EventConstraint(
                        fixed_video_id="video2",
                        fixed_frame_id=99,
                        fixed_timestamp_seconds=5.0,
                    )
                }
            ),
        )
        self.assertFalse(
            any(item.user_status == "fixed" for item in bundle.artifacts.proposals),
            "the old video/frame-10 proposal should no longer be marked fixed",
        )
        self.assertEqual(
            bundle.session.constraints.event_constraints["e1"].fixed_video_id, "video2"
        )


if __name__ == "__main__":
    unittest.main()
