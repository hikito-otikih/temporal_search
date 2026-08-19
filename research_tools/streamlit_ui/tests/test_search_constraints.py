import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.search_constraints import (
    confirm_fixed_frame,
    patch_constraints,
    prioritize_item,
    reject_item,
)
from models.ui_models import SessionResponse
from services.api_client import ApiError
from state import keys as K


class FakeClient:
    def __init__(self, *, session=None, video_priorities=None, fail_replace=False):
        self.session = session or {"id": "s1", "revision": 3, "constraints": {}}
        self.video_priorities = video_priorities or {"items": [], "boundary_refinement_capability": {}}
        self.fail_replace = fail_replace
        self.replace_calls: list[dict] = []
        self.fix_frame_calls: list[dict] = []

    def get_session(self, session_id):
        return SessionResponse(session=self.session)

    def replace_constraints(self, session_id, *, expected_revision, constraints):
        if self.fail_replace:
            raise ApiError("boom")
        self.replace_calls.append(
            {"session_id": session_id, "expected_revision": expected_revision, "constraints": constraints}
        )

    def get_video_priorities(self, session_id, *, limit, apply_boundary_refinement):
        return self.video_priorities

    def fix_frame(self, session_id, *, expected_revision, event_id, video_id, frame_id, timestamp_seconds, region_id=None):
        self.fix_frame_calls.append(
            {
                "session_id": session_id,
                "expected_revision": expected_revision,
                "event_id": event_id,
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp_seconds": timestamp_seconds,
                "region_id": region_id,
            }
        )


class PatchConstraintsTests(unittest.TestCase):
    def test_merges_current_constraints_with_updater_result(self):
        client = FakeClient(
            session={
                "id": "s1", "revision": 5,
                "constraints": {"rejected_video_ids": ["v0"], "prioritized_video_ids": []},
            }
        )
        patch_constraints(client, "s1", lambda c: {"prioritized_video_ids": ["v1"]})
        self.assertEqual(len(client.replace_calls), 1)
        call = client.replace_calls[0]
        self.assertEqual(call["expected_revision"], 5)
        # The field the updater didn't touch survives - this is the whole
        # point over a naive partial PUT (which is a full REST replace).
        self.assertEqual(call["constraints"]["rejected_video_ids"], ["v0"])
        self.assertEqual(call["constraints"]["prioritized_video_ids"], ["v1"])


class RejectItemTests(unittest.TestCase):
    def test_legacy_pipeline_hides_locally_without_calling_the_client(self):
        store = {K.SEARCH_RESULTS: {"pipeline": "legacy_temporal", "session_id": None}}
        client = FakeClient()
        reject_item(store, client, "0:video_a")
        self.assertIn("0:video_a", store[K.SEARCH_REJECTED_IDS])
        self.assertEqual(client.replace_calls, [])

    def test_adaptive_reject_atomically_clears_prioritized_allowed_and_fixed(self):
        client = FakeClient(
            session={
                "id": "s1", "revision": 2,
                "constraints": {
                    "prioritized_video_ids": ["video_a"],
                    "allowed_video_ids": ["video_a", "video_b"],
                    "rejected_video_ids": [],
                    "event_constraints": {
                        "e1": {"fixed_video_id": "video_a", "fixed_region_id": "r1", "fixed_frame_id": 5, "fixed_timestamp_seconds": 1.0},
                        "e2": {"fixed_video_id": "video_c"},
                    },
                },
            },
            video_priorities={"items": [], "boundary_refinement_capability": {}},
        )
        store = {
            K.SEARCH_RESULTS: {
                "pipeline": "adaptive_coarse", "session_id": "s1",
                "display_limit": 20, "apply_refinement": False, "event_count": 1,
            }
        }

        reject_item(store, client, "video_a")

        sent = client.replace_calls[0]["constraints"]
        self.assertEqual(sent["rejected_video_ids"], ["video_a"])
        self.assertNotIn("video_a", sent["prioritized_video_ids"])
        self.assertNotIn("video_a", sent["allowed_video_ids"])
        self.assertIsNone(sent["event_constraints"]["e1"]["fixed_video_id"])
        self.assertIsNone(sent["event_constraints"]["e1"]["fixed_region_id"])
        # A different video's fix is untouched.
        self.assertEqual(sent["event_constraints"]["e2"]["fixed_video_id"], "video_c")
        self.assertIn("video_a", store[K.SEARCH_REJECTED_IDS])

    def test_failed_request_does_not_hide_the_item_locally(self):
        # Regression test for a real reported bug: the local hide used to
        # happen unconditionally before the server call, so a failed PATCH
        # still hid a result the server never actually rejected.
        client = FakeClient(
            session={"id": "s1", "revision": 1, "constraints": {}},
            fail_replace=True,
        )
        store = {
            K.SEARCH_RESULTS: {
                "pipeline": "adaptive_coarse", "session_id": "s1",
                "display_limit": 20, "apply_refinement": False, "event_count": 1,
            }
        }
        reject_item(store, client, "video_a")
        self.assertNotIn(K.SEARCH_REJECTED_IDS, store)
        self.assertEqual(store[K.SEARCH_ERROR][0], "api")


class PrioritizeItemTests(unittest.TestCase):
    def test_moves_video_to_front_and_clears_its_own_rejection(self):
        client = FakeClient(
            session={
                "id": "s1", "revision": 1,
                "constraints": {
                    "prioritized_video_ids": ["video_b"],
                    "rejected_video_ids": ["video_a"],
                },
            },
        )
        store = {
            K.SEARCH_RESULTS: {
                "pipeline": "adaptive_coarse", "session_id": "s1",
                "display_limit": 20, "apply_refinement": False, "event_count": 1,
            }
        }
        prioritize_item(store, client, "video_a")
        sent = client.replace_calls[0]["constraints"]
        self.assertEqual(sent["prioritized_video_ids"], ["video_a", "video_b"])
        self.assertNotIn("video_a", sent["rejected_video_ids"])

    def test_no_op_outside_an_adaptive_session(self):
        store = {K.SEARCH_RESULTS: {"pipeline": "legacy_temporal", "session_id": None}}
        client = FakeClient()
        prioritize_item(store, client, "video_a")
        self.assertEqual(client.replace_calls, [])


class ConfirmFixedFrameTests(unittest.TestCase):
    def test_region_id_is_passed_through_for_provenance(self):
        client = FakeClient(session={"id": "s1", "revision": 4, "constraints": {}})
        store = {}
        confirm_fixed_frame(
            store, client,
            session_id="s1", event_id="e1", video_id="video_a",
            frame={"pts_ms": 5000, "timestamp_seconds": 5.0},
            region_id="region-42",
        )
        self.assertEqual(client.fix_frame_calls[0]["region_id"], "region-42")
        self.assertEqual(client.fix_frame_calls[0]["expected_revision"], 4)

    def test_defaults_to_no_region_id_for_the_raw_frame_fixer(self):
        client = FakeClient(session={"id": "s1", "revision": 0, "constraints": {}})
        confirm_fixed_frame(
            {}, client,
            session_id="s1", event_id="e1", video_id="video_a",
            frame={"pts_ms": 1000, "timestamp_seconds": 1.0},
        )
        self.assertIsNone(client.fix_frame_calls[0]["region_id"])

    def test_refreshes_search_results_only_for_the_matching_session(self):
        client = FakeClient(
            session={"id": "s1", "revision": 0, "constraints": {}},
            video_priorities={"items": [{"video_id": "video_a", "priority_score": 0.9, "event_coverage": 1}], "boundary_refinement_capability": {"requested": False}},
        )
        store = {
            K.SEARCH_RESULTS: {
                "pipeline": "adaptive_coarse", "session_id": "s1",
                "display_limit": 20, "apply_refinement": False, "event_count": 1,
            }
        }
        confirm_fixed_frame(
            store, client,
            session_id="s1", event_id="e1", video_id="video_a",
            frame={"pts_ms": 1000, "timestamp_seconds": 1.0},
        )
        self.assertEqual(len(store[K.SEARCH_RESULTS]["items"]), 1)


if __name__ == "__main__":
    unittest.main()
