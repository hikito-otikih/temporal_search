import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ui_models import MutationResponse, SessionResponse
from services.api_client import ApiError, RevisionConflictError
from state import keys as K
from state import session_state as SS


class FakeClient:
    def __init__(self, *, get_session_response=None):
        self.get_session_response = get_session_response
        self.get_session_calls = 0

    def get_session(self, session_id):
        self.get_session_calls += 1
        if self.get_session_response is None:
            raise ApiError("backend down")
        return self.get_session_response


def session_response(revision: int) -> SessionResponse:
    return SessionResponse(
        session={
            "id": "session-1",
            "revision": revision,
            "status": "ready",
            "event_count": 2,
            "hyperparameters": {},
        }
    )


class StateTransitionTests(unittest.TestCase):
    def setUp(self):
        self.store = {}
        SS.init_ui_state(self.store)

    def test_init_sets_defaults(self):
        self.assertEqual(self.store[K.BACKEND_URL], "http://127.0.0.1:8001")
        self.assertIsNone(self.store[K.ACTIVE_SESSION_ID])
        self.assertEqual(self.store[K.SESSION_REVISION], 0)

    def test_set_active_session_stores_revision_and_response(self):
        SS.set_active_session(self.store, session_response(3))
        self.assertEqual(self.store[K.ACTIVE_SESSION_ID], "session-1")
        self.assertEqual(self.store[K.SESSION_REVISION], 3)
        self.assertIsInstance(self.store[K.SESSION_RESPONSE], SessionResponse)

    def test_apply_mutation_success_updates_revision(self):
        SS.set_active_session(self.store, session_response(0))

        def mutation(expected_revision):
            self.assertEqual(expected_revision, 0)
            return MutationResponse(session_revision=1, invalidated_stages=["tuple"])

        result, error = SS.apply_mutation(self.store, FakeClient(), mutation)
        self.assertIsNone(error)
        self.assertEqual(result.session_revision, 1)
        self.assertEqual(self.store[K.SESSION_REVISION], 1)

    def test_apply_mutation_conflict_refreshes_and_does_not_retry(self):
        refreshed = session_response(1)
        client = FakeClient(get_session_response=refreshed)
        SS.set_active_session(self.store, session_response(0))
        retried = {"count": 0}

        def mutation(expected_revision):
            retried["count"] += 1
            raise RevisionConflictError(expected_revision, 1)

        result, error = SS.apply_mutation(self.store, client, mutation)
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertIn("Revision conflict", error)
        self.assertEqual(retried["count"], 1, "conflict must not auto-retry")
        self.assertEqual(client.get_session_calls, 1, "session must be refreshed")
        self.assertEqual(self.store[K.SESSION_REVISION], 1)

    def test_apply_mutation_api_error_is_surfaced(self):
        SS.set_active_session(self.store, session_response(0))

        def mutation(expected_revision):
            raise ApiError("upstream exploded")

        result, error = SS.apply_mutation(self.store, FakeClient(), mutation)
        self.assertIsNone(result)
        self.assertIn("upstream exploded", error)
        self.assertEqual(self.store[K.SESSION_REVISION], 0)

    def test_clear_active_session_resets_state(self):
        SS.set_active_session(self.store, session_response(2))
        SS.clear_active_session(self.store)
        self.assertIsNone(self.store[K.ACTIVE_SESSION_ID])
        self.assertEqual(self.store[K.SESSION_REVISION], 0)

    def test_refresh_session_updates_revision(self):
        SS.set_active_session(self.store, session_response(0))
        client = FakeClient(get_session_response=session_response(5))
        response = SS.refresh_session(self.store, client)
        self.assertEqual(response.session["revision"], 5)
        self.assertEqual(self.store[K.SESSION_REVISION], 5)


if __name__ == "__main__":
    unittest.main()
