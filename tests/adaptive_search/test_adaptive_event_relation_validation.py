"""create_session must reject an event relation graph with a dangling
reference_event_id or a temporal_relation cycle - both are silently
mishandled downstream otherwise (build_order_constraints drops a dangling
reference with no error, and a cycle's transitive closure produces
impossible self-precedence constraints (i, i) that no video can ever
satisfy). See tuple_ranking.py's build_order_constraints for the edge
semantics this validation mirrors.
"""

import unittest

from fastapi.testclient import TestClient

from adaptive_search.dependencies import adaptive_service
from adaptive_search.exceptions import AdaptiveInputError
from adaptive_search.schemas import EventDefinition
from adaptive_search.service import AdaptiveSearchService
from main import app


def _event(event_id, *, temporal_relation="unknown", reference_event_id=None):
    return EventDefinition(
        event_id=event_id,
        original_query=f"query for {event_id}",
        anchor_query=f"query for {event_id}",
        temporal_relation=temporal_relation,
        reference_event_id=reference_event_id,
    )


class EventRelationGraphValidationTests(unittest.TestCase):
    def setUp(self):
        self.service = AdaptiveSearchService()

    def test_dangling_reference_event_id_is_rejected(self):
        events = [_event("e1", temporal_relation="after", reference_event_id="ghost")]
        with self.assertRaisesRegex(AdaptiveInputError, "unknown reference_event_id"):
            self.service.create_session(events=events)

    def test_direct_two_event_cycle_is_rejected(self):
        events = [
            _event("e1", temporal_relation="before", reference_event_id="e2"),
            _event("e2", temporal_relation="before", reference_event_id="e1"),
        ]
        with self.assertRaisesRegex(AdaptiveInputError, "cycle"):
            self.service.create_session(events=events)

    def test_transitive_three_event_cycle_is_rejected(self):
        events = [
            _event("e1", temporal_relation="before", reference_event_id="e2"),
            _event("e2", temporal_relation="before", reference_event_id="e3"),
            _event("e3", temporal_relation="before", reference_event_id="e1"),
        ]
        with self.assertRaisesRegex(AdaptiveInputError, "cycle"):
            self.service.create_session(events=events)

    def test_acyclic_relation_graph_is_accepted(self):
        events = [
            _event("e1"),
            _event("e2", temporal_relation="after", reference_event_id="e1"),
            _event("e3", temporal_relation="after", reference_event_id="e2"),
        ]
        bundle = self.service.create_session(events=events)
        self.assertEqual(bundle.session.revision, 0)


class EventRelationGraphValidationHttpTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_two_event_cycle_returns_422_not_201(self):
        response = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "cut onion",
                        "anchor_query": "cut onion",
                        "temporal_relation": "before",
                        "reference_event_id": "e2",
                    },
                    {
                        "event_id": "e2",
                        "original_query": "fry onion",
                        "anchor_query": "fry onion",
                        "temporal_relation": "before",
                        "reference_event_id": "e1",
                    },
                ]
            },
        )
        self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
