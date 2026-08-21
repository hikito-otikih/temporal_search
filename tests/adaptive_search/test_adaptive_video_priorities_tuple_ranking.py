"""HTTP-level tests for GET /video-priorities.

Order-aware tuple ranking (`tuple_ranking.py`) is the only ranking algorithm
this endpoint runs - there is no more separate "flag on/off" mode to choose
between (Round 8: prioritize_videos()'s independent per-event argmax was
removed from this endpoint entirely; it remains real, tested, callable code,
just not reachable through this endpoint). Algorithm correctness (pooling
thresholds, order scoring, confidence gating, the order_weight=0 <->
prioritize_videos() equivalence) is covered at the pure-function level in
test_tuple_ranking.py against directly-constructed TemporalRegion/
SparseCandidate objects. These tests cover integration instead: does the
response shape carry the right fields, and does matched_frame_ids correctly
reflect the winning tuple's own per-event frame choice.
"""

import unittest

from fastapi.testclient import TestClient

from adaptive_search.algorithms import prioritize_videos
from adaptive_search.dependencies import adaptive_service
from main import app


def event(event_id, text, *, temporal_relation=None, reference_event_id=None):
    payload = {"event_id": event_id, "anchor_query": text}
    if temporal_relation is not None:
        payload["temporal_relation"] = temporal_relation
    if reference_event_id is not None:
        payload["reference_event_id"] = reference_event_id
    return payload


def candidate(candidate_id, session_id, event_id, video_id, timestamp_seconds, raw_relevance_score):
    return {
        "id": candidate_id,
        "session_id": session_id,
        "event_id": event_id,
        "video_id": video_id,
        "frame_id": int(timestamp_seconds * 30),
        "timestamp_seconds": timestamp_seconds,
        "raw_relevance_score": raw_relevance_score,
    }


class VideoPrioritiesTupleRankingTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def _create_session(self, events, tuple_ranking_overrides=None):
        payload = {"events": events}
        hyperparameters = {}
        if tuple_ranking_overrides is not None:
            hyperparameters["tuple_ranking"] = tuple_ranking_overrides
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        response = self.client.post("/v1/search-sessions", json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["session"]["id"]

    def _submit_candidates(self, session_id, event_ids, candidates):
        response = self.client.post(
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={"expected_revision": 0, "event_ids": event_ids, "candidates": candidates},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_production_defaults_does_not_crash(self):
        # Smoke test at the actual shipped configuration (order_weight=1.2,
        # confidence_gate="threshold"@0.5) - precise score/order assertions
        # for this config are covered at the pure-function level; here we
        # only need "the real HTTP+region pipeline feeding into it doesn't
        # blow up and returns a well-formed response."
        session_id = self._create_session([event("e1", "cut onion"), event("e2", "fry onion")])
        self._submit_candidates(
            session_id, ["e1", "e2"],
            [
                candidate("c1", session_id, "e1", "video_full", 10.0, 0.9),
                candidate("c2", session_id, "e1", "video_full", 50.0, 0.5),
                candidate("c3", session_id, "e2", "video_full", 5.0, 0.9),
                candidate("c4", session_id, "e2", "video_full", 60.0, 0.5),
            ],
        )
        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]
        self.assertEqual(item["video_id"], "video_full")
        self.assertTrue(0.0 <= item["priority_score"] <= 1.0)

    def test_prioritized_video_id_floats_to_front_without_changing_its_score(self):
        # video_a scores higher than video_b on pure retrieval evidence - a
        # naive "boost the score" implementation would be observable in
        # priority_score; prioritized_video_ids must reorder the response
        # only, leaving every video's own score untouched (distinct from
        # fix_frame, which is scoped to its own video and never reorders
        # videos against each other - see the PUT below, no event_constraints
        # involved at all).
        session_id = self._create_session([event("e1", "cut onion")])
        self._submit_candidates(
            session_id, ["e1"],
            [
                candidate("c1", session_id, "e1", "video_a", 10.0, 0.9),
                candidate("c2", session_id, "e1", "video_b", 20.0, 0.3),
            ],
        )
        baseline = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities").json()
        self.assertEqual(baseline["items"][0]["video_id"], "video_a")
        baseline_scores = {item["video_id"]: item["priority_score"] for item in baseline["items"]}

        session = self.client.get(f"/v1/search-sessions/{session_id}").json()
        revision = session["session"]["revision"]
        response = self.client.put(
            f"/v1/search-sessions/{session_id}/constraints",
            json={
                "expected_revision": revision,
                "constraints": {"prioritized_video_ids": ["video_b"]},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

        reordered = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities").json()
        self.assertEqual([item["video_id"] for item in reordered["items"]], ["video_b", "video_a"])
        reordered_scores = {item["video_id"]: item["priority_score"] for item in reordered["items"]}
        self.assertEqual(reordered_scores, baseline_scores)

    def test_fixed_frame_is_respected_by_ranking_itself(self):
        # commands/fix-frame's timestamp is arbitrary, with no candidate
        # anywhere near it, so the ranker itself must synthesize a region for
        # it - confirmed here via mean_best_event_score (should land at 1.0,
        # the synthesized region's confidence, not near the real 0.6
        # candidate's score).
        session_id = self._create_session([event("e1", "cut onion")])
        self._submit_candidates(
            session_id, ["e1"],
            [candidate("c1", session_id, "e1", "video_full", 10.0, 0.6)],
        )
        fix_response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/fix-frame",
            json={
                "expected_revision": 0,
                "event_id": "e1",
                "video_id": "video_full",
                "frame_id": 1260,
                "timestamp_seconds": 42.0,
            },
        )
        self.assertEqual(fix_response.status_code, 200, fix_response.text)

        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]
        self.assertEqual(item["video_id"], "video_full")
        self.assertAlmostEqual(item["mean_best_event_score"], 1.0)

    def test_distinctness_is_reported_but_does_not_affect_ranking(self):
        # distinctness is now a purely descriptive VideoPriority field, not a
        # ranking input at this endpoint (the old video_distinctness_weight
        # session hyperparameter was removed - it degenerated to a no-op
        # here regardless of its value, see video_priorities.py's comment).
        # video_a's two events collapse onto the same timestamp (distinctness
        # 0); video_b's are 90s apart (distinctness 1) - same region scores
        # otherwise. order_weight=0 removes any order contribution, and
        # default_adjacent_gap_lambda=0.0 isolates from the separate
        # default-on gap penalty (which would otherwise cost video_b's 90s
        # gap on its own, unrelated to distinctness) - with both off, the
        # raw tuple score is driven only by region_mean, identical for both.
        # distinctness must still differ, but priority_score must not.
        session_id = self._create_session(
            [event("e1", "cut onion"), event("e2", "fry onion")],
            tuple_ranking_overrides={
                "order_weight": 0.0,
                "default_adjacent_gap_lambda": 0.0,
            },
        )
        self._submit_candidates(
            session_id, ["e1", "e2"],
            [
                candidate("ca1", session_id, "e1", "video_a", 10.0, 0.9),
                candidate("ca2", session_id, "e2", "video_a", 10.0, 0.9),
                candidate("cb1", session_id, "e1", "video_b", 10.0, 0.9),
                candidate("cb2", session_id, "e2", "video_b", 100.0, 0.9),
            ],
        )
        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        items = {item["video_id"]: item for item in response.json()["items"]}
        self.assertEqual(items["video_a"]["distinctness"], 0.0)
        self.assertEqual(items["video_b"]["distinctness"], 1.0)
        self.assertAlmostEqual(
            items["video_a"]["priority_score"], items["video_b"]["priority_score"]
        )

    def test_video_priority_fields_reflect_the_actual_winning_tuple(self):
        # e2's winning tuple picks the order-satisfying 60.0s candidate
        # (score 0.5), not the independently-best 5.0s candidate (score
        # 0.9) prioritize_videos() would pick alone. mean_best_event_score
        # must reflect the tuple's own choice, not what independent
        # per-event argmax would say - computed directly against the same
        # real internal regions (not via HTTP: apply_tuple_ranking is no
        # longer a separate mode to call it through) as the ground truth
        # this response must differ from.
        session_id = self._create_session(
            [event("e1", "cut onion"), event("e2", "fry onion")],
            tuple_ranking_overrides={
                "relative_delta": 1.0,
                "order_weight": 0.8,
                "confidence_gate": "none",
                # Isolate from the default-on gap penalty - e1@10/e2@60's 50s
                # gap is irrelevant to what this test is actually about
                # (joint vs. independent-per-event picking); left on, it
                # outweighs the order bonus for this fixture's deliberately
                # narrow margin and flips the winner back to e2@5.0, which
                # defeats the point of the test.
                "default_adjacent_gap_lambda": 0.0,
            },
        )
        self._submit_candidates(
            session_id, ["e1", "e2"],
            [
                candidate("c1", session_id, "e1", "video_full", 10.0, 0.9),
                candidate("c3", session_id, "e2", "video_full", 5.0, 0.9),
                candidate("c4", session_id, "e2", "video_full", 60.0, 0.5),
            ],
        )

        bundle = adaptive_service.get_session(session_id)
        independent = prioritize_videos(bundle.artifacts.regions, ["e1", "e2"])[0]

        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]

        self.assertNotAlmostEqual(item["mean_best_event_score"], independent.mean_best_event_score)
        self.assertNotAlmostEqual(item["min_best_event_score"], independent.min_best_event_score)
        self.assertEqual(item["event_coverage"], 2)
        self.assertEqual(item["normalized_coverage"], 1.0)
        self.assertTrue(0.0 <= item["distinctness"] <= 1.0)

    def test_uses_real_temporal_relation_direction_not_list_position(self):
        # e1 is listed FIRST but is explicitly "after" e2 (relation direction
        # disagreeing with list position - exactly the scenario the list-
        # position-only implementation would have gotten backwards). e1 has
        # two candidates: 5.0 (best score, but chronologically BEFORE e2's
        # 50.0 - violates "e1 after e2") and 80.0 (worse score, but AFTER
        # e2's 50.0 - satisfies the relation). A relation-aware, ungated
        # order bonus should prefer 80.0; a list-position default would have
        # preferred 5.0 (treating "e1 then e2" as the expected order, which
        # is not what this session's relation actually claims). Verified via
        # matched_frame_ids, in event order [e1, e2]: frame_id = int(seconds
        # * 30), so 80.0 -> "2400" and 50.0 -> "1500".
        session_id = self._create_session(
            [
                event("e1", "cut onion", temporal_relation="after", reference_event_id="e2"),
                event("e2", "fry onion", temporal_relation="sequence_start"),
            ],
            tuple_ranking_overrides={"relative_delta": 1.0, "order_weight": 0.8, "confidence_gate": "none"},
        )
        self._submit_candidates(
            session_id, ["e1", "e2"],
            [
                candidate("c1", session_id, "e1", "video_full", 5.0, 0.9),
                candidate("c2", session_id, "e1", "video_full", 80.0, 0.5),
                candidate("c3", session_id, "e2", "video_full", 50.0, 0.9),
            ],
        )

        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]
        self.assertEqual(item["matched_frame_ids"], ["2400", "1500"])

    def test_user_fixed_frame_still_wins_over_tuple_ranking(self):
        # commands/fix-frame's authoritative-constraint short-circuit must
        # keep taking priority over BOTH independent selection AND tuple
        # ranking - a user's explicit correction shouldn't be second-guessed
        # by either automatic mechanism. Verified via matched_frame_ids: e1's
        # entry must be the fixed frame_id (1260), not either candidate's.
        session_id = self._create_session(
            [event("e1", "cut onion"), event("e2", "fry onion")],
            tuple_ranking_overrides={"relative_delta": 1.0, "order_weight": 0.8, "confidence_gate": "none"},
        )
        self._submit_candidates(
            session_id, ["e1", "e2"],
            [
                candidate("c1", session_id, "e1", "video_full", 10.0, 0.9),
                candidate("c3", session_id, "e2", "video_full", 5.0, 0.9),
                candidate("c4", session_id, "e2", "video_full", 60.0, 0.5),
            ],
        )

        fix_response = self.client.post(
            f"/v1/search-sessions/{session_id}/commands/fix-frame",
            json={
                "expected_revision": 0,
                "event_id": "e1",
                "video_id": "video_full",
                "frame_id": 1260,
                "timestamp_seconds": 42.0,
            },
        )
        self.assertEqual(fix_response.status_code, 200, fix_response.text)

        response = self.client.get(f"/v1/search-sessions/{session_id}/video-priorities")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]
        self.assertEqual(item["matched_frame_ids"][0], "1260")


if __name__ == "__main__":
    unittest.main()
