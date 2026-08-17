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
response shape carry the right fields, and - the one piece of real wiring
logic that lives in router.py itself, not in tuple_ranking.py - does
boundary refinement seed from the winning tuple's own per-event timestamps.
"""

import unittest

import numpy as np
from fastapi.testclient import TestClient

from adaptive_search.algorithms import prioritize_videos
from adaptive_search.dependencies import adaptive_service, boundary_refinement_runtime
from adaptive_search.embedding import l2_normalize
from adaptive_search.providers import FrameProviderCapabilities, FrameReference
from main import app


class FakeFrameProvider:
    def __init__(self, known_video_ids=("video_full",)):
        self._known = set(known_video_ids)

    def capabilities(self):
        return FrameProviderCapabilities(
            available=True, supports_batch=True, supports_dense_sampling=True, supports_thumbnails=False
        )

    def get_frames(self, video_id, pts_ms):
        return [self._frame(video_id, pts) for pts in pts_ms]

    @staticmethod
    def _frame(video_id, pts_ms):
        phase = (pts_ms // 1000) % 3
        image = np.eye(3, dtype=np.float32)[phase]
        return FrameReference(video_id=video_id, pts_ms=pts_ms, frame_index=pts_ms // 40, image=image)

    @property
    def catalog(self):
        return _FakeCatalog(self._known)


class _FakeMetadata:
    def __init__(self, fps=30.0, duration_ms=600_000):
        self.fps = fps
        self.duration_ms = duration_ms


class _FakeCatalog:
    def __init__(self, known_video_ids):
        self._known = known_video_ids

    def metadata(self, video_id):
        return _FakeMetadata() if video_id in self._known else None


class FakeEmbedder:
    model_id = "fake/siglip"
    revision = "commit-abc123"
    output_dimension = 3
    preprocess_version = "fake-rgb-v1"
    instruction = ""

    def encode_texts(self, texts):
        return np.eye(3, dtype=np.float32)

    def encode_images(self, images):
        return l2_normalize(np.asarray(images, dtype=np.float32))


def event(event_id, text, *, temporal_relation=None, reference_event_id=None):
    payload = {"event_id": event_id, "original_query": text, "anchor_query": text}
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
        "query_variant": f"{event_id}-anchor-en-0",
    }


class VideoPrioritiesTupleRankingTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)
        self._original_provider = boundary_refinement_runtime.frame_provider
        self._original_embedder = boundary_refinement_runtime.embedder

    def tearDown(self):
        self.client.close()
        boundary_refinement_runtime.frame_provider = self._original_provider
        boundary_refinement_runtime.embedder = self._original_embedder

    def _create_session(self, events, tuple_ranking_overrides=None):
        payload = {"events": events}
        if tuple_ranking_overrides is not None:
            payload["hyperparameters"] = {"tuple_ranking": tuple_ranking_overrides}
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
        # Smoke test at the actual shipped configuration (order_weight=0.8,
        # confidence_gate="threshold"@0.5) - precise score/order assertions
        # for this config are covered at the pure-function level; here we
        # only need "the real HTTP+RRF+region pipeline feeding into it
        # doesn't blow up and returns a well-formed response."
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

    def test_boundary_refinement_seeds_from_the_winning_tuple(self):
        # e1's best-scoring candidate (10.0) and e2's best-scoring candidate
        # (5.0) are chronologically INVERTED (e1 after e2) - an independent,
        # order-blind pick would use exactly this inverted pair. A large,
        # ungated order_weight must make the winning tuple instead prefer
        # e1=10.0 paired with e2's WORSE-scored but chronologically-later
        # candidate (60.0), since 10.0 < 60.0 is in order. relative_delta=1.0
        # ensures the worse candidate isn't pooled out before it gets a
        # chance to be chosen.
        session_id = self._create_session(
            [event("e1", "cut onion"), event("e2", "fry onion")],
            tuple_ranking_overrides={
                "relative_delta": 1.0,
                "order_weight": 0.8,
                "confidence_gate": "none",
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

        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=["video_full"])
        boundary_refinement_runtime.embedder = FakeEmbedder()

        response = self.client.get(
            f"/v1/search-sessions/{session_id}/video-priorities",
            params={"apply_boundary_refinement": "true"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        events = {
            e["event_id"]: e for e in response.json()["items"][0]["boundary_refinement"]["events"]
        }
        self.assertEqual(events["e1"]["anchor_seconds"], 10.0)
        self.assertEqual(events["e2"]["anchor_seconds"], 60.0)
        self.assertEqual(events["e1"]["source"], "tuple_ranking")
        self.assertEqual(events["e2"]["source"], "tuple_ranking")

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
        independent = prioritize_videos(
            bundle.artifacts.regions, ["e1", "e2"], bundle.session.hyperparameters.refinement,
        )[0]

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
        # is not what this session's relation actually claims).
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
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=["video_full"])
        boundary_refinement_runtime.embedder = FakeEmbedder()

        response = self.client.get(
            f"/v1/search-sessions/{session_id}/video-priorities",
            params={"apply_boundary_refinement": "true"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        events = {
            e["event_id"]: e for e in response.json()["items"][0]["boundary_refinement"]["events"]
        }
        self.assertEqual(events["e1"]["anchor_seconds"], 80.0)
        self.assertEqual(events["e2"]["anchor_seconds"], 50.0)

    def test_user_fixed_frame_still_wins_over_tuple_ranking(self):
        # commands/fix-frame's authoritative-constraint short-circuit must
        # keep taking priority over BOTH independent selection AND tuple
        # ranking - a user's explicit correction shouldn't be second-guessed
        # by either automatic mechanism.
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
        boundary_refinement_runtime.frame_provider = FakeFrameProvider(known_video_ids=["video_full"])
        boundary_refinement_runtime.embedder = FakeEmbedder()

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

        response = self.client.get(
            f"/v1/search-sessions/{session_id}/video-priorities",
            params={"apply_boundary_refinement": "true"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        events = {e["event_id"]: e for e in response.json()["items"][0]["boundary_refinement"]["events"]}
        self.assertEqual(events["e1"]["source"], "user_fixed")
        self.assertEqual(events["e1"]["anchor_seconds"], 42.0)


if __name__ == "__main__":
    unittest.main()
