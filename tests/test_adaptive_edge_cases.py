import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from adaptive_search.api import adaptive_service
from adaptive_search.models import (
    ClusteringHyperparameters,
    RetrievalHyperparameters,
    SparseCandidate,
)
from adaptive_search.algorithms import cluster_temporal_regions
from adaptive_search.retrieval import fuse_candidates_rrf
from app import app


def sparse(candidate_id, event_id, variant, frame_id, score):
    return SparseCandidate(
        id=candidate_id,
        session_id="session",
        event_id=event_id,
        video_id="video",
        frame_id=frame_id,
        timestamp_seconds=float(frame_id),
        raw_relevance_score=float(score),
        query_variant=variant,
    )


class AdaptiveEdgeCaseTests(unittest.TestCase):
    def test_rrf_fuses_variants_without_creating_extra_events(self):
        fused = fuse_candidates_rrf(
            [
                sparse("a1", "e1", "vi-0", 1, 0.9),
                sparse("a2", "e1", "en-0", 1, 50.0),
                sparse("b1", "e1", "vi-0", 2, 0.8),
                sparse("b2", "e1", "en-0", 2, 49.0),
                sparse("c1", "e2", "vi-0", 3, 0.7),
            ],
            RetrievalHyperparameters(
                top_n_per_variant=10,
                top_n_fused=10,
                rrf_k=60,
                query_variants_per_event=4,
            ),
        )

        self.assertEqual(len(fused), 3)
        self.assertEqual([item.event_id for item in fused].count("e1"), 2)
        e1_frame_one = next(
            item for item in fused if item.event_id == "e1" and item.frame_id == 1
        )
        self.assertEqual(e1_frame_one.query_variant, "rrf:en-0,vi-0")
        self.assertNotEqual(e1_frame_one.raw_relevance_score, (0.9 + 50.0) / 2)

    def test_region_chain_is_split_and_not_remerged_past_max_span(self):
        candidates = [
            sparse(f"c{frame}", "e1", "rrf", frame, 1.0)
            for frame in range(21)
        ]
        regions = cluster_temporal_regions(
            candidates,
            ClusteringHyperparameters(
                gap_seconds=2.0,
                margin_seconds=1.0,
                max_region_seconds=10.0,
            ),
        )

        self.assertGreater(len(regions), 1)
        self.assertTrue(
            all(
                region.end_seconds - region.start_seconds <= 10.0
                for region in regions
            )
        )

    def test_region_limit_must_leave_room_for_both_margins(self):
        with self.assertRaises(ValidationError):
            ClusteringHyperparameters(
                gap_seconds=1.0,
                margin_seconds=3.0,
                max_region_seconds=5.0,
            )

    def test_candidate_calibration_is_independent_per_event(self):
        regions = cluster_temporal_regions(
            [
                sparse("a", "e1", "rrf", 1, 100.0),
                sparse("b", "e1", "rrf", 2, 101.0),
                sparse("c", "e2", "rrf", 1, 0.0),
                sparse("d", "e2", "rrf", 2, 1.0),
            ],
            ClusteringHyperparameters(
                gap_seconds=3.0,
                margin_seconds=0.0,
                max_region_seconds=10.0,
            ),
        )
        scores = {region.event_id: region.normalized_coarse_score for region in regions}
        self.assertAlmostEqual(scores["e1"], scores["e2"])


class AdaptiveNoOpApiTests(unittest.TestCase):
    def setUp(self):
        adaptive_service.repository.clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_identical_patches_do_not_increment_revision(self):
        created = self.client.post(
            "/v1/search-sessions",
            json={
                "events": [
                    {
                        "event_id": "e1",
                        "original_query": "event",
                        "anchor_query": "event",
                    }
                ]
            },
        )
        self.assertEqual(created.status_code, 201, created.text)
        session_id = created.json()["session"]["id"]

        event_patch = self.client.patch(
            f"/v1/search-sessions/{session_id}/events/e1",
            json={
                "expected_revision": 0,
                "patch": {"anchor_query": "event"},
            },
        )
        self.assertEqual(event_patch.status_code, 200, event_patch.text)
        self.assertEqual(event_patch.json()["session_revision"], 0)
        self.assertEqual(event_patch.json()["invalidated_stages"], [])

        parameter_patch = self.client.patch(
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={
                "expected_revision": 0,
                "patch": {"ranking": {"gap_lambda": 0.01}},
            },
        )
        self.assertEqual(parameter_patch.status_code, 200, parameter_patch.text)
        self.assertEqual(parameter_patch.json()["session_revision"], 0)
        self.assertEqual(parameter_patch.json()["invalidated_stages"], [])

    def test_legacy_query_rejects_whitespace_only_text(self):
        response = self.client.post("/temporal-search", json={"query": ["   "]})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
