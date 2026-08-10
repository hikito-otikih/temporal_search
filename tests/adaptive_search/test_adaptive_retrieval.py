import unittest

from adaptive_search.schemas import RetrievalHyperparameters, SparseCandidate
from adaptive_search.retrieval import fuse_candidates_rrf


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_variant_count_is_bounded_per_event(self):
        candidates = [
            SparseCandidate(
                id=f"c{index}",
                session_id="session",
                event_id="event",
                video_id="video",
                frame_id=index,
                timestamp_seconds=float(index),
                raw_relevance_score=1.0,
                query_variant=f"variant-{index}",
            )
            for index in range(3)
        ]

        with self.assertRaisesRegex(ValueError, "maximum is 2"):
            fuse_candidates_rrf(
                candidates,
                RetrievalHyperparameters(query_variants_per_event=2),
            )


if __name__ == "__main__":
    unittest.main()
