import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.search_rewrite_preview import get_authoritative_analysis
from state import keys as K


class GetAuthoritativeAnalysisTests(unittest.TestCase):
    # This is the exact logic that decides whether Search reuses a preview
    # (create_session_from_rewrite, zero extra LLM calls) or falls back to
    # rewriting from scratch (create_session_from_queries) - getting the
    # staleness check wrong either silently reuses a preview that no longer
    # matches the query, or wastes a perfectly good, still-fresh preview.

    def test_no_preview_ever_taken_returns_none(self):
        store = {}
        self.assertIsNone(get_authoritative_analysis(store, "some query"))

    def test_fresh_preview_matching_current_text_is_returned(self):
        analysis = {"events": [{"event_id": 0}]}
        store = {
            K.SEARCH_REWRITE_PREVIEW: analysis,
            K.SEARCH_REWRITE_PREVIEW_INPUT: "cuts an onion",
        }
        self.assertIs(get_authoritative_analysis(store, "cuts an onion"), analysis)

    def test_stale_preview_edited_text_since_returns_none(self):
        store = {
            K.SEARCH_REWRITE_PREVIEW: {"events": []},
            K.SEARCH_REWRITE_PREVIEW_INPUT: "cuts an onion",
        }
        self.assertIsNone(get_authoritative_analysis(store, "slices a tomato"))

    def test_preview_error_left_analysis_none_is_treated_as_no_preview(self):
        store = {
            K.SEARCH_REWRITE_PREVIEW: None,
            K.SEARCH_REWRITE_PREVIEW_INPUT: "cuts an onion",
        }
        self.assertIsNone(get_authoritative_analysis(store, "cuts an onion"))


if __name__ == "__main__":
    unittest.main()
