"""README.md's illustrative /rewrite response must stay valid against the
real schema - regression test for a stale example that had one event for
two input queries and was missing required fields (anchor_query, pre_state,
post_state)."""

import json
import unittest
from pathlib import Path

from rewrite.schemas import RewriteResponse

README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def _extract_json_block_after(readme_text: str, marker: str) -> dict:
    marker_index = readme_text.index(marker)
    fence = "```json"
    start = readme_text.index(fence, marker_index) + len(fence)
    end = readme_text.index("```", start)
    return json.loads(readme_text[start:end])


class ReadmeRewriteExampleTests(unittest.TestCase):
    def test_illustrative_response_validates_and_matches_request_event_count(self) -> None:
        readme_text = README_PATH.read_text(encoding="utf-8")
        request_payload = _extract_json_block_after(readme_text, "Example request")
        response_payload = _extract_json_block_after(readme_text, "Illustrative response")

        response = RewriteResponse.model_validate(response_payload)

        self.assertEqual(len(response.events), len(request_payload["query"]))
        for index, (query, event) in enumerate(
            zip(request_payload["query"], response.events)
        ):
            self.assertEqual(event.event_id, index)
            self.assertEqual(event.original_query, query)


if __name__ == "__main__":
    unittest.main()
