"""Regression test for a real reported bug: _search_one called urlopen with
no timeout at all, so a hung upstream frame-search API blocked the request
indefinitely."""

import json
import unittest
from io import BytesIO
from unittest.mock import patch

from legacy_search.client import UPSTREAM_SEARCH_TIMEOUT_SECONDS, search_queries


class SearchQueriesTimeoutTests(unittest.TestCase):
    def test_urlopen_is_called_with_a_bounded_timeout(self) -> None:
        response_body = json.dumps({"query": "cat", "results": []}).encode("utf-8")

        class _FakeResponse(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        with patch("legacy_search.client.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(response_body)
            search_queries("cat", 5)

        self.assertEqual(mock_urlopen.call_count, 1)
        _, kwargs = mock_urlopen.call_args
        self.assertEqual(kwargs.get("timeout"), UPSTREAM_SEARCH_TIMEOUT_SECONDS)
        self.assertIsNotNone(kwargs.get("timeout"))


if __name__ == "__main__":
    unittest.main()
