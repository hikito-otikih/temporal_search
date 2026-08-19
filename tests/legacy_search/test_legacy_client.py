"""Regression test for a real reported bug: _search_one called urlopen with
no timeout at all, so a hung upstream frame-search API blocked the request
indefinitely."""

import json
import unittest
from io import BytesIO
from unittest.mock import patch

from adaptive_search.exceptions import UpstreamSearchError
from legacy_search.client import UPSTREAM_SEARCH_TIMEOUT_SECONDS, search_queries


class _FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _ResetOnReadResponse:
    """Simulates a connection that resets mid-body-read, after urlopen()
    already returned a response object - the failure surfaces from
    response.read(), not from urlopen() itself."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        raise ConnectionResetError("connection reset by peer")


class SearchQueriesTimeoutTests(unittest.TestCase):
    def test_urlopen_is_called_with_a_bounded_timeout(self) -> None:
        response_body = json.dumps({"query": "cat", "results": []}).encode("utf-8")

        with patch("legacy_search.client.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(response_body)
            search_queries("cat", 5)

        self.assertEqual(mock_urlopen.call_count, 1)
        _, kwargs = mock_urlopen.call_args
        self.assertEqual(kwargs.get("timeout"), UPSTREAM_SEARCH_TIMEOUT_SECONDS)
        self.assertIsNotNone(kwargs.get("timeout"))


class SearchQueriesUpstreamErrorClassificationTests(unittest.TestCase):
    # Regression test for a real reported bug: invalid UTF-8 and a
    # schema-mismatched (but otherwise valid) JSON response both left an
    # uncaught UnicodeDecodeError/pydantic ValidationError propagating to
    # _raise_api_error's generic fallback, which maps both to a 422 client
    # input error - misreporting a real *upstream* problem as if this
    # service's own request were malformed. Both must now surface as
    # UpstreamSearchError (mapped to 502) instead.

    def test_invalid_utf8_is_reported_as_upstream_error_not_a_value_error(self) -> None:
        with patch("legacy_search.client.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(b"\xff\xfe not valid utf-8")
            with self.assertRaises(UpstreamSearchError):
                search_queries("cat", 5)

    def test_valid_json_with_wrong_schema_is_reported_as_upstream_error(self) -> None:
        # Valid JSON, but missing QueryResponse's required "results" field.
        response_body = json.dumps({"query": "cat"}).encode("utf-8")

        with patch("legacy_search.client.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(response_body)
            with self.assertRaises(UpstreamSearchError):
                search_queries("cat", 5)

    def test_connection_reset_during_read_is_reported_as_upstream_error(self) -> None:
        # Regression test for a real reported bug: ConnectionResetError
        # during response.read() is a plain OSError subclass, not wrapped in
        # URLError like most urllib failures - it used to propagate
        # uncaught into _raise_api_error's generic fallback (which doesn't
        # recognize it either), producing an unstructured 500 instead of the
        # 502 every other upstream failure mode gets.
        with patch("legacy_search.client.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _ResetOnReadResponse()
            with self.assertRaises(UpstreamSearchError):
                search_queries("cat", 5)


if __name__ == "__main__":
    unittest.main()
