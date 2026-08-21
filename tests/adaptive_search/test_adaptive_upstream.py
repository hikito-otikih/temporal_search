import unittest

from adaptive_search.client import (
    QueryVariant,
    UpstreamSearchClient,
    UpstreamSearchError,
    normalize_video_id,
    parse_display_timestamp,
    variants_from_mapping,
)


class AdaptiveUpstreamTests(unittest.TestCase):
    def test_timestamp_and_video_id_normalization(self):
        self.assertEqual(parse_display_timestamp("7"), 7.0)
        self.assertEqual(parse_display_timestamp("1:07"), 67.0)
        self.assertEqual(parse_display_timestamp("1:02:03.5"), 3723.5)
        self.assertEqual(normalize_video_id(r"training\101\abc.mp4"), "abc")
        with self.assertRaises(UpstreamSearchError):
            parse_display_timestamp("1:61")

    def test_retrieve_candidates_preserves_event_and_variant(self):
        calls = []

        def fake_transport(url, payload, timeout):
            calls.append((url, dict(payload), timeout))
            return {
                "query": [payload["query"]],
                "english_query": payload["query"],
                "results": [
                    {
                        "video_name": "video-a.mp4",
                        "frame_index": 90,
                        "timestamp": "0:03",
                        "score": 0.75,
                        "video_title": "ignored external metadata",
                    }
                ],
            }

        client = UpstreamSearchClient(transport=fake_transport)
        candidates = client.retrieve_candidates(
            session_id="session",
            variants=[
                QueryVariant("event-1", "anchor-vi", "cắt hành"),
                QueryVariant("event-1", "anchor-en", "cut an onion"),
            ],
            top_k=5,
            timestamp_resolver=lambda _video, frame, _fallback: frame / 30.0,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual({item.event_id for item in candidates}, {"event-1"})
        self.assertEqual(
            {item.query_variant for item in candidates},
            {"anchor-vi", "anchor-en"},
        )
        self.assertEqual(candidates[0].timestamp_seconds, 3.0)
        self.assertEqual(candidates[0].video_id, "video-a")
        self.assertEqual(calls[0][1], {"query": "cắt hành", "top_k": 5})

    def test_exact_timestamp_from_upstream_is_preferred(self):
        def fake_transport(_url, payload, _timeout):
            return {
                "query": [payload["query"]],
                "results": [
                    {
                        "video_name": "video.mp4",
                        "frame_index": 1,
                        "timestamp": "9:59",
                        "timestamp_seconds": 1.234,
                        "score": 0.1,
                    }
                ],
            }

        result = UpstreamSearchClient(transport=fake_transport).retrieve_candidates(
            session_id="s",
            variants=[QueryVariant("e", "v", "query")],
            top_k=1,
        )
        self.assertEqual(result[0].timestamp_seconds, 1.234)

    def test_invalid_external_schema_is_not_silently_accepted(self):
        client = UpstreamSearchClient(
            transport=lambda _url, _payload, _timeout: {
                "query": ["q"],
                "results": [{"video_name": "v.mp4", "score": 1.0}],
            }
        )
        with self.assertRaisesRegex(UpstreamSearchError, "schema"):
            client.search("q", 1)

    def test_echoed_query_mismatch_and_result_count_overrun_are_rejected(self):
        responses = (
            {"query": ["different"], "results": []},
            {
                "query": ["q"],
                "results": [
                    {
                        "video_name": "a.mp4",
                        "frame_index": 1,
                        "timestamp": "0:01",
                        "score": 0.9,
                    },
                    {
                        "video_name": "b.mp4",
                        "frame_index": 2,
                        "timestamp": "0:02",
                        "score": 0.8,
                    },
                ],
            },
        )
        for response in responses:
            with self.subTest(response=response):
                client = UpstreamSearchClient(
                    transport=lambda _url, _payload, _timeout, value=response: value
                )
                with self.assertRaises(UpstreamSearchError):
                    client.search("q", 1)

    def test_mapping_is_flattened_deterministically(self):
        variants = variants_from_mapping(
            {"e2": {"vi": "hai"}, "e1": {"vi": "mot", "en": "one"}}
        )
        self.assertEqual(
            [(item.event_id, item.variant_id) for item in variants],
            [("e1", "en"), ("e1", "vi"), ("e2", "vi")],
        )


class UrlopenJsonTransportTests(unittest.TestCase):
    # Regression test for a real reported bug: invalid UTF-8 in the upstream
    # response body raised an uncaught UnicodeDecodeError (a ValueError
    # subclass) that _raise_api_error's generic fallback mapped to a 422
    # client input error - misreporting a real upstream problem as if this
    # service's own request were malformed.

    def test_invalid_utf8_is_reported_as_upstream_error_not_a_value_error(self) -> None:
        from io import BytesIO
        from unittest.mock import patch

        from adaptive_search.client import _urlopen_json

        class _FakeResponse(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        with patch("adaptive_search.client.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _FakeResponse(b"\xff\xfe not valid utf-8")
            with self.assertRaises(UpstreamSearchError):
                _urlopen_json("http://example.invalid/search", {"query": "q"}, 5.0)


if __name__ == "__main__":
    unittest.main()
