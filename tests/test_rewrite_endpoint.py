import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import app
from rewrite_queries import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
)


class RewriteEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_rewrite_returns_rewritten_queries_and_request_metadata(self) -> None:
        payload = {
            "modelname": "qwen3:8b",
            "common_query": "Video biểu diễn múa lân màu vàng, đen và trắng.",
            "query": [
                "E1: Khoảnh khắc đầu tiên lân bắt đầu xoay vòng.",
                "E2: Khoảnh khắc bốn chân hoàn toàn chạm đất đầu tiên.",
            ],
        }
        rewritten = [
            "Trong màn biểu diễn múa lân màu vàng, đen và trắng, khoảnh khắc "
            "đầu tiên con lân bắt đầu xoay vòng trên cột số 4 bằng hai chân trước.",
            "Trong màn biểu diễn múa lân màu vàng, đen và trắng, khoảnh khắc "
            "đầu tiên cả bốn chân của con lân hoàn toàn chạm đất sau khi rời cột.",
        ]
        rewrite_mock = AsyncMock(return_value=rewritten)

        with patch("app.rewrite_queries", rewrite_mock):
            response = self.client.post("/rewrite", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "modelname": payload["modelname"],
                "common_query": payload["common_query"],
                "query": rewritten,
            },
        )
        rewrite_mock.assert_awaited_once_with(
            modelname=payload["modelname"],
            common_query=payload["common_query"],
            queries=payload["query"],
        )

    def test_common_query_is_optional(self) -> None:
        payload = {
            "modelname": "qwen3:8b",
            "query": ["E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây."],
        }
        rewritten = [
            "Trong quá trình chế biến món ăn với măng tây, khoảnh khắc đầu tiên "
            "bột được cho vào tô chứa măng tây."
        ]
        rewrite_mock = AsyncMock(return_value=rewritten)

        with patch("app.rewrite_queries", rewrite_mock):
            response = self.client.post("/rewrite", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "modelname": payload["modelname"],
                "common_query": None,
                "query": rewritten,
            },
        )
        rewrite_mock.assert_awaited_once_with(
            modelname=payload["modelname"], common_query=None, queries=payload["query"]
        )

    def test_invalid_request_bodies_return_422_without_calling_ollama(self) -> None:
        invalid_payloads = (
            {"query": ["E1: Một sự kiện."]},
            {"modelname": "qwen3:8b"},
            {"modelname": "qwen3:8b", "query": "E1: Một sự kiện."},
            {"modelname": "", "query": ["E1: Một sự kiện."]},
            {"modelname": "qwen3:8b", "query": []},
            {"modelname": "qwen3:8b", "query": ["   "]},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                rewrite_mock = AsyncMock()
                with patch("app.rewrite_queries", rewrite_mock):
                    response = self.client.post("/rewrite", json=payload)

                self.assertEqual(response.status_code, 422)
                rewrite_mock.assert_not_awaited()

    def test_service_errors_are_mapped_to_http_statuses(self) -> None:
        payload = {"modelname": "qwen3:8b", "query": ["E1: Một sự kiện."]}
        cases = (
            (
                ConfigurationError("a deliberately different message"),
                500,
                "OLLAMA_API_KEY is not configured",
            ),
            (OllamaTimeoutError("Ollama request timed out"), 504, "Ollama request timed out"),
            (
                OllamaRateLimitError("Ollama rate limit exceeded"),
                503,
                "Ollama rate limit exceeded",
            ),
            (OllamaServiceError("Ollama is unavailable"), 502, "Ollama request failed"),
        )

        for error, expected_status, expected_detail in cases:
            with self.subTest(error=type(error).__name__):
                rewrite_mock = AsyncMock(side_effect=error)
                with patch("app.rewrite_queries", rewrite_mock):
                    response = self.client.post("/rewrite", json=payload)

                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json(), {"detail": expected_detail})


if __name__ == "__main__":
    unittest.main()
