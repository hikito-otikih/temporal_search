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
from rewrite_schema import RewriteResponse


def analysis_result(queries: list[str]) -> RewriteResponse:
    return RewriteResponse.model_validate(
        {
            'video_context': {
                'scene': 'múa lân',
                'main_entities': ['một con lân màu vàng, đen và trắng'],
            },
            'events': [
                {
                    'event_id': index,
                    'original_query': query,
                    'target_moment_vi': (
                        f'Khoảnh khắc độc lập của con lân cho sự kiện {index}.'
                    ),
                    'retrieval_queries_vi': [
                        f'Con lân thực hiện hành động ở sự kiện {index}.',
                        f'Khoảnh khắc con lân của sự kiện {index}.',
                    ],
                    'retrieval_queries_en': [
                        f'The lion performs the action in event {index}.',
                        f'The lion moment for event {index}.',
                    ],
                    'subject': 'con lân',
                    'action': 'thực hiện hành động',
                    'visible_state': 'hành động của con lân có thể nhìn thấy',
                    'boundary': 'start',
                    'temporal_relation': {
                        'relation': (
                            'sequence_start' if index == 0 else 'independent'
                        ),
                        'reference_event_id': None,
                    },
                    'required_entities': ['con lân'],
                    'soft_context': ['màn múa lân'],
                    'excluded_context': [],
                    'inferred_information': [],
                    'ambiguities': [],
                }
                for index, query in enumerate(queries)
            ],
        }
    )


class RewriteEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_rewrite_returns_structured_analysis_without_model_field(self) -> None:
        payload = {
            'common_query': 'Video múa lân màu vàng, đen và trắng.',
            'query': [
                'Khoảnh khắc đầu tiên lân bắt đầu xoay vòng.',
                'Khoảnh khắc bốn chân hoàn toàn chạm đất đầu tiên.',
            ],
        }
        result = analysis_result(payload['query'])
        rewrite_mock = AsyncMock(return_value=result)

        with patch('app.rewrite_queries', rewrite_mock):
            response = self.client.post('/rewrite', json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result.model_dump())
        self.assertNotIn('modelname', response.json())
        self.assertNotIn('model_name', response.json())
        rewrite_mock.assert_awaited_once_with(
            common_query=payload['common_query'],
            queries=payload['query'],
        )

    def test_common_query_is_optional(self) -> None:
        payload = {'query': ['Khoảnh khắc đầu tiên bột được bỏ vào tô.']}
        result = analysis_result(payload['query'])
        rewrite_mock = AsyncMock(return_value=result)

        with patch('app.rewrite_queries', rewrite_mock):
            response = self.client.post('/rewrite', json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result.model_dump())
        rewrite_mock.assert_awaited_once_with(
            common_query=None,
            queries=payload['query'],
        )

    def test_model_fields_and_other_invalid_bodies_return_422(self) -> None:
        valid_query = ['Một sự kiện hợp lệ.']
        invalid_payloads = (
            {'query': valid_query, 'modelname': 'gpt-oss:20b'},
            {'query': valid_query, 'model_name': 'gpt-oss:20b'},
            {},
            {'query': 'Một sự kiện.'},
            {'query': []},
            {'query': ['   ']},
            {'query': valid_query, 'unexpected': True},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                rewrite_mock = AsyncMock()
                with patch('app.rewrite_queries', rewrite_mock):
                    response = self.client.post('/rewrite', json=payload)

                self.assertEqual(response.status_code, 422)
                rewrite_mock.assert_not_awaited()

    def test_service_errors_are_mapped_to_http_statuses(self) -> None:
        payload = {'query': ['Một sự kiện.']}
        cases = (
            (
                ConfigurationError('different internal message'),
                500,
                'OLLAMA_API_KEY is not configured',
            ),
            (
                OllamaTimeoutError('Ollama request timed out'),
                504,
                'Ollama request timed out',
            ),
            (
                OllamaRateLimitError('Ollama rate limit exceeded'),
                503,
                'Ollama rate limit exceeded',
            ),
            (
                OllamaServiceError('Ollama is unavailable'),
                502,
                'Ollama request failed',
            ),
        )

        for error, expected_status, expected_detail in cases:
            with self.subTest(error=type(error).__name__):
                rewrite_mock = AsyncMock(side_effect=error)
                with patch('app.rewrite_queries', rewrite_mock):
                    response = self.client.post('/rewrite', json=payload)

                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json(), {'detail': expected_detail})


if __name__ == '__main__':
    unittest.main()
