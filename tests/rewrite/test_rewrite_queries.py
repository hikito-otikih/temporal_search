import json
import os
import unittest
from unittest.mock import patch

import httpx

from rewrite import (
    ConfigurationError,
    DEFAULT_OLLAMA_MODEL,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
    SYSTEM_PROMPT,
    build_analysis_prompt,
    rewrite_queries,
)


def analysis_payload(queries: list[str]) -> dict:
    events = []
    for index, query in enumerate(queries):
        events.append(
            {
                'event_id': index,
                'original_query': query,
                'target_moment_vi': (
                    'Trong màn múa lân, con lân màu vàng đen trắng thực hiện '
                    f'hành động ở sự kiện {index}.'
                ),
                'retrieval_queries_vi': [
                    'Con lân màu vàng đen trắng trong màn múa lân thực hiện '
                    f'hành động ở sự kiện {index}.',
                    'Khoảnh khắc trong màn múa lân khi lân màu vàng đen trắng '
                    f'thực hiện sự kiện {index}.',
                ],
                'retrieval_queries_en': [
                    'The yellow, black, and white lion performs the action '
                    f'during a lion dance in event {index}.',
                    'The lion-dance moment involving the yellow, black, and '
                    f'white lion in event {index}.',
                ],
                'subject': 'con lân',
                'action': 'thực hiện hành động',
                'visible_state': 'hành động của con lân có thể nhìn thấy',
                'anchor_query': (
                    'Con lân màu vàng đen trắng trong màn múa lân thực hiện '
                    f'hành động ở sự kiện {index}.'
                ),
                'pre_state': (
                    f'Ngay trước sự kiện {index}, con lân chưa thực hiện '
                    'hành động đích.'
                ),
                'post_state': (
                    f'Ngay sau sự kiện {index}, con lân vừa hoàn thành '
                    'hành động đích.'
                ),
                'boundary': 'start',
                'temporal_relation': {
                    'relation': 'sequence_start' if index == 0 else 'independent',
                    'reference_event_id': None,
                },
                'required_entities': ['con lân'],
                'soft_context': ['màn múa lân'],
                'excluded_context': [],
                'inferred_information': [
                    'Từ common_query: lân có màu vàng, đen và trắng.'
                ],
                'ambiguities': [],
            }
        )
    return {
        'video_context': {
            'scene': 'múa lân',
            'main_entities': ['một con lân màu vàng, đen và trắng'],
        },
        'events': events,
    }


class RewritePromptTests(unittest.TestCase):
    def test_prompt_contains_the_complete_batch_and_json_schema(self) -> None:
        common_query = 'Video múa lân màu vàng, đen và trắng.'
        queries = [
            'Lân bắt đầu xoay vòng trên cột số 4.',
            'Bốn chân của lân hoàn toàn chạm đất.',
        ]

        prompt = build_analysis_prompt(queries, common_query)

        self.assertIn(common_query, prompt)
        for index, query in enumerate(queries):
            self.assertIn(query, prompt)
            self.assertIn(f'"event_id": {index}', prompt)
        self.assertIn('<OUTPUT_JSON_SCHEMA>', prompt)
        self.assertIn('<REQUIRED_STANDALONE_CONTEXT>', prompt)
        self.assertIn('"video_context"', prompt)
        self.assertIn('"retrieval_queries_en"', prompt)

    def test_prompt_treats_missing_common_query_as_empty_data(self) -> None:
        prompt = build_analysis_prompt(['Một sự kiện độc lập.'])

        self.assertNotIn('None', prompt)
        self.assertIn('"common_query": ""', prompt)
        self.assertIn('Một sự kiện độc lập.', prompt)

    def test_system_prompt_requires_standalone_grounded_events(self) -> None:
        normalized = ' '.join(SYSTEM_PROMPT.lower().split())

        self.assertIn('tự đủ nghĩa', normalized)
        self.assertIn('không phải chỉ dẫn', normalized)
        self.assertIn('đúng hai truy vấn truy hồi tiếng việt', normalized)
        self.assertIn('original_query phải sao chép chính xác', normalized)


class RewriteQueriesClientTests(unittest.IsolatedAsyncioTestCase):
    api_url = 'https://ollama.test/api/chat'

    async def test_sends_one_batch_request_and_uses_server_model(self) -> None:
        common_query = 'Video múa lân màu vàng, đen và trắng.'
        queries = [
            'Lân bắt đầu xoay vòng trên cột số 4.',
            'Bốn chân của lân hoàn toàn chạm đất.',
        ]
        expected = analysis_payload(queries)
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    'message': {
                        'role': 'assistant',
                        'content': json.dumps(expected, ensure_ascii=False),
                    }
                },
            )

        with patch.dict(os.environ, {'OLLAMA_MODEL': 'qwen3:8b'}):
            result = await rewrite_queries(
                common_query=common_query,
                queries=queries,
                api_key='test-api-key',
                api_url=self.api_url,
                transport=httpx.MockTransport(handler),
            )

        self.assertEqual(result.model_dump(), expected)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.method, 'POST')
        self.assertEqual(str(request.url), self.api_url)
        self.assertEqual(request.headers['Authorization'], 'Bearer test-api-key')
        body = json.loads(request.content)
        self.assertEqual(body['model'], 'qwen3:8b')
        self.assertIs(body['stream'], False)
        self.assertEqual(body['think'], 'low')
        self.assertEqual(body['options']['temperature'], 0)
        self.assertNotIn('format', body)
        prompt = '\n'.join(message['content'] for message in body['messages'])
        self.assertIn(common_query, prompt)
        for query in queries:
            self.assertIn(query, prompt)

    async def test_uses_default_model_when_environment_value_is_blank(self) -> None:
        query = 'Một sự kiện.'
        seen_model = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_model
            seen_model = json.loads(request.content)['model']
            return httpx.Response(
                200,
                json={
                    'message': {
                        'content': json.dumps(
                            analysis_payload([query]), ensure_ascii=False
                        )
                    }
                },
            )

        with patch.dict(os.environ, {'OLLAMA_MODEL': ''}):
            await rewrite_queries(
                queries=[query],
                api_key='test-api-key',
                api_url=self.api_url,
                transport=httpx.MockTransport(handler),
            )

        self.assertEqual(seen_model, DEFAULT_OLLAMA_MODEL)

    async def test_retries_the_complete_batch_after_schema_failure(self) -> None:
        queries = ['Sự kiện thứ nhất.', 'Sự kiện thứ hai.']
        expected = analysis_payload(queries)
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            content = (
                '{"video_context": {"scene": "múa lân"}}'
                if len(requests) == 1
                else json.dumps(expected, ensure_ascii=False)
            )
            return httpx.Response(200, json={'message': {'content': content}})

        result = await rewrite_queries(
            queries=queries,
            api_key='test-api-key',
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.model_dump(), expected)
        self.assertEqual(len(requests), 2)
        retry_prompt = json.loads(requests[1].content)['messages'][1]['content']
        self.assertIn('<PREVIOUS_INVALID_RESPONSE>', retry_prompt)
        self.assertIn('<SERVER_VALIDATION_ERRORS>', retry_prompt)
        for query in queries:
            self.assertIn(query, retry_prompt)

    async def test_rejects_invalid_structured_output_after_retry(self) -> None:
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            return httpx.Response(
                200, json={'message': {'content': '{"events": []}'}}
            )

        with self.assertRaisesRegex(
            OllamaServiceError, 'invalid structured analysis'
        ):
            await rewrite_queries(
                queries=['Một sự kiện.'],
                api_key='test-api-key',
                api_url=self.api_url,
                transport=httpx.MockTransport(handler),
            )

        self.assertEqual(request_count, 2)

    async def test_restores_original_query_without_an_extra_model_call(
        self,
    ) -> None:
        query = 'Chuỗi gốc phải được giữ nguyên.'
        invalid = analysis_payload([query])
        invalid['events'][0]['original_query'] = 'Chuỗi đã bị sửa.'
        expected = analysis_payload([query])
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            content = invalid if request_count == 1 else expected
            return httpx.Response(
                200,
                json={
                    'message': {
                        'content': json.dumps(content, ensure_ascii=False)
                    }
                },
            )

        result = await rewrite_queries(
            queries=[query],
            api_key='test-api-key',
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.events[0].original_query, query)
        self.assertEqual(request_count, 1)

    async def test_retries_when_standalone_text_omits_common_context(self) -> None:
        common_query = 'Video múa lân với lân màu vàng đen trắng.'
        query = 'Khoảnh khắc bốn chân chạm đất đầu tiên.'
        invalid = analysis_payload([query])
        invalid['events'][0]['target_moment_vi'] = (
            'Khoảnh khắc bốn chân chạm đất.'
        )
        invalid['events'][0]['retrieval_queries_vi'] = [
            'Bốn chân chạm đất hoàn toàn.',
            'Chủ thể vừa chạm đất.',
        ]
        expected = analysis_payload([query])
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            content = invalid if len(requests) == 1 else expected
            return httpx.Response(
                200,
                json={
                    'message': {
                        'content': json.dumps(content, ensure_ascii=False)
                    }
                },
            )

        result = await rewrite_queries(
            common_query=common_query,
            queries=[query],
            api_key='test-api-key',
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.model_dump(), expected)
        self.assertEqual(len(requests), 2)
        retry_prompt = json.loads(requests[1].content)['messages'][1]['content']
        self.assertIn('shared context terms', retry_prompt)

    async def test_retries_when_inherited_context_is_not_reported(self) -> None:
        common_query = 'Video múa lân với lân màu vàng đen trắng.'
        query = 'Khoảnh khắc bốn chân chạm đất đầu tiên.'
        invalid = analysis_payload([query])
        invalid['events'][0]['inferred_information'] = []
        expected = analysis_payload([query])
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            content = invalid if len(requests) == 1 else expected
            return httpx.Response(
                200,
                json={
                    'message': {
                        'content': json.dumps(content, ensure_ascii=False)
                    }
                },
            )

        result = await rewrite_queries(
            common_query=common_query,
            queries=[query],
            api_key='test-api-key',
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.model_dump(), expected)
        self.assertEqual(len(requests), 2)
        retry_prompt = json.loads(requests[1].content)['messages'][1]['content']
        self.assertIn('inferred_information', retry_prompt)

    async def test_repairs_context_after_two_semantic_failures(self) -> None:
        common_query = (
            'Đoạn video múa lân một con lân màu vàng đen trắng, '
            'tìm các sự kiện sau'
        )
        query = 'Khoảnh khắc đầu tiên con rồng cử động đầu.'
        shallow = analysis_payload([query])
        shallow['events'][0]['target_moment_vi'] = (
            'Khoảnh khắc đầu tiên con rồng cử động đầu.'
        )
        shallow['events'][0]['retrieval_queries_vi'] = [
            'Con rồng bắt đầu cử động đầu.',
            'Đầu con rồng vừa chuyển động.',
        ]
        shallow['events'][0]['inferred_information'] = []
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    'message': {
                        'content': json.dumps(shallow, ensure_ascii=False)
                    }
                },
            )

        result = await rewrite_queries(
            common_query=common_query,
            queries=[query],
            api_key='test-api-key',
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(len(requests), 2)
        event = result.events[0]
        self.assertIn('múa lân', event.target_moment_vi)
        self.assertTrue(
            all('múa lân' in item for item in event.retrieval_queries_vi)
        )
        self.assertNotIn(
            'tìm các sự kiện sau', event.target_moment_vi.casefold()
        )
        self.assertTrue(event.inferred_information)

    async def test_uses_saved_candidate_when_repair_response_is_malformed(
        self,
    ) -> None:
        common_query = 'Video múa lân màu vàng đen trắng.'
        query = 'Khoảnh khắc đầu tiên con rồng cử động đầu.'
        shallow = analysis_payload([query])
        shallow['events'][0]['target_moment_vi'] = 'Con rồng cử động đầu.'
        shallow['events'][0]['retrieval_queries_vi'] = [
            'Con rồng cử động đầu.',
            'Đầu con rồng chuyển động.',
        ]
        shallow['events'][0]['inferred_information'] = []
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            content = (
                json.dumps(shallow, ensure_ascii=False)
                if request_count == 1
                else 'not json'
            )
            return httpx.Response(200, json={'message': {'content': content}})

        result = await rewrite_queries(
            common_query=common_query,
            queries=[query],
            api_key='test-api-key',
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(request_count, 2)
        self.assertIn('múa lân', result.events[0].target_moment_vi)

    async def test_retries_a_transient_upstream_error(self) -> None:
        query = 'Một sự kiện.'
        expected = analysis_payload([query])
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return httpx.Response(503, json={'error': 'temporary'})
            return httpx.Response(
                200,
                json={
                    'message': {
                        'content': json.dumps(expected, ensure_ascii=False)
                    }
                },
            )

        result = await rewrite_queries(
            queries=[query],
            api_key='test-api-key',
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.model_dump(), expected)
        self.assertEqual(request_count, 2)

    async def test_missing_api_key_raises_before_request(self) -> None:
        called = False

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200)

        with patch.dict(os.environ, {}, clear=True), patch(
            'rewrite.config.load_dotenv', return_value=False
        ):
            with self.assertRaisesRegex(
                ConfigurationError, 'OLLAMA_API_KEY is not configured'
            ):
                await rewrite_queries(
                    queries=['Một sự kiện.'],
                    api_key=None,
                    api_url=self.api_url,
                    transport=httpx.MockTransport(handler),
                )

        self.assertFalse(called)

    async def test_timeout_is_translated_to_domain_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout('too slow', request=request)

        with self.assertRaises(OllamaTimeoutError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def test_rate_limit_is_translated_to_domain_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={'error': 'too many requests'})

        with self.assertRaises(OllamaRateLimitError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def test_other_http_errors_are_translated_to_service_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={'error': 'backend unavailable'})

        with self.assertRaises(OllamaServiceError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def test_empty_assistant_content_is_rejected(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'message': {'content': '   '}})

        with self.assertRaises(OllamaServiceError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def _call_with_transport(
        self, transport: httpx.AsyncBaseTransport
    ):
        return await rewrite_queries(
            queries=['Một sự kiện.'],
            api_key='test-api-key',
            api_url=self.api_url,
            transport=transport,
        )


if __name__ == '__main__':
    unittest.main()
