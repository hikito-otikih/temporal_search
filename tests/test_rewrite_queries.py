import asyncio
import json
import os
import re
import unittest
from unittest.mock import patch

import httpx

from rewrite_queries import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
    SYSTEM_PROMPT,
    build_rewrite_prompt,
    rewrite_queries,
)


class RewritePromptTests(unittest.TestCase):
    def test_prompt_supplies_shared_context_and_the_complete_event_sequence(self) -> None:
        common_query = "Video biểu diễn múa lân màu vàng, đen và trắng."
        queries = [
            "E1: Lân bắt đầu xoay vòng trên cột số 4.",
            "E2: Bốn chân của lân hoàn toàn chạm đất.",
            "E3: Hai người biểu diễn cúi chào ban giám khảo.",
        ]

        first_prompt = build_rewrite_prompt(queries, 0, common_query)
        second_prompt = build_rewrite_prompt(queries, 1, common_query)

        for prompt in (first_prompt, second_prompt):
            self.assertIn(common_query, prompt)
            for query in queries:
                self.assertIn(query, prompt)

        # Changing only the target must change the instruction while retaining the
        # surrounding events that give the target its meaning.
        self.assertNotEqual(first_prompt, second_prompt)

    def test_prompt_does_not_render_missing_context_as_literal_none(self) -> None:
        prompt = build_rewrite_prompt(["E1: Một sự kiện độc lập."], 0)

        self.assertNotIn("None", prompt)
        self.assertIn("E1: Một sự kiện độc lập.", prompt)

    def test_required_context_terms_exclude_search_instruction_words(self) -> None:
        prompt = build_rewrite_prompt(
            ["Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên."],
            0,
            "Đoạn video múa lân màu vàng đen trắng, tìm các sự kiện sau",
        )
        match = re.search(
            r"<REQUIRED_CONTEXT_TERMS>\s*(.*?)\s*</REQUIRED_CONTEXT_TERMS>",
            prompt,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(match)
        terms = {term.strip() for term in match.group(1).split(",")}
        self.assertNotIn("kiện", terms)
        self.assertTrue({"múa", "lân", "vàng", "đen", "trắng"}.issubset(terms))

    def test_prompt_requires_context_expansion_instead_of_shallow_proofreading(self) -> None:
        common_query = (
            "Đoạn video biểu diễn múa lân với một con lân màu vàng, đen và trắng."
        )
        prompt = build_rewrite_prompt(
            ["Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên."],
            0,
            common_query,
        )
        system_instruction = " ".join(SYSTEM_PROMPT.lower().split())
        target_instruction = " ".join(prompt.lower().split())

        # Weak models otherwise tend to only change digits or polish grammar. The
        # contract must explicitly require a missing identifying detail from the
        # shared context to appear in the standalone result.
        self.assertRegex(
            system_instruction,
            r"không phải.{0,100}(?:sửa|hiệu đính)",
        )
        self.assertRegex(
            system_instruction,
            r"common_query.{0,300}(?:chi tiết|đặc điểm|thuộc tính) nhận diện",
        )
        self.assertRegex(
            target_instruction,
            r"bắt buộc.{0,240}(?:common_query|common_context)",
        )
        self.assertIn(common_query.lower(), target_instruction)


class RewriteQueriesClientTests(unittest.IsolatedAsyncioTestCase):
    api_url = "https://ollama.test/api/chat"

    async def test_rewrites_each_query_in_order_using_ollama_chat_api(self) -> None:
        common_query = "Video biểu diễn múa lân màu vàng, đen và trắng."
        queries = [
            "E1: Lân bắt đầu xoay vòng trên cột số 4.",
            "E2: Bốn chân của lân hoàn toàn chạm đất.",
        ]
        rewritten = [
            "Trong màn múa lân của con lân màu vàng, đen và trắng, con lân "
            "bắt đầu xoay vòng trên cột số 4.",
            "Trong màn múa lân của con lân màu vàng, đen và trắng, cả bốn chân "
            "của con lân chạm đất sau khi rời cột.",
        ]
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": rewritten[len(requests) - 1]}},
            )

        result = await rewrite_queries(
            modelname="qwen3:8b",
            common_query=common_query,
            queries=queries,
            api_key="test-api-key",
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result, rewritten)
        self.assertEqual(len(requests), len(queries))
        for index, request in enumerate(requests):
            with self.subTest(index=index):
                self.assertEqual(request.method, "POST")
                self.assertEqual(str(request.url), self.api_url)
                self.assertEqual(request.headers["Authorization"], "Bearer test-api-key")
                body = json.loads(request.content)
                self.assertEqual(body["model"], "qwen3:8b")
                self.assertIs(body["stream"], False)
                self.assertIs(body["think"], False)
                self.assertEqual(body["options"]["temperature"], 0)
                prompt = "\n".join(message["content"] for message in body["messages"])
                self.assertIn(common_query, prompt)
                for query in queries:
                    self.assertIn(query, prompt)

        self.assertNotEqual(requests[0].content, requests[1].content)

    async def test_retries_a_shallow_rewrite_that_omits_common_context(self) -> None:
        common_query = (
            "Đoạn video biểu diễn múa lân với một con lân màu vàng, đen và trắng."
        )
        query = "Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên."
        shallow_draft = "Khoảnh khắc đầu tiên bốn chân của con lân hoàn toàn chạm đất."
        enriched_draft = (
            "Trong đoạn video biểu diễn múa lân với một con lân màu vàng, đen và "
            "trắng, khoảnh khắc đầu tiên cả bốn chân của con lân hoàn toàn chạm đất."
        )
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            content = shallow_draft if len(requests) == 1 else enriched_draft
            return httpx.Response(200, json={"message": {"content": content}})

        result = await rewrite_queries(
            modelname="gpt-oss:20b",
            common_query=common_query,
            queries=[query],
            api_key="test-api-key",
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result, [enriched_draft])
        self.assertEqual(len(requests), 2)
        retry_body = json.loads(requests[1].content)
        retry_prompt = "\n".join(
            message["content"] for message in retry_body["messages"]
        )
        self.assertIn(shallow_draft, retry_prompt)
        self.assertIn(common_query, retry_prompt)

    async def test_fallback_prepends_clean_common_context_after_two_shallow_drafts(
        self,
    ) -> None:
        common_query = (
            "Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau"
        )
        drafts = [
            "Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.",
            "Khoảnh khắc đầu tiên bốn chân của lân hoàn toàn chạm đất.",
        ]
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            content = drafts[request_count]
            request_count += 1
            return httpx.Response(200, json={"message": {"content": content}})

        result = await rewrite_queries(
            modelname="gpt-oss:20b",
            common_query=common_query,
            queries=[drafts[0]],
            api_key="test-api-key",
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(request_count, 2)
        self.assertEqual(
            result,
            [
                "Đoạn video múa lân một con lân màu vàng đen trắng. "
                + drafts[1]
            ],
        )
        self.assertNotIn("tìm các sự kiện sau", result[0].lower())

    async def test_preserves_input_order_when_responses_finish_out_of_order(
        self,
    ) -> None:
        common_query = "Đoạn video múa lân màu vàng, đen và trắng."
        queries = [
            "E1: Lân bắt đầu xoay vòng trên cột số 4.",
            "E2: Bốn chân của lân hoàn toàn chạm đất.",
            "E3: Hai người biểu diễn lân cúi chào ban giám khảo.",
        ]
        rewritten = [
            f"{common_query} Sự kiện cần tìm: {query}" for query in queries
        ]
        completion_order: list[int] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            prompt = "\n".join(
                message["content"] for message in body["messages"]
            )
            match = re.search(
                r"<TARGET_QUERY>\s*(.*?)\s*</TARGET_QUERY>",
                prompt,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match, "request prompt must identify its target query")
            index = queries.index(match.group(1))
            await asyncio.sleep((0.03, 0.01, 0.0)[index])
            completion_order.append(index)
            return httpx.Response(
                200,
                json={"message": {"content": rewritten[index]}},
            )

        result = await rewrite_queries(
            modelname="gpt-oss:20b",
            common_query=common_query,
            queries=queries,
            api_key="test-api-key",
            api_url=self.api_url,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(completion_order, [2, 1, 0])
        self.assertEqual(result, rewritten)

    async def test_missing_api_key_raises_configuration_error_before_request(self) -> None:
        called = False

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"message": {"content": "unused"}})

        with patch.dict(os.environ, {}, clear=True), patch(
            "rewrite_queries.load_dotenv", return_value=False
        ):
            with self.assertRaisesRegex(
                ConfigurationError, "OLLAMA_API_KEY is not configured"
            ):
                await rewrite_queries(
                    modelname="qwen3:8b",
                    common_query=None,
                    queries=["E1: Một sự kiện."],
                    api_key=None,
                    api_url=self.api_url,
                    transport=httpx.MockTransport(handler),
                )

        self.assertFalse(called)

    async def test_timeout_is_translated_to_domain_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        with self.assertRaises(OllamaTimeoutError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def test_rate_limit_is_translated_to_domain_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "too many requests"})

        with self.assertRaises(OllamaRateLimitError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def test_other_http_errors_are_translated_to_service_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "backend unavailable"})

        with self.assertRaises(OllamaServiceError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def test_empty_assistant_content_is_rejected(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": {"content": "   "}})

        with self.assertRaises(OllamaServiceError):
            await self._call_with_transport(httpx.MockTransport(handler))

    async def _call_with_transport(self, transport: httpx.AsyncBaseTransport) -> list[str]:
        return await rewrite_queries(
            modelname="qwen3:8b",
            common_query=None,
            queries=["E1: Một sự kiện."],
            api_key="test-api-key",
            api_url=self.api_url,
            transport=transport,
        )


if __name__ == "__main__":
    unittest.main()
