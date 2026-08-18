import asyncio
import json
import logging
import os
from typing import Sequence

import httpx
from pydantic import ValidationError

from .config import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_API_URL,
    OLLAMA_TIMEOUT_SECONDS,
    load_ollama_env,
)
from .constants import (
    COMMON_QUERY_INSTRUCTION_SUFFIX,
    CONTEXT_STOPWORDS,
    CONTEXT_TERM_PATTERN,
    MAX_RETRY_RESPONSE_CHARS,
    MAX_TRANSPORT_ATTEMPTS,
    MAX_VALIDATION_ATTEMPTS,
    OLLAMA_RETRY_DELAY_SECONDS,
    SYSTEM_PROMPT,
    TRANSIENT_OLLAMA_STATUSES,
    VIETNAMESE_SIGNATURE_PATTERN,
)
from .exceptions import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
    SemanticValidationError,
)
from .schemas import RewriteResponse

logger = logging.getLogger(__name__)


def _extract_context_terms(common_query: str | None) -> list[str]:
    if not common_query:
        return []

    terms: list[str] = []
    for token in CONTEXT_TERM_PATTERN.findall(common_query.casefold()):
        if len(token) < 2 or token in CONTEXT_STOPWORDS or token in terms:
            continue
        terms.append(token)
    return terms[:12]


def _clean_common_context(common_query: str | None) -> str:
    if not common_query:
        return ''
    context = COMMON_QUERY_INSTRUCTION_SUFFIX.sub('', common_query)
    return context.strip(' \t\r\n,;:.!?')


def _matched_context_terms(text: str, context_terms: Sequence[str]) -> list[str]:
    text_terms = set(CONTEXT_TERM_PATTERN.findall(text.casefold()))
    return [term for term in context_terms if term in text_terms]


def build_analysis_prompt(
    queries: Sequence[str],
    common_query: str | None = None,
    previous_response: str | None = None,
    validation_errors: str | None = None,
) -> str:
    '''Build one batch prompt containing the input data and required JSON schema.'''
    input_data = {
        'common_query': common_query or '',
        'events': [
            {'event_id': index, 'original_query': query}
            for index, query in enumerate(queries)
        ],
    }
    schema = RewriteResponse.model_json_schema()
    context_terms = _extract_context_terms(common_query)
    prompt = (
        '<INPUT_DATA>\n'
        + json.dumps(input_data, ensure_ascii=False, indent=2)
        + '\n</INPUT_DATA>'
    )
    if context_terms:
        minimum_matches = min(2, len(context_terms))
        prompt += (
            '\n\n<REQUIRED_STANDALONE_CONTEXT>\n'
            + json.dumps(context_terms, ensure_ascii=False)
            + f'\nMỗi target_moment_vi và từng retrieval_queries_vi phải chứa '
            + f'ít nhất {minimum_matches} từ khóa khác nhau trong danh sách '
            + 'trên. retrieval_queries_en phải dịch cùng bối cảnh đó sang '
            + 'tiếng Anh.\n</REQUIRED_STANDALONE_CONTEXT>'
        )
    prompt += (
        '\n\n<OUTPUT_JSON_SCHEMA>\n'
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + '\n</OUTPUT_JSON_SCHEMA>\n\n'
        + 'Phân tích toàn bộ INPUT_DATA và chỉ trả về JSON object hoàn chỉnh '
        + 'khớp OUTPUT_JSON_SCHEMA.'
    )

    if previous_response is not None:
        prompt += (
            '\n\n<PREVIOUS_INVALID_RESPONSE>\n'
            + previous_response[:MAX_RETRY_RESPONSE_CHARS]
            + '\n</PREVIOUS_INVALID_RESPONSE>\n\n'
            + '<SERVER_VALIDATION_ERRORS>\n'
            + (validation_errors or 'JSON hoặc schema không hợp lệ')
            + '\n</SERVER_VALIDATION_ERRORS>\n\n'
            + 'Response trước không hợp lệ. Hãy tạo lại TOÀN BỘ JSON từ '
            + 'INPUT_DATA gốc; không vá từng phần và không giải thích.'
        )
    return prompt


def _extract_json_object(content: str) -> dict:
    text = content.strip()
    if not text:
        raise ValueError('assistant content is empty')

    code_fence = chr(96) * 3
    if text.startswith(code_fence):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith(code_fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == code_fence:
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('assistant content does not contain a JSON object')

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError('assistant JSON root must be an object')
    return payload


def _validate_source_alignment(
    analysis: RewriteResponse,
    queries: Sequence[str],
) -> None:
    if len(analysis.events) != len(queries):
        raise ValueError(
            f'expected {len(queries)} events but received {len(analysis.events)}'
        )

    for index, (event, original_query) in enumerate(
        zip(analysis.events, queries)
    ):
        if event.event_id != index:
            raise ValueError(
                f'events[{index}].event_id must equal {index}'
            )
        if event.original_query != original_query:
            if event.original_query in queries:
                raise ValueError(
                    f'events[{index}] appears to contain a reordered query'
                )
            event.original_query = original_query
        vi_queries = {query.casefold() for query in event.retrieval_queries_vi}
        en_queries = {query.casefold() for query in event.retrieval_queries_en}
        if len(vi_queries) != len(event.retrieval_queries_vi):
            raise ValueError(
                f'events[{index}].retrieval_queries_vi must be distinct'
            )
        if len(en_queries) != len(event.retrieval_queries_en):
            raise ValueError(
                f'events[{index}].retrieval_queries_en must be distinct'
            )


def _validate_standalone_context(
    analysis: RewriteResponse,
    common_query: str | None,
) -> None:
    context_terms = _extract_context_terms(common_query)
    if not context_terms:
        return

    # retrieval_queries_en gets no term-matching check below (context_terms
    # are extracted from the Vietnamese common_query - checking whether
    # translated English text literally contains Vietnamese words is not
    # meaningful, unlike the retrieval_queries_vi check). This narrower,
    # language-agnostic check instead catches the most common way EN
    # coverage silently goes missing despite SYSTEM_PROMPT explicitly
    # requiring it (build_analysis_prompt's REQUIRED_STANDALONE_CONTEXT
    # block): the LLM echoing original_query back unchanged as an EN
    # "translation" instead of actually translating and expanding it.
    for index, event in enumerate(analysis.events):
        original_normalized = event.original_query.strip().casefold()
        for query_index, en_query in enumerate(event.retrieval_queries_en):
            if en_query.strip().casefold() == original_normalized:
                raise SemanticValidationError(
                    f'events[{index}].retrieval_queries_en[{query_index}] must be '
                    'an English translation carrying the same standalone context, '
                    'not a literal copy of original_query'
                )
            # A literal-echo check alone lets an LLM leave the field in (or
            # paraphrase it into) Vietnamese pass as long as it doesn't
            # byte-match original_query - genuine English text never
            # contains these characters, so this catches "not actually
            # translated" independent of the echo check above.
            if VIETNAMESE_SIGNATURE_PATTERN.search(en_query):
                raise SemanticValidationError(
                    f'events[{index}].retrieval_queries_en[{query_index}] must be '
                    'written in English - it contains Vietnamese-specific '
                    'characters, so it was not actually translated'
                )

    minimum_matches = min(2, len(context_terms))
    for index, event in enumerate(analysis.events):
        standalone_fields = [
            ('target_moment_vi', event.target_moment_vi),
            ('anchor_query', event.anchor_query),
            *[
                (f'retrieval_queries_vi[{query_index}]', query)
                for query_index, query in enumerate(event.retrieval_queries_vi)
            ],
        ]
        for field_name, text in standalone_fields:
            matched_terms = _matched_context_terms(text, context_terms)
            if len(matched_terms) < minimum_matches:
                raise SemanticValidationError(
                    f'events[{index}].{field_name} must contain at least '
                    f'{minimum_matches} shared context terms from '
                    f'{context_terms}; found {matched_terms}'
                )

        original_terms = set(
            CONTEXT_TERM_PATTERN.findall(event.original_query.casefold())
        )
        standalone_terms = set(
            CONTEXT_TERM_PATTERN.findall(
                ' '.join(text for _, text in standalone_fields).casefold()
            )
        )
        inherited_terms = [
            term
            for term in context_terms
            if term in standalone_terms and term not in original_terms
        ]
        if inherited_terms and not event.inferred_information:
            raise SemanticValidationError(
                f'events[{index}].inferred_information must describe '
                f'inherited common-query terms {inherited_terms}'
            )


def _repair_standalone_context(
    analysis: RewriteResponse,
    common_query: str | None,
) -> RewriteResponse:
    context_terms = _extract_context_terms(common_query)
    clean_context = _clean_common_context(common_query)
    if not context_terms or not clean_context:
        return analysis

    repaired = analysis.model_copy(deep=True)
    minimum_matches = min(2, len(context_terms))
    context_prefix = clean_context.rstrip(' .!?')

    for event in repaired.events:
        if (
            len(
                _matched_context_terms(
                    event.target_moment_vi, context_terms
                )
            )
            < minimum_matches
        ):
            event.target_moment_vi = (
                f'{context_prefix}. {event.target_moment_vi}'
            )

        if (
            len(_matched_context_terms(event.anchor_query, context_terms))
            < minimum_matches
        ):
            event.anchor_query = f'{context_prefix}. {event.anchor_query}'

        event.retrieval_queries_vi = [
            query
            if len(_matched_context_terms(query, context_terms))
            >= minimum_matches
            else f'{context_prefix}. {query}'
            for query in event.retrieval_queries_vi
        ]

        original_terms = set(
            CONTEXT_TERM_PATTERN.findall(event.original_query.casefold())
        )
        # Must include anchor_query here, matching _validate_standalone_context's
        # own standalone_fields set exactly - anchor_query is mutated above
        # just like target_moment_vi/retrieval_queries_vi, so omitting it
        # from this event's own "did we introduce new inherited terms" check
        # let a term introduced only via anchor_query's prefix go completely
        # unaccounted for. inferred_information would then stay empty here
        # while _validate_standalone_context's re-check below (which does
        # include anchor_query) found a real, unreported inherited term -
        # raising SemanticValidationError from a call site with no
        # surrounding handler, an uncaught 500 for output this function
        # itself produced.
        standalone_terms = set(
            CONTEXT_TERM_PATTERN.findall(
                ' '.join(
                    [
                        event.target_moment_vi,
                        event.anchor_query,
                        *event.retrieval_queries_vi,
                    ]
                ).casefold()
            )
        )
        inherited_terms = [
            term
            for term in context_terms
            if term in standalone_terms and term not in original_terms
        ]
        if inherited_terms and not event.inferred_information:
            event.inferred_information = [
                f'Từ common_query: {context_prefix}.'
            ]

    _validate_standalone_context(repaired, common_query)
    # The mutations above are plain attribute assignment on an already-
    # validated model (validate_assignment is not enabled), so a field's own
    # schema constraints - max_length in particular, now that context_prefix
    # has been prepended - are never re-checked. A full model round-trip
    # forces that check: any now-oversized field raises here (caught by
    # _repair_standalone_context_or_fail below) instead of shipping in the
    # HTTP response as if it were still schema-valid.
    return RewriteResponse.model_validate(repaired.model_dump())


def _repair_standalone_context_or_fail(
    analysis: RewriteResponse,
    common_query: str | None,
) -> RewriteResponse:
    """Wraps `_repair_standalone_context` so a repair that still can't
    satisfy full validation - its heuristic text-prefixing reconstruction is
    not guaranteed to succeed in every case - surfaces as the same
    already-handled `OllamaServiceError` every other exhausted-retry path in
    `rewrite_queries` already raises, not an uncaught `SemanticValidationError`
    or `ValidationError` (a 500, not the intended error response)."""

    try:
        return _repair_standalone_context(analysis, common_query)
    except (SemanticValidationError, ValidationError) as exc:
        raise OllamaServiceError(
            'Ollama returned structurally valid but semantically inconsistent '
            f'output that automatic repair could not fix: {exc}'
        ) from exc


def _parse_analysis(
    content: str,
    queries: Sequence[str],
) -> RewriteResponse:
    payload = _extract_json_object(content)
    analysis = RewriteResponse.model_validate(payload)
    _validate_source_alignment(analysis, queries)
    return analysis


def _format_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        messages: list[str] = []
        for error in exc.errors(include_url=False)[:12]:
            location = '.'.join(str(part) for part in error['loc'])
            messages.append(f'{location}: {error["msg"]}')
        return '\n'.join(messages)
    return str(exc)[:2000]


async def rewrite_queries(
    *,
    queries: Sequence[str],
    common_query: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RewriteResponse:
    '''Analyze all queries in one Ollama call and return validated structured data.'''
    load_ollama_env()

    resolved_api_key = (
        api_key if api_key is not None else os.getenv('OLLAMA_API_KEY')
    )
    if not resolved_api_key or not resolved_api_key.strip():
        raise ConfigurationError('OLLAMA_API_KEY is not configured')

    resolved_api_url = api_url or os.getenv('OLLAMA_API_URL', OLLAMA_API_URL)
    resolved_model = (
        os.getenv('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL).strip()
        or DEFAULT_OLLAMA_MODEL
    )
    headers = {
        'Authorization': f'Bearer {resolved_api_key.strip()}',
        'Content-Type': 'application/json',
    }
    previous_response: str | None = None
    validation_errors: str | None = None
    last_structured_analysis: RewriteResponse | None = None
    last_semantic_error: str | None = None
    # Separate counters (MAX_TRANSPORT_ATTEMPTS / MAX_VALIDATION_ATTEMPTS) -
    # a transport blip and a validation failure are unrelated failure
    # categories that used to share one budget, so a single network hiccup
    # could consume attempts the repair-retry loop needed. Reaching either
    # cap ends the function (raise, or fall through to the final fallback
    # below); nothing else in the loop body increments a counter without
    # also returning/raising/breaking, so this can't spin forever.
    transport_attempts = 0
    validation_attempts = 0

    async with httpx.AsyncClient(
        headers=headers,
        timeout=OLLAMA_TIMEOUT_SECONDS,
        transport=transport,
    ) as client:
        while True:
            payload = {
                'model': resolved_model,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {
                        'role': 'user',
                        'content': build_analysis_prompt(
                            queries=queries,
                            common_query=common_query,
                            previous_response=previous_response,
                            validation_errors=validation_errors,
                        ),
                    },
                ],
                'stream': False,
                'think': 'low',
                'options': {'temperature': 0},
            }

            try:
                response = await client.post(resolved_api_url, json=payload)
            except httpx.TimeoutException as exc:
                transport_attempts += 1
                logger.warning(
                    'rewrite_failed category=ollama_timeout attempt=%s/%s',
                    transport_attempts,
                    MAX_TRANSPORT_ATTEMPTS,
                )
                if (
                    last_structured_analysis is not None
                    and last_semantic_error is not None
                ):
                    return _repair_standalone_context_or_fail(
                        last_structured_analysis,
                        common_query,
                    )
                if transport_attempts < MAX_TRANSPORT_ATTEMPTS:
                    await asyncio.sleep(OLLAMA_RETRY_DELAY_SECONDS)
                    continue
                raise OllamaTimeoutError('Ollama request timed out') from exc
            except httpx.RequestError as exc:
                transport_attempts += 1
                logger.warning(
                    'rewrite_failed category=ollama_connection_failed '
                    'attempt=%s/%s',
                    transport_attempts,
                    MAX_TRANSPORT_ATTEMPTS,
                )
                if (
                    last_structured_analysis is not None
                    and last_semantic_error is not None
                ):
                    return _repair_standalone_context_or_fail(
                        last_structured_analysis,
                        common_query,
                    )
                if transport_attempts < MAX_TRANSPORT_ATTEMPTS:
                    await asyncio.sleep(OLLAMA_RETRY_DELAY_SECONDS)
                    continue
                raise OllamaServiceError('Could not connect to Ollama') from exc

            if response.status_code == 429:
                transport_attempts += 1
                logger.warning(
                    'rewrite_failed category=ollama_rate_limited '
                    'attempt=%s/%s',
                    transport_attempts,
                    MAX_TRANSPORT_ATTEMPTS,
                )
                if (
                    last_structured_analysis is not None
                    and last_semantic_error is not None
                ):
                    return _repair_standalone_context_or_fail(
                        last_structured_analysis,
                        common_query,
                    )
                if transport_attempts < MAX_TRANSPORT_ATTEMPTS:
                    await asyncio.sleep(OLLAMA_RETRY_DELAY_SECONDS)
                    continue
                raise OllamaRateLimitError('Ollama rate limit exceeded')
            if response.is_error:
                transport_attempts += 1
                logger.warning(
                    'rewrite_failed category=ollama_http_error status=%s '
                    'attempt=%s/%s',
                    response.status_code,
                    transport_attempts,
                    MAX_TRANSPORT_ATTEMPTS,
                )
                if (
                    last_structured_analysis is not None
                    and last_semantic_error is not None
                ):
                    return _repair_standalone_context_or_fail(
                        last_structured_analysis,
                        common_query,
                    )
                if (
                    response.status_code in TRANSIENT_OLLAMA_STATUSES
                    and transport_attempts < MAX_TRANSPORT_ATTEMPTS
                ):
                    await asyncio.sleep(OLLAMA_RETRY_DELAY_SECONDS)
                    continue
                raise OllamaServiceError(
                    f'Ollama returned HTTP status {response.status_code}'
                )

            try:
                body = response.json()
                content = body['message']['content']
            except (KeyError, TypeError, ValueError) as exc:
                raise OllamaServiceError(
                    'Ollama returned an invalid API response'
                ) from exc

            if not isinstance(content, str) or not content.strip():
                raise OllamaServiceError('Ollama returned an empty analysis')

            try:
                analysis = _parse_analysis(content, queries)
            except (ValidationError, ValueError) as exc:
                validation_attempts += 1
                previous_response = content
                validation_errors = _format_validation_error(exc)
                logger.warning(
                    'rewrite_validation_failed kind=structure '
                    'attempt=%s/%s',
                    validation_attempts,
                    MAX_VALIDATION_ATTEMPTS,
                )
                if validation_attempts >= MAX_VALIDATION_ATTEMPTS:
                    break
                continue

            last_structured_analysis = analysis
            try:
                _validate_standalone_context(analysis, common_query)
            except SemanticValidationError as exc:
                validation_attempts += 1
                previous_response = content
                validation_errors = _format_validation_error(exc)
                last_semantic_error = validation_errors
                logger.warning(
                    'rewrite_validation_failed kind=semantic '
                    'attempt=%s/%s',
                    validation_attempts,
                    MAX_VALIDATION_ATTEMPTS,
                )
                if validation_attempts >= MAX_VALIDATION_ATTEMPTS:
                    break
                continue

            return analysis

    if last_structured_analysis is not None and last_semantic_error is not None:
        logger.warning(
            'rewrite_context_repair_applied after_attempts=%s',
            validation_attempts,
        )
        return _repair_standalone_context_or_fail(
            last_structured_analysis,
            common_query,
        )

    error_detail = validation_errors or 'unknown validation error'
    logger.error(
        'rewrite_failed category=ollama_invalid_output attempts=%s',
        validation_attempts,
    )
    raise OllamaServiceError(
        'Ollama returned invalid structured analysis after retry: '
        + error_detail
    )
