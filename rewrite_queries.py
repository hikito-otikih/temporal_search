import asyncio
import os
import re
import unicodedata
from pathlib import Path
from typing import Sequence

import httpx
from dotenv import load_dotenv


OLLAMA_API_URL = 'https://ollama.com/api/chat'
OLLAMA_TIMEOUT_SECONDS = 60.0
MAX_CONCURRENT_REQUESTS = 4
MAX_REWRITE_ATTEMPTS = 2
MIN_CONTEXT_TERM_COVERAGE = 0.75

GENERIC_CONTEXT_TERMS = frozenset(
    '''
    a an and are as at by clip common context event events find following for
    from in into is it locate of on query scene search showing the this to video
    with
    các cái cảnh chung có con của đây đoạn được hãy kiếm là một những ở sau sự
    kiện theo tìm trong truy vấn và video với
    '''.split()
)
WORD_PATTERN = re.compile(r'[^\W_]+', flags=re.UNICODE)
VIETNAMESE_CONTEXT_SUFFIX = re.compile(
    r'[,;:\s]*(?:hãy\s+)?tìm(?:\s+kiếm)?\s+(?:các|những)?\s*'
    r'sự\s+kiện(?:\s+sau(?:\s+đây)?)?\s*[:.!?]*$',
    flags=re.IGNORECASE,
)
ENGLISH_CONTEXT_SUFFIX = re.compile(
    r'[,;:\s]*(?:find|identify|locate)\s+(?:the\s+)?following\s+'
    r'events?\s*[:.!?]*$',
    flags=re.IGNORECASE,
)


class ConfigurationError(RuntimeError):
    '''Raised when the server is missing required Ollama configuration.'''


class OllamaServiceError(RuntimeError):
    '''Raised when Ollama cannot return a usable rewrite.'''


class OllamaTimeoutError(OllamaServiceError):
    '''Raised when an Ollama request exceeds the configured timeout.'''


class OllamaRateLimitError(OllamaServiceError):
    '''Raised when Ollama rejects a request because of rate limiting.'''


def _normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize('NFKC', text).casefold()
    return WORD_PATTERN.findall(normalized)


def _required_context_terms(common_query: str | None) -> list[str]:
    if not common_query:
        return []

    terms: list[str] = []
    for word in _normalized_words(common_query):
        if len(word) < 2 or word in GENERIC_CONTEXT_TERMS or word in terms:
            continue
        terms.append(word)
    return terms[:16]


def _missing_context_terms(
    rewritten_query: str, common_query: str | None
) -> list[str]:
    rewritten_words = set(_normalized_words(rewritten_query))
    return [
        term
        for term in _required_context_terms(common_query)
        if term not in rewritten_words
    ]


def _has_required_context(
    rewritten_query: str, common_query: str | None
) -> bool:
    required_terms = _required_context_terms(common_query)
    if not required_terms:
        return True

    missing_terms = _missing_context_terms(rewritten_query, common_query)
    matched_count = len(required_terms) - len(missing_terms)
    minimum_matches = max(
        1, int(len(required_terms) * MIN_CONTEXT_TERM_COVERAGE + 0.999999)
    )
    return matched_count >= minimum_matches


def _clean_common_context(common_query: str) -> str:
    context = common_query.strip()
    context = VIETNAMESE_CONTEXT_SUFFIX.sub('', context)
    context = ENGLISH_CONTEXT_SUFFIX.sub('', context)
    return context.rstrip(' \t\r\n,;:.!?')


def _contextual_fallback(common_query: str, rewritten_query: str) -> str:
    context = _clean_common_context(common_query)
    if not context:
        return rewritten_query.strip()
    return f'{context}. {rewritten_query.strip()}'


SYSTEM_PROMPT = '''Bạn là bộ máy MỞ RỘNG NGỮ CẢNH cho truy vấn tìm kiếm sự kiện
trong video, không phải bộ sửa chính tả hay diễn đạt.

Viết lại TARGET_QUERY thành một truy vấn TỰ CHỨA. Người đọc chỉ được nhìn đầu ra,
không được xem COMMON_QUERY hoặc ALL_QUERIES, nhưng vẫn phải hiểu:
1. Video/bối cảnh tổng quát nói về hoạt động gì.
2. Chủ thể là ai hoặc vật gì, gồm đặc điểm nhận diện như màu sắc, vị trí, số thứ
   tự hoặc vai trò đã có trong dữ liệu.
3. Hành động hoặc trạng thái chính xác cần tìm.
4. Ranh giới thời gian như đầu tiên, cuối cùng, bắt đầu, hoàn toàn, trước hoặc sau.

QUY TẮC BẮT BUỘC:
- COMMON_QUERY và ALL_QUERIES là nguồn dữ kiện, KHÔNG phải ngữ cảnh ẩn. Khi
  COMMON_QUERY có nội dung, đầu ra phải trực tiếp nhắc lại hoạt động, chủ thể và
  các thuộc tính nhận diện cốt lõi của nó.
- Giữ các từ nhận diện quan trọng từ COMMON_QUERY. Ví dụ: múa lân, màu vàng, màu
  đen, màu trắng.
- Giải tham chiếu mơ hồ như nó, con này, 4 chân, sau đó hoặc chủ thể bị lược bỏ
  bằng thông tin cụ thể từ COMMON_QUERY hay ALL_QUERIES.
- Giữ nguyên chủ thể, hành động, số đếm, trình tự và mọi từ chỉ biên thời gian.
  Không làm mất hoặc làm yếu đầu tiên, cuối cùng, bắt đầu, hoàn toàn, rời khỏi.
- Không thêm dữ kiện không có trong đầu vào. Không gộp hoặc tách sự kiện.
- Chỉ đổi số thành chữ, đảo từ, sửa ngữ pháp hoặc thêm một chủ thể ngắn KHÔNG
  được xem là thành công. Đầu ra phải thực sự thêm bối cảnh còn thiếu.
- Thường nên bắt đầu bằng một mệnh đề như Trong đoạn video ...
- Nếu TARGET_QUERY có nhãn E1, E2, ... thì giữ nguyên nhãn. Nếu nó chứa nhiều
  mục có nhãn, mỗi mục phải tự lặp lại bối cảnh cần thiết và tự đủ nghĩa.
- Trả lời bằng đúng ngôn ngữ của TARGET_QUERY. Chỉ trả về truy vấn đã viết lại,
  không giải thích, không Markdown, không JSON và không đặt trong dấu nháy.

VÍ DỤ:
COMMON_QUERY:
Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau.

TARGET_QUERY:
Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.

SAI:
Khoảnh khắc đầu tiên 4 chân của lân hoàn toàn chạm đất.

ĐÚNG:
Trong đoạn video múa lân với một con lân màu vàng, đen và trắng, khoảnh khắc đầu
tiên cả bốn chân của con lân hoàn toàn chạm đất.

TARGET_QUERY:
Sau đó lân tiến lại chào một con rồng. Khoảnh khắc đầu tiên con rồng cử động đầu.

ĐÚNG:
Trong đoạn video múa lân với một con lân màu vàng, đen và trắng, sau khi con lân
tiến lại chào một con rồng, khoảnh khắc đầu tiên con rồng bắt đầu cử động đầu.

Trước khi trả lời, tự kiểm tra thầm: chỉ nhìn đầu ra có biết đây là video gì, có
nhận diện đúng chủ thể và có giữ đủ điều kiện thời gian không? Nếu không, viết
lại trước khi xuất kết quả.
'''


def build_rewrite_prompt(
    queries: Sequence[str],
    target_index: int,
    common_query: str | None = None,
    previous_draft: str | None = None,
) -> str:
    '''Build a prompt containing shared context and one explicit rewrite target.'''
    numbered_queries = '\n'.join(
        f'{index + 1}. {query}' for index, query in enumerate(queries)
    )
    required_terms = _required_context_terms(common_query)
    required_terms_text = ', '.join(required_terms) or '(không có)'
    common_context = common_query or '(không có)'

    prompt = f'''<COMMON_QUERY>
{common_context}
</COMMON_QUERY>

<REQUIRED_CONTEXT_TERMS>
{required_terms_text}
</REQUIRED_CONTEXT_TERMS>

<ALL_QUERIES>
{numbered_queries}
</ALL_QUERIES>

<TARGET_QUERY>
{queries[target_index]}
</TARGET_QUERY>

Viết lại duy nhất TARGET_QUERY thành một truy vấn tự chứa.
BẮT BUỘC đưa trực tiếp các dữ kiện nhận diện cốt lõi từ COMMON_QUERY đang thiếu
trong TARGET_QUERY vào đầu ra. Các từ trong REQUIRED_CONTEXT_TERMS là neo ngữ
nghĩa cần được giữ lại khi tự nhiên. Không chỉ sửa ngữ pháp hoặc đổi cách viết.
'''

    if previous_draft is not None:
        missing_terms = _missing_context_terms(previous_draft, common_query)
        missing_text = ', '.join(missing_terms) or '(ngữ cảnh vẫn chưa đủ rõ)'
        prompt += f'''
<PREVIOUS_REJECTED_DRAFT>
{previous_draft}
</PREVIOUS_REJECTED_DRAFT>

Bản trước bị từ chối vì chưa trực tiếp chứa đủ bối cảnh chung.
Các neo ngữ nghĩa còn thiếu: {missing_text}.
Hãy viết lại từ đầu, nêu rõ bối cảnh, chủ thể và đặc điểm nhận diện; không chỉ
chỉnh sửa bản bị từ chối. Chỉ trả về truy vấn mới.
'''

    return prompt


async def _rewrite_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    modelname: str,
    queries: Sequence[str],
    target_index: int,
    common_query: str | None,
    api_url: str,
) -> str:
    previous_draft: str | None = None

    for _ in range(MAX_REWRITE_ATTEMPTS):
        payload = {
            'model': modelname,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {
                    'role': 'user',
                    'content': build_rewrite_prompt(
                        queries=queries,
                        target_index=target_index,
                        common_query=common_query,
                        previous_draft=previous_draft,
                    ),
                },
            ],
            'stream': False,
            'think': False,
            'options': {'temperature': 0},
        }

        try:
            async with semaphore:
                response = await client.post(api_url, json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError('Ollama request timed out') from exc
        except httpx.RequestError as exc:
            raise OllamaServiceError('Could not connect to Ollama') from exc

        if response.status_code == 429:
            raise OllamaRateLimitError('Ollama rate limit exceeded')
        if response.is_error:
            raise OllamaServiceError(
                f'Ollama returned HTTP status {response.status_code}'
            )

        try:
            body = response.json()
            content = body['message']['content']
        except (KeyError, TypeError, ValueError) as exc:
            raise OllamaServiceError('Ollama returned an invalid response') from exc

        if not isinstance(content, str) or not content.strip():
            raise OllamaServiceError('Ollama returned an empty rewrite')

        candidate = content.strip()
        if _has_required_context(candidate, common_query):
            return candidate
        previous_draft = candidate

    if common_query is not None and previous_draft is not None:
        return _contextual_fallback(common_query, previous_draft)

    if previous_draft is not None:
        return previous_draft
    raise OllamaServiceError('Ollama did not return a usable rewrite')


async def rewrite_queries(
    *,
    modelname: str,
    queries: Sequence[str],
    common_query: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[str]:
    '''Rewrite each query independently while supplying the shared context.'''
    load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

    resolved_api_key = (
        api_key if api_key is not None else os.getenv('OLLAMA_API_KEY')
    )
    if not resolved_api_key or not resolved_api_key.strip():
        raise ConfigurationError('OLLAMA_API_KEY is not configured')

    resolved_api_url = api_url or os.getenv('OLLAMA_API_URL', OLLAMA_API_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    headers = {
        'Authorization': f'Bearer {resolved_api_key.strip()}',
        'Content-Type': 'application/json',
    }

    async with httpx.AsyncClient(
        headers=headers,
        timeout=OLLAMA_TIMEOUT_SECONDS,
        transport=transport,
    ) as client:
        results = await asyncio.gather(
            *(
                _rewrite_one(
                    client,
                    semaphore,
                    modelname=modelname,
                    queries=queries,
                    target_index=index,
                    common_query=common_query,
                    api_url=resolved_api_url,
                )
                for index in range(len(queries))
            ),
            return_exceptions=True,
        )

    for result in results:
        if isinstance(result, BaseException):
            raise result

    return [result for result in results if isinstance(result, str)]
