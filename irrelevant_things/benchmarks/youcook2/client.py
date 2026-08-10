"""Small standard-library client for the upstream frame-search service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


class SearchApiError(RuntimeError):
    """Raised for transport failures or an invalid API response."""


@dataclass(frozen=True)
class SearchResponse:
    hits: list[Mapping[str, Any]]
    response_query: str | None
    english_query: str | None
    response_top_k: int | None


class SearchClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 30.0,
        retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        self.base_url = base_url.rstrip("/.")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def _request_json(self, request: urllib.request.Request) -> Any:
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read(2048).decode("utf-8", errors="replace")
                last_error = SearchApiError(f"HTTP {exc.code} from {request.full_url}: {body}")
                # Invalid requests should not be retried; transient server errors should.
                if exc.code < 500:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise SearchApiError(f"request failed after {self.retries + 1} attempt(s): {last_error}")

    def health(self) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"Accept": "application/json", "User-Agent": "youcook2-video-recall-benchmark/1"},
            method="GET",
        )
        value = self._request_json(request)
        if not isinstance(value, Mapping):
            raise SearchApiError("GET /health did not return a JSON object")
        return value

    def search(self, query: str, top_k: int) -> SearchResponse:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        # Deliberately allow only retrieval inputs in this payload.  In
        # particular, query ids, video paths, answer intervals, and GT video ids
        # never cross the API boundary.
        payload = {"query": query, "top_k": int(top_k)}
        request = urllib.request.Request(
            f"{self.base_url}/search",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "youcook2-video-recall-benchmark/1",
            },
            method="POST",
        )
        value = self._request_json(request)
        if not isinstance(value, Mapping):
            raise SearchApiError("POST /search did not return a JSON object")
        hits = value.get("results")
        if not isinstance(hits, list):
            raise SearchApiError("POST /search response is missing a results list")
        if not all(isinstance(hit, Mapping) for hit in hits):
            raise SearchApiError("POST /search results must contain JSON objects")
        if len(hits) > top_k:
            raise SearchApiError(
                f"POST /search returned {len(hits)} results for requested top_k={top_k}"
            )
        response_top_k = value.get("top_k")
        if response_top_k is None:
            parsed_top_k = None
        else:
            if isinstance(response_top_k, bool) or (
                isinstance(response_top_k, float)
                and not response_top_k.is_integer()
            ):
                raise SearchApiError("POST /search returned an invalid top_k")
            try:
                parsed_top_k = int(response_top_k)
            except (TypeError, ValueError) as exc:
                raise SearchApiError(
                    "POST /search returned an invalid top_k"
                ) from exc
            if parsed_top_k != top_k:
                raise SearchApiError(
                    f"POST /search echoed top_k={parsed_top_k}, expected {top_k}"
                )
        return SearchResponse(
            hits=hits,
            response_query=str(value["query"]) if value.get("query") is not None else None,
            english_query=str(value["english_query"]) if value.get("english_query") is not None else None,
            response_top_k=parsed_top_k,
        )
