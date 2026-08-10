"""Small standard-library client for this repo's own FastAPI backend.

Distinct from `client.py`'s `SearchClient`, which talks to the external
sparse frame-search service (port 8000). This client talks to the backend
built in this repo (`src/main.py`, port 8001 by default): the legacy
`/temporal-search` endpoint and the adaptive `/v1/search-sessions/*` API.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


class BackendApiError(RuntimeError):
    """Raised for transport failures or an invalid backend response."""


LegacySearcherType = Literal["TemporalSearcher", "AmbiguousSearcher"]


@dataclass(frozen=True)
class LegacyTupleResult:
    score: float
    video_name: str
    frames: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RefineOutcome:
    available: bool
    session_revision: int | None = None
    metrics: Mapping[str, Any] | None = None
    unavailable_reason: str | None = None


class TemporalSearchBackendClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout_seconds: float = 60.0,
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

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[int, Any]:
        """Issue one request, retrying transient (5xx/network) failures.

        Returns `(status_code, decoded_json_body)` for ANY response the
        server sent, including 4xx - callers distinguish expected 4xx
        outcomes (like a 503 "live refinement unavailable") from genuine
        transport failures, which raise `BackendApiError` instead.
        """

        data = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {
            "Accept": "application/json",
            "User-Agent": "youcook2-tuple-eval-benchmark/1",
        }
        if data is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"

        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}{path}", data=data, headers=headers, method=method
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    body = json.loads(raw.decode("utf-8")) if raw else None
                    return response.status, body
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else None
                except json.JSONDecodeError:
                    body = raw.decode("utf-8", errors="replace")
                # Unlike SearchClient (an external, possibly-flaky service),
                # this client only ever talks to this repo's own backend,
                # which uses 4xx/5xx deliberately for business states (e.g.
                # 503 live_refinement_unavailable). Never retry any HTTP
                # response - only genuine transport failures below - so
                # callers like refine() can see and interpret every status.
                return exc.code, body
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise BackendApiError(f"request failed after {self.retries + 1} attempt(s): {last_error}")

    # -- legacy pipeline --------------------------------------------------

    def legacy_temporal_search(
        self,
        queries: Sequence[str],
        *,
        top_k_tuple: int,
        top_k_each_query: int,
        gamma: float,
        searcher_type: LegacySearcherType,
    ) -> list[LegacyTupleResult]:
        payload = {
            "query": list(queries),
            "top_k_tuple": top_k_tuple,
            "top_k_each_query": top_k_each_query,
            "gamma": gamma,
            "searcher_type": searcher_type,
        }
        status, body = self._request("POST", "/temporal-search", payload)
        if status != 200 or not isinstance(body, Mapping):
            raise BackendApiError(f"POST /temporal-search returned HTTP {status}: {body}")
        results = body.get("results")
        if not isinstance(results, list):
            raise BackendApiError("POST /temporal-search response is missing a results list")
        return [
            LegacyTupleResult(
                score=float(item["score"]),
                video_name=str(item["video_name"]),
                frames=tuple(item["tuple"]),
            )
            for item in results
        ]

    # -- adaptive pipeline -------------------------------------------------

    def create_adaptive_session(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        common_query: str | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"events": list(events)}
        if common_query is not None:
            payload["common_query"] = common_query
        if hyperparameters is not None:
            payload["hyperparameters"] = hyperparameters
        status, body = self._request("POST", "/v1/search-sessions", payload)
        if status != 201 or not isinstance(body, Mapping):
            raise BackendApiError(f"POST /v1/search-sessions returned HTTP {status}: {body}")
        session = body.get("session")
        session_id = session.get("id") if isinstance(session, Mapping) else None
        if not session_id:
            raise BackendApiError("POST /v1/search-sessions response is missing session.id")
        return str(session_id)

    def retrieve(self, session_id: str, *, top_k: int) -> int:
        status, body = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/commands/retrieve",
            {"top_k": top_k},
        )
        if status != 200 or not isinstance(body, Mapping):
            raise BackendApiError(f"POST .../commands/retrieve returned HTTP {status}: {body}")
        return int(body["session_revision"])

    def get_video_priorities(self, session_id: str) -> list[Mapping[str, Any]]:
        status, body = self._request("GET", f"/v1/search-sessions/{session_id}/video-priorities")
        if status != 200 or not isinstance(body, Mapping):
            raise BackendApiError(f"GET .../video-priorities returned HTTP {status}: {body}")
        items = body.get("items")
        if not isinstance(items, list):
            raise BackendApiError(".../video-priorities response is missing an items list")
        return items

    def refine(self, session_id: str, *, expected_revision: int) -> RefineOutcome:
        status, body = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/commands/refine",
            {"expected_revision": expected_revision},
        )
        if status in (503, 422) and isinstance(body, Mapping):
            detail = body.get("detail")
            code = detail.get("code") if isinstance(detail, Mapping) else None
            if isinstance(code, str) and code.startswith("live_refinement_"):
                message = detail.get("message") if isinstance(detail, Mapping) else None
                return RefineOutcome(available=False, unavailable_reason=str(message or code))
        if status != 200 or not isinstance(body, Mapping):
            raise BackendApiError(f"POST .../commands/refine returned HTTP {status}: {body}")
        return RefineOutcome(
            available=True,
            session_revision=int(body["session_revision"]),
            metrics=body.get("metrics"),
        )

    def get_tuples(self, session_id: str) -> list[Mapping[str, Any]]:
        status, body = self._request("GET", f"/v1/search-sessions/{session_id}/tuples")
        if status != 200 or not isinstance(body, Mapping):
            raise BackendApiError(f"GET .../tuples returned HTTP {status}: {body}")
        items = body.get("items")
        if not isinstance(items, list):
            raise BackendApiError(".../tuples response is missing an items list")
        return items
