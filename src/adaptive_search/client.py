"""Validated adapter for the sparse frame-search service on port 8000.

This module is independent of the legacy client. One query per event - no
fusion across variants (that used to happen here has been removed)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import PurePosixPath
import socket
from typing import Any, Protocol
from urllib import error, request

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .exceptions import UpstreamSearchError
from .schemas import SparseCandidate


class UpstreamHit(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, str_strip_whitespace=True)

    video_name: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    timestamp: str | None = None
    timestamp_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    score: float = Field(allow_inf_nan=False)
    watch_url: str | None = None


class UpstreamResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, str_strip_whitespace=True)

    # /text_retrieve echoes the query back as a one-element list (its
    # sibling endpoint /text_rrf_generated_queries_search shares this same
    # response schema and echoes multiple expanded queries there) - not a
    # single string.
    query: list[str] = Field(min_length=1)
    results: list[UpstreamHit]


@dataclass(frozen=True)
class EventQuery:
    event_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if not self.text.strip():
            raise ValueError("query text must not be empty")


class TimestampResolver(Protocol):
    def __call__(
        self, video_id: str, frame_index: int, fallback_seconds: float
    ) -> float: ...


Transport = Callable[[str, dict[str, Any], float], Mapping[str, Any]]


class UpstreamSearchClient:
    """Synchronous client with an injectable transport for offline tests."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout_seconds: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        self.base_url = normalized
        self.timeout_seconds = timeout_seconds
        self._transport = transport or _urlopen_json

    def health(self) -> dict[str, Any]:
        payload = self._transport(
            f"{self.base_url}/health",
            {"__http_method__": "GET"},
            self.timeout_seconds,
        )
        return dict(payload)

    def search(self, query_text: str, top_k: int) -> UpstreamResponse:
        text = query_text.strip()
        if not text:
            raise ValueError("query_text must not be empty")
        if not 1 <= top_k <= 10_000:
            raise ValueError("top_k must be in [1, 10000]")
        try:
            payload = self._transport(
                f"{self.base_url}/text_retrieve",
                {"query": text, "top_k": top_k},
                self.timeout_seconds,
            )
            response = UpstreamResponse.model_validate(payload)
        except ValidationError as exc:
            raise UpstreamSearchError(
                "upstream /text_retrieve returned an invalid response schema"
            ) from exc
        if response.query != [text]:
            raise UpstreamSearchError(
                "upstream /text_retrieve echoed a different query"
            )
        if len(response.results) > top_k:
            raise UpstreamSearchError(
                "upstream /text_retrieve returned more hits than requested top_k"
            )
        return response

    def retrieve_candidates(
        self,
        *,
        session_id: str,
        events: Sequence[EventQuery],
        top_k: int,
        timestamp_resolver: TimestampResolver | None = None,
    ) -> list[SparseCandidate]:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        candidates: list[SparseCandidate] = []
        for event_query in events:
            response = self.search(event_query.text, top_k)
            for rank, hit in enumerate(response.results, start=1):
                video_id = normalize_video_id(hit.video_name)
                fallback = (
                    hit.timestamp_seconds
                    if hit.timestamp_seconds is not None
                    else parse_display_timestamp(hit.timestamp)
                )
                timestamp = (
                    timestamp_resolver(video_id, hit.frame_index, fallback)
                    if timestamp_resolver is not None
                    else fallback
                )
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise UpstreamSearchError(
                        "timestamp resolver returned an invalid timestamp"
                    )
                candidates.append(
                    SparseCandidate(
                        id=_candidate_id(
                            session_id,
                            event_query.event_id,
                            video_id,
                            hit.frame_index,
                            rank,
                        ),
                        session_id=session_id,
                        event_id=event_query.event_id,
                        video_id=video_id,
                        frame_id=hit.frame_index,
                        timestamp_seconds=timestamp,
                        raw_relevance_score=hit.score,
                        watch_url=hit.watch_url,
                    )
                )
        return candidates


def parse_display_timestamp(value: str | None) -> float:
    """Parse ``SS``, ``MM:SS`` or ``HH:MM:SS`` legacy timestamps."""

    if value is None or not value.strip():
        raise UpstreamSearchError(
            "upstream hit has neither timestamp_seconds nor timestamp"
        )
    parts = value.strip().split(":")
    if len(parts) > 3:
        raise UpstreamSearchError(f"invalid upstream timestamp: {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise UpstreamSearchError(f"invalid upstream timestamp: {value!r}") from exc
    if any(not math.isfinite(item) or item < 0 for item in numbers):
        raise UpstreamSearchError(f"invalid upstream timestamp: {value!r}")
    if len(numbers) > 1 and numbers[-1] >= 60:
        raise UpstreamSearchError(f"invalid upstream timestamp: {value!r}")
    if len(numbers) == 3 and numbers[-2] >= 60:
        raise UpstreamSearchError(f"invalid upstream timestamp: {value!r}")
    padded = [0.0] * (3 - len(numbers)) + numbers
    hours, minutes, seconds = padded
    return hours * 3600.0 + minutes * 60.0 + seconds


def normalize_video_id(video_name: str) -> str:
    normalized = video_name.strip().replace("\\", "/")
    filename = PurePosixPath(normalized).name
    if filename.lower().endswith(".mp4"):
        filename = filename[:-4]
    if not filename:
        raise UpstreamSearchError("upstream video_name has no usable identifier")
    return filename


def _candidate_id(
    session_id: str,
    event_id: str,
    video_id: str,
    frame_index: int,
    rank: int,
) -> str:
    payload = "\x1f".join(
        (session_id, event_id, video_id, str(frame_index), str(rank))
    ).encode("utf-8")
    return f"candidate_upstream_{sha256(payload).hexdigest()[:20]}"


def _urlopen_json(
    url: str, payload: dict[str, Any], timeout_seconds: float
) -> Mapping[str, Any]:
    payload = dict(payload)
    method = str(payload.pop("__http_method__", "POST"))
    body = None if method == "GET" else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise UpstreamSearchError(
            f"upstream request failed with HTTP {exc.code}"
        ) from exc
    except (error.URLError, socket.timeout, TimeoutError) as exc:
        raise UpstreamSearchError("upstream request failed or timed out") from exc
    except UnicodeDecodeError as exc:
        # A ValueError subclass - left uncaught, this would otherwise reach
        # _raise_api_error's generic ValueError fallback and get reported as
        # a 422 client input error, when invalid UTF-8 is an upstream
        # response problem, not something wrong with the request this
        # service sent.
        raise UpstreamSearchError("upstream returned invalid UTF-8") from exc
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UpstreamSearchError("upstream returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise UpstreamSearchError("upstream JSON response must be an object")
    return decoded
