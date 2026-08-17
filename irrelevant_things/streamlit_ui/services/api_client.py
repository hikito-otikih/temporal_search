"""Typed HTTP client for the Temporal Search FastAPI backend.

Every UI page goes through this client so that timeout handling, error mapping,
and optimistic-revision handling stay in one place.  Components never call
``httpx`` directly.

Error contract
--------------
``ApiError`` subclasses are raised for non-2xx responses and transport
failures.  ``RevisionConflictError`` carries the expected/current revisions so
the UI can refresh the session and ask the user to re-apply the change instead
of silently retrying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import httpx

from models.ui_models import (
    CapabilityResponse,
    MutationResponse,
    RunResponse,
    SessionResponse,
)


class ApiError(Exception):
    """Base class for all UI<->backend communication errors."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id


class ConnectionFailure(ApiError):
    """Transport-level failure (DNS, refused connection, timeouts)."""


class ValidationFailure(ApiError):
    """FastAPI 422 with a machine readable detail payload."""

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail


class HttpFailure(ApiError):
    def __init__(self, status_code: int, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class SessionNotFound(HttpFailure):
    pass


class RevisionConflictError(ApiError):
    def __init__(self, expected: int, current: int) -> None:
        super().__init__(
            f"stale session revision: expected {expected}, current revision is {current}"
        )
        self.expected = expected
        self.current = current


@dataclass(frozen=True)
class ClientSettings:
    backend_url: str = "http://127.0.0.1:8001"
    upstream_url: str = "http://127.0.0.1:8000"
    connect_timeout: float = 5.0
    # GET /video-priorities and POST /temporal-search with
    # apply_boundary_refinement=true do real per-event GPU decode+embed work
    # that scales with result count - e.g. limit=50 with 3 events measured at
    # ~65s. 60s was cutting off requests that were still correctly working,
    # just slower than the client was willing to wait.
    read_timeout: float = 240.0
    max_retries: int = 1


class TemporalApiClient:
    """Synchronous, state-free client.  Cache one instance via st.cache_resource."""

    def __init__(
        self,
        settings: ClientSettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or ClientSettings()
        self._backend = self.settings.backend_url.rstrip("/")
        self._upstream = self.settings.upstream_url.rstrip("/")
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                connect=self.settings.connect_timeout,
                read=self.settings.read_timeout,
                write=self.settings.read_timeout,
                pool=self.settings.read_timeout,
            ),
            transport=transport,
        )

    # ------------------------------------------------------------------
    # Capability / health
    # ------------------------------------------------------------------

    def check_backend_health(self) -> dict[str, Any]:
        return self._request("GET", "/")

    def get_capabilities(self) -> CapabilityResponse:
        data = self._request("GET", "/v1/searchers")
        return CapabilityResponse.model_validate(data)

    def check_upstream_health(self) -> dict[str, Any]:
        return self._request("GET", "/health", base=self._upstream)

    def upstream_search(self, query: str, top_k: int = 100) -> dict[str, Any]:
        """Raw frame-search call used only by the YouCook2 evaluation page.

        The payload is intentionally minimal (text + top-K) so ground-truth
        video identities never reach the retrieval index.
        """
        return self._request("POST", "/search", base=self._upstream, json={"query": query, "top_k": top_k})

    # ------------------------------------------------------------------
    # Legacy
    # ------------------------------------------------------------------

    def legacy_search(
        self,
        *,
        queries: list[str],
        top_k_tuple: int = 100,
        top_k_each_query: int = 100,
        gamma: float = 0.05,
        searcher_type: str = "TemporalSearcher",
        object_filter: bool = False,
        object_name_list: list[str] | None = None,
        object_threshold: float = 0.5,
        apply_boundary_refinement: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/temporal-search",
            json={
                "query": queries,
                "top_k_tuple": top_k_tuple,
                "top_k_each_query": top_k_each_query,
                "gamma": gamma,
                "searcher_type": searcher_type,
                "objectFilterMode": object_filter,
                "object_name_list": object_name_list,
                "objectThreshold": object_threshold,
                "apply_boundary_refinement": apply_boundary_refinement,
            },
        )

    # ------------------------------------------------------------------
    # Adaptive sessions
    # ------------------------------------------------------------------

    def create_session(
        self,
        *,
        events: list[dict[str, Any]],
        common_query: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> SessionResponse:
        data = self._request(
            "POST",
            "/v1/search-sessions",
            json={
                "searcher_type": "adaptive_temporal",
                "common_query": common_query,
                "events": events,
                "hyperparameters": hyperparameters or {},
                "constraints": constraints or {},
            },
        )
        return SessionResponse.model_validate(data)

    def create_session_from_queries(
        self,
        *,
        queries: list[str],
        common_query: str | None = None,
    ) -> SessionResponse:
        """LLM-rewrite based session creation: one input string per event, in order.

        The rewrite step enriches each line into a full event definition
        (anchor_query, pre_state, post_state, boundary_type, retrieval
        variants) - it does not split one line into multiple events.
        """
        data = self._request(
            "POST",
            "/v1/search-sessions/from-queries",
            json={"query": queries, "common_query": common_query},
        )
        return SessionResponse.model_validate(data)

    def get_session(self, session_id: str) -> SessionResponse:
        data = self._request("GET", f"/v1/search-sessions/{session_id}")
        return SessionResponse.model_validate(data)

    def delete_session(self, session_id: str, expected_revision: int) -> None:
        self._request(
            "DELETE",
            f"/v1/search-sessions/{session_id}",
            params={"expected_revision": expected_revision},
            expect_204=True,
        )

    def patch_event(
        self,
        session_id: str,
        event_id: str,
        *,
        expected_revision: int,
        patch: dict[str, Any],
    ) -> MutationResponse:
        data = self._request(
            "PATCH",
            f"/v1/search-sessions/{session_id}/events/{event_id}",
            json={"expected_revision": expected_revision, "patch": patch},
        )
        return MutationResponse.model_validate(data)

    def patch_hyperparameters(
        self,
        session_id: str,
        *,
        expected_revision: int,
        patch: dict[str, Any],
    ) -> MutationResponse:
        data = self._request(
            "PATCH",
            f"/v1/search-sessions/{session_id}/hyperparameters",
            json={"expected_revision": expected_revision, "patch": patch},
        )
        return MutationResponse.model_validate(data)

    def replace_constraints(
        self,
        session_id: str,
        *,
        expected_revision: int,
        constraints: dict[str, Any],
    ) -> MutationResponse:
        data = self._request(
            "PUT",
            f"/v1/search-sessions/{session_id}/constraints",
            json={"expected_revision": expected_revision, "constraints": constraints},
        )
        return MutationResponse.model_validate(data)

    def ingest_candidates(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_ids: list[str],
        candidates: list[dict[str, Any]],
    ) -> RunResponse:
        data = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/artifacts/candidates",
            json={
                "expected_revision": expected_revision,
                "event_ids": event_ids,
                "candidates": candidates,
            },
        )
        return RunResponse.model_validate(data)

    def ingest_frame_scores(
        self,
        session_id: str,
        *,
        expected_revision: int,
        region_ids: list[str],
        samples: list[dict[str, Any]],
    ) -> RunResponse:
        data = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/artifacts/frame-scores",
            json={
                "expected_revision": expected_revision,
                "region_ids": region_ids,
                "samples": samples,
            },
        )
        return RunResponse.model_validate(data)

    def retrieve_session(
        self,
        session_id: str,
        *,
        top_k: int = 50,
        event_ids: list[str] | None = None,
    ) -> RunResponse:
        data = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/commands/retrieve",
            json={"top_k": top_k, "event_ids": event_ids},
        )
        return RunResponse.model_validate(data)

    def mark_videos(
        self,
        session_id: str,
        *,
        expected_revision: int,
        video_ids: list[str],
    ) -> MutationResponse:
        data = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/commands/mark-videos",
            json={"expected_revision": expected_revision, "video_ids": video_ids},
        )
        return MutationResponse.model_validate(data)

    def fix_frame(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_id: str,
        video_id: str,
        frame_id: int,
        timestamp_seconds: float,
        region_id: str | None = None,
    ) -> MutationResponse:
        data = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/commands/fix-frame",
            json={
                "expected_revision": expected_revision,
                "event_id": event_id,
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp_seconds": timestamp_seconds,
                "region_id": region_id,
            },
        )
        return MutationResponse.model_validate(data)

    def reject_proposal(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_id: str,
        proposal_id: str,
    ) -> MutationResponse:
        data = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/commands/reject-proposal",
            json={
                "expected_revision": expected_revision,
                "event_id": event_id,
                "proposal_id": proposal_id,
            },
        )
        return MutationResponse.model_validate(data)

    def clear_event_constraint(
        self,
        session_id: str,
        *,
        expected_revision: int,
        event_id: str,
    ) -> MutationResponse:
        data = self._request(
            "POST",
            f"/v1/search-sessions/{session_id}/commands/clear-event-constraint",
            json={"expected_revision": expected_revision, "event_id": event_id},
        )
        return MutationResponse.model_validate(data)

    # ------------------------------------------------------------------
    # Read-only artifact paging
    # ------------------------------------------------------------------

    def get_frame_preview(
        self,
        video_id: str,
        *,
        anchor_seconds: float,
        radius_frames: int = 15,
        stride: int = 1,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/media/{video_id}/frame-preview",
            params={
                "anchor_seconds": anchor_seconds,
                "radius_frames": radius_frames,
                "stride": stride,
            },
        )

    def get_regions(
        self,
        session_id: str,
        *,
        event_id: str | None = None,
        video_id: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/search-sessions/{session_id}/regions",
            params={
                "event_id": event_id,
                "video_id": video_id,
                "offset": offset,
                "limit": limit,
            },
        )

    def get_proposals(
        self,
        session_id: str,
        *,
        event_id: str | None = None,
        video_id: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/search-sessions/{session_id}/proposals",
            params={
                "event_id": event_id,
                "video_id": video_id,
                "offset": offset,
                "limit": limit,
            },
        )

    def get_video_priorities(
        self,
        session_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        apply_boundary_refinement: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/search-sessions/{session_id}/video-priorities",
            params={
                "offset": offset,
                "limit": limit,
                "apply_boundary_refinement": apply_boundary_refinement,
            },
        )

    def get_tuples(
        self, session_id: str, *, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/search-sessions/{session_id}/tuples",
            params={"offset": offset, "limit": limit},
        )

    def get_frame_scores(
        self,
        session_id: str,
        *,
        event_id: str | None = None,
        video_id: str | None = None,
        region_id: str | None = None,
        offset: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/search-sessions/{session_id}/frame-scores",
            params={
                "event_id": event_id,
                "video_id": video_id,
                "region_id": region_id,
                "offset": offset,
                "limit": limit,
            },
        )

    def get_runs(
        self, session_id: str, *, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/search-sessions/{session_id}/runs",
            params={"offset": offset, "limit": limit},
        )

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        base: str | None = None,
        json: Any = None,
        params: dict[str, Any] | None = None,
        expect_204: bool = False,
    ) -> Any:
        url = f"{(base or self._backend).rstrip('/')}{path}"
        if params:
            params = {key: value for key, value in params.items() if value is not None}
        attempt = 0
        while True:
            try:
                response = self._http.request(
                    method, url, json=json, params=params
                )
                break
            except httpx.ConnectError as exc:
                attempt += 1
                if attempt > self.settings.max_retries:
                    raise ConnectionFailure(
                        f"cannot reach {url}: {exc}"
                    ) from exc
            except httpx.TimeoutException as exc:
                raise ConnectionFailure(
                    f"request to {url} timed out: {exc}"
                ) from exc
            except httpx.RequestError as exc:
                raise ConnectionFailure(f"request to {url} failed: {exc}") from exc

        if response.status_code == 204 and expect_204:
            return None
        if response.is_success:
            if response.content and response.headers.get("content-type", "").startswith(
                "application/json"
            ):
                return response.json()
            return {}
        return self._raise_for_status(response)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        request_id = response.headers.get("x-request-id")
        detail: Any = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail", payload)
        message = _detail_message(detail, response.text)

        if response.status_code == 409 and isinstance(detail, dict):
            if detail.get("code") == "revision_conflict":
                raise RevisionConflictError(
                    int(detail.get("expected_revision", -1)),
                    int(detail.get("current_revision", -1)),
                )
        if response.status_code == 404:
            raise SessionNotFound(
                response.status_code, message, detail=detail
            )
        if response.status_code == 422:
            raise ValidationFailure(message, detail=detail)
        raise HttpFailure(response.status_code, message, detail=detail)


def _detail_message(detail: Any, fallback: str) -> str:
    if isinstance(detail, dict):
        message = detail.get("message")
        if message:
            return str(message)
        return str(detail)
    if isinstance(detail, str) and detail:
        return detail
    return fallback[:500]


def close_client(client: TemporalApiClient) -> None:
    client._http.close()
