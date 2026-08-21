from __future__ import annotations

import json
import os
import socket
from urllib import error, request

from pydantic import TypeAdapter, ValidationError

from adaptive_search.exceptions import UpstreamSearchError

from .schemas import QueryResponse

UPSTREAM_SEARCH_TIMEOUT_SECONDS = 30.0


def upstream_search_url() -> str:
    """Base URL of the frame-search API, overridable via env var.

    The default loopback works when both processes share a network namespace
    (e.g. all in WSL or all on Windows).  When the backend runs in WSL and the
    frame-search API runs on the Windows host, set UPSTREAM_SEARCH_URL to the
    WSL-visible gateway IP of the host.
    """
    return os.getenv("UPSTREAM_SEARCH_URL", "http://172.26.176.1:8000").rstrip("/")


def search_queries(queries: str | list[str], top_k: int) -> QueryResponse | list[QueryResponse]:
    """Call <UPSTREAM_SEARCH_URL>/search for one query or many queries.

    For a single string, returns one validated QueryResponse.
    For a list of strings, returns a list of validated QueryResponse objects in the same order.
    """

    if isinstance(queries, str):
        return _search_one(queries, top_k)

    return [_search_one(query, top_k) for query in queries]


def _search_one(query: str, top_k: int) -> QueryResponse:
    payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    req = request.Request(
        f"{upstream_search_url()}/text_retrieve",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Translated into UpstreamSearchError (mapped to a 502 by
    # adaptive_search.router._raise_api_error) instead of letting a raw
    # urllib/decode/validation exception propagate into _raise_api_error's
    # generic ValueError-or-ValidationError fallback, which maps to a 422
    # (client input error) - a slow/unreachable/misbehaving upstream is a
    # real, expected failure mode caused by the *upstream* service, not a
    # bug in the client's own request. UnicodeDecodeError (invalid UTF-8
    # bytes) is itself a ValueError subclass and pydantic's ValidationError
    # (valid JSON, wrong schema) is explicitly listed in that same fallback
    # branch - both would otherwise be misreported as if the caller's
    # request were malformed.
    try:
        with request.urlopen(req, timeout=UPSTREAM_SEARCH_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise UpstreamSearchError(
            f"upstream /text_retrieve failed with HTTP {exc.code}"
        ) from exc
    except (error.URLError, socket.timeout, TimeoutError) as exc:
        raise UpstreamSearchError("upstream /text_retrieve failed or timed out") from exc
    except ConnectionError as exc:
        # A reset/aborted/refused connection during response.read() is a
        # plain OSError subclass, not wrapped in URLError like most urllib
        # failures - without this it fell through to _raise_api_error's
        # generic fallback, which doesn't recognize it either, producing an
        # unstructured 500 instead of the 502 every other upstream failure
        # mode gets.
        raise UpstreamSearchError("upstream /text_retrieve connection was reset") from exc
    except UnicodeDecodeError as exc:
        raise UpstreamSearchError("upstream /text_retrieve returned invalid UTF-8") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise UpstreamSearchError("upstream /text_retrieve returned invalid JSON") from exc
    try:
        return TypeAdapter(QueryResponse).validate_python(decoded)
    except ValidationError as exc:
        raise UpstreamSearchError(
            "upstream /text_retrieve returned an invalid response schema"
        ) from exc

if __name__ == "__main__":
    # Example usage
    query = "cat"
    top_k = 5
    result = search_queries(query, top_k)
    print(result.results[0])
