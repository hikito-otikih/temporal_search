from __future__ import annotations

import json
import os
from urllib import request

from pydantic import TypeAdapter

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
        f"{upstream_search_url()}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=UPSTREAM_SEARCH_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8")
        return TypeAdapter(QueryResponse).validate_python(json.loads(body))

if __name__ == "__main__":
    # Example usage
    query = "cat"
    top_k = 5
    result = search_queries(query, top_k)
    print(result.results[0])
