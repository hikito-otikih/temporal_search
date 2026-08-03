from __future__ import annotations

import json
from urllib import request

from pydantic import TypeAdapter

from response_schema import QueryResponse


def search_queries(queries: str | list[str], top_k: int) -> QueryResponse | list[QueryResponse]:
    """Call http://127.0.0.1:8000/search for one query or many queries.

    For a single string, returns one validated QueryResponse.
    For a list of strings, returns a list of validated QueryResponse objects in the same order.
    """

    if isinstance(queries, str):
        return _search_one(queries, top_k)

    return [_search_one(query, top_k) for query in queries]


def _search_one(query: str, top_k: int) -> QueryResponse:
    payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    req = request.Request(
        "http://127.0.0.1:8000/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req) as response:
        body = response.read().decode("utf-8")
        return TypeAdapter(QueryResponse).validate_python(json.loads(body))

if __name__ == "__main__":
    # Example usage
    query = "cat"
    top_k = 5
    result = search_queries(query, top_k)
    print(result.results[0])
