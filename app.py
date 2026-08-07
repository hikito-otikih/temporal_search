from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rewrite_queries import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
    rewrite_queries,
)
from rewrite_schema import RewriteRequest, RewriteResponse
from temporal_search import temporal_search

app = FastAPI(title="Temporal Search API")


class TemporalSearchRequest(BaseModel):
    query: list[str]
    top_k_tuple: int = 100
    top_k_each_query: int = 100
    gamma: float = 0.05
    searcher_type: str = "TemporalSearcher"
    objectFilterMode: bool = False
    object_name_list: Optional[list[str]] = None
    objectThreshold: float = 0.5


@app.post('/rewrite', response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    try:
        analysis = await rewrite_queries(
            common_query=request.common_query,
            queries=request.query,
        )
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=500, detail='OLLAMA_API_KEY is not configured'
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail='Ollama request timed out') from exc
    except OllamaRateLimitError as exc:
        raise HTTPException(
            status_code=503, detail='Ollama rate limit exceeded'
        ) from exc
    except OllamaServiceError as exc:
        raise HTTPException(status_code=502, detail='Ollama request failed') from exc

    return analysis


@app.get("/")
def read_root():
    return {"message": "Temporal Search API"}


@app.post("/temporal-search")
def search(request: TemporalSearchRequest):
    results = temporal_search(
        request.query,
        request.top_k_tuple,
        request.top_k_each_query,
        request.gamma,
        request.searcher_type,
        request.objectFilterMode,
        request.object_name_list,
        request.objectThreshold,
    )
    return {"query": request.query, "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
