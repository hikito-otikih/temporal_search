from typing import Annotated, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from adaptive_search.api import router as adaptive_router

from rewrite_queries import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
    rewrite_queries,
)
from rewrite_schema import RewriteRequest, RewriteResponse
from temporal_search import temporal_search

NonEmptyQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]

app = FastAPI(title="Temporal Search API")


class TemporalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: list[NonEmptyQuery] = Field(min_length=1, max_length=32)
    top_k_tuple: int = Field(default=100, ge=1, le=10_000)
    top_k_each_query: int = Field(default=100, ge=1, le=10_000)
    gamma: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)
    searcher_type: Literal["TemporalSearcher", "AmbiguousSearcher"] = (
        "TemporalSearcher"
    )
    objectFilterMode: bool = False
    object_name_list: Optional[list[str]] = None
    objectThreshold: float = Field(default=0.5, ge=0.0, le=1.0)


app.include_router(adaptive_router)


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
