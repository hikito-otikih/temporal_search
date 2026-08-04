from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from rewrite_queries import (
    ConfigurationError,
    OllamaRateLimitError,
    OllamaServiceError,
    OllamaTimeoutError,
    rewrite_queries,
)
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


NonEmptyQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]


class RewriteRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    modelname: str = Field(min_length=1, max_length=200)
    common_query: Optional[str] = Field(default=None, max_length=8000)
    query: list[NonEmptyQuery] = Field(min_length=1, max_length=32)

    @field_validator('common_query', mode='before')
    @classmethod
    def normalize_empty_common_query(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode='after')
    def limit_total_query_length(self):
        if sum(len(item) for item in self.query) > 24000:
            raise ValueError('query content must not exceed 24000 characters')
        return self


class RewriteResponse(BaseModel):
    modelname: str
    common_query: Optional[str]
    query: list[str]


@app.post('/rewrite', response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    try:
        rewritten_queries = await rewrite_queries(
            modelname=request.modelname,
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

    return RewriteResponse(
        modelname=request.modelname,
        common_query=request.common_query,
        query=rewritten_queries,
    )


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
