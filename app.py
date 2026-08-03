from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
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
