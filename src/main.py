import config  # noqa: F401  (loads .env before routers import their own config)

from fastapi import FastAPI

from adaptive_search.router import router as adaptive_router
from legacy_search.router import router as legacy_router

app = FastAPI(title="Temporal Search API")

app.include_router(adaptive_router)
app.include_router(legacy_router)


@app.get("/")
def read_root():
    return {"message": "Temporal Search API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
