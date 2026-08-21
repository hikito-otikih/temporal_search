# Temporal Search API

Temporal Search ranks sequences of video keyframes that match a list of text
queries. The repository keeps the original frame-index search API and adds a
versioned adaptive session API with user constraints and ordered tuple
ranking.

The repository exposes the search through a FastAPI server on port `8001`.

## How it works

1. Each text query is sent to `POST http://127.0.0.1:8000/search`.
2. Candidate frames are grouped by video and ordered by frame index.
3. A search strategy builds one frame per query.
4. Tuples are ranked by average relevance, penalized by temporal distance:

   ```text
   final score = average frame score / (1 + temporal distance * gamma)
   ```

Available strategies:

- `TemporalSearcher` (default) matches queries in their supplied order.
- `AmbiguousSearcher` matches every query once without requiring query order.

## Adaptive temporal search

Call `GET /v1/searchers` to discover the legacy and adaptive contracts. The
adaptive workflow uses `POST /v1/search-sessions`, ingests sparse candidates
via `POST .../artifacts/candidates`, and ranks videos with order-aware tuple
ranking via `GET .../video-priorities`. The system supports temporal regions,
hard constraints, optimistic revisions, and bounded same-video tuple assembly.
There is no live GPU boundary-refinement stage anywhere in this repository
(removed from both the adaptive and legacy verticals) - ranking runs entirely
on the upstream search service's own candidate scores and timestamps.

The self-contained [YouCook2 benchmark](research_tools/benchmarks/youcook2/README.md) queries
`http://127.0.0.1:8000/search` and measures corpus Video Recall@K without sending
ground truth to the backend. Current saved runs are smoke/pilot runs only; they
are not full validation or temporal-boundary results.

See [the implementation plan](docs/IMPLEMENTATION_PLAN.md) and the
[complete adaptive module specification](docs/ADAPTIVE_TEMPORAL_SEARCH.md) for
pipeline formulas, defaults, API integration, invalidation rules, limitations,
and paper evaluation guidance.

## Requirements

- Python 3.12 or newer (matches `pyproject.toml`'s `requires-python`)
- [`uv`](https://docs.astral.sh/uv/) for environment/dependency management
- An upstream frame-search API running at `http://127.0.0.1:8000/search`

Set up the environment with `uv`:

```bash
uv sync
```

This creates `.venv` and installs the exact dependency versions pinned in
`uv.lock` (regenerated from `pyproject.toml`'s `[project.dependencies]` via
`uv lock` whenever a dependency changes). Run project commands through `uv`
without activating the venv manually:

```bash
uv run python src/main.py
uv run python -m unittest discover -s tests -t .
```

To activate the venv directly instead:

```bash
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
```

## Upstream search API

For every query, this project sends:

```json
{
  "query": "a person opens a door",
  "top_k": 100
}
```

The upstream `POST /search` endpoint must return this shape:

```json
{
  "query": "a person opens a door",
  "results": [
    {
      "video_name": "L21_V001.mp4",
      "video_title": "Example video",
      "author": "Example author",
      "watch_url": "https://example.com/video",
      "frame_index": 120,
      "timestamp": "00:00:04.000",
      "score": 0.91
    }
  ]
}
```

## Run the API

Start the upstream search service on port `8000`, then run:

```bash
uv run python src/main.py
```

The Temporal Search API will be available at `http://127.0.0.1:8001`. Interactive API documentation is available at `http://127.0.0.1:8001/docs`.

Check that it is running:

```bash
curl http://127.0.0.1:8001/
```

## Search example

The order of `query` items describes the expected chronological sequence when using `TemporalSearcher`.

```bash
curl -X POST http://127.0.0.1:8001/temporal-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": ["a person enters a room", "the person sits down"],
    "top_k_tuple": 10,
    "top_k_each_query": 100,
    "gamma": 0.05,
    "searcher_type": "TemporalSearcher"
  }'
```

Example response:

```json
{
  "query": ["a person enters a room", "the person sits down"],
  "results": [
    {
      "score": 0.82,
      "video_name": "L21_V001.mp4",
      "tuple": [
        {
          "frame_index": 120,
          "timestamp": "00:00:04.000",
          "score": 0.91,
          "query_id": 0,
          "satisfiedObjects": null
        },
        {
          "frame_index": 180,
          "timestamp": "00:00:06.000",
          "score": 0.87,
          "query_id": 1,
          "satisfiedObjects": null
        }
      ]
    }
  ],
  "search_truncated": false
}
```

`search_truncated` is `true` if a video's backtracking search (`TemporalSearcher`/`AmbiguousSearcher`) hit its internal node budget before exploring every combination exhaustively. `results` are still the best combinations found so far, but are not guaranteed to be the true best possible for that video - treat a `true` value as a signal to narrow the query or lower `top_k_each_query` rather than trusting the ranking as final.

### Request fields

| Field | Default | Description |
| --- | ---: | --- |
| `query` | required | Ordered list of text queries. |
| `top_k_tuple` | `100` | Maximum number of result tuples returned across all videos. |
| `top_k_each_query` | `100` | Candidate frames requested from the upstream service for each query. |
| `gamma` | `0.05` | Temporal-distance penalty; larger values favor more compact tuples. |
| `searcher_type` | `TemporalSearcher` | Use `TemporalSearcher` or `AmbiguousSearcher`. |
| `objectFilterMode` | `false` | Enables filtering using local object-detection metadata. |
| `object_name_list` | `null` | Object class names required in at least one frame of a tuple. |
| `objectThreshold` | `0.5` | Minimum object-detection confidence. |

## Object filtering

Object filtering requires keyframe maps and object-detection JSON files under `data/`:

```text
data/
├── map-keyframes/
│   └── L21_V001.csv
└── objects/
    └── L21_V001/
        ├── 001.json
        └── 002.json
```

The CSV must contain `frame_idx` and `n` columns, with `n` starting at 1 (e.g. `frame_idx=0` pairs with `n=1`). The matching row's own `n` value (not its position in the file) selects the object JSON filename, zero-padded to three digits (`001.json`, `002.json`, and so on). Each JSON file must provide parallel `detection_class_entities` and `detection_scores` arrays - `detection_class_entities` holds the human-readable label (e.g. `"Person"`), not `detection_class_names` (the raw Open Images machine-ID string, e.g. `"/m/01g317"`), which this endpoint never matches against.

```json
{
  "detection_class_entities": ["Person", "Chair"],
  "detection_scores": [0.98, 0.84]
}
```

Enable the filter in a request:

```json
{
  "query": ["a person enters", "the person sits"],
  "objectFilterMode": true,
  "object_name_list": ["person", "chair"],
  "objectThreshold": 0.5
}
```

A tuple is retained when at least one of its frames contains all requested object classes at or above the threshold. Matching against `detection_class_entities` is case-insensitive, so `object_name_list` values don't need to match the JSON's original capitalization.

## Run from Python

The core function can also be called directly:

```python
from legacy_search.service import temporal_search

results = temporal_search(
    query=["a person enters", "the person sits"],
    top_k_tuple=10,
    top_k_each_query=100,
    gamma=0.05,
    searcher_type="TemporalSearcher",
)
```

## Streamlit UI

A research/debug console lives in `research_tools/streamlit_ui/`. It talks only to the FastAPI
backend through a typed client (`research_tools/streamlit_ui/services/api_client.py`) and
never reimplements clustering/scoring.

```bash
# Install UI dependencies (streamlit, plotly) into the same venv
.venv/bin/python -m pip install -r research_tools/streamlit_ui/requirements.txt

# Optional configuration
cp research_tools/streamlit_ui/.env.example research_tools/streamlit_ui/.env

# Run the console (backend on :8001 must be running)
.venv/bin/python -m streamlit run research_tools/streamlit_ui/Home.py
```

Pages:

| Page | Purpose |
|---|---|
| `Home.py` | Landing + capability discovery. |
| `pages/00_Search.py` | The primary, consumer-facing search experience: rewrite preview, retrieval tuning, ranked video results, per-event keyframe browsing, and manual frame fixing. |
| `pages/01_Legacy_Search.py` | One-shot `/temporal-search`, no session. |
| `pages/02_Adaptive_Session.py` | Session stepper: events → candidates → regions → frame scores → proposals → tuples → feedback. |
| `pages/03_Region_Inspector.py` | Timeline, keyframe filmstrip, score curves. |
| `pages/05_YouCook2_Evaluation.py` | Corpus Video Recall@K with leakage guard and video dedup. |
| `pages/06_Run_Comparison.py` | Compare runs/hyperparameter presets. |

Design notes:

- The backend is the source of truth. Adaptive mutations always send
  `expected_revision`; on a revision conflict the UI reloads the session and
  asks you to re-apply instead of auto-retrying.
- Raw and normalized scores are shown separately and never labeled as
  probability/confidence.
- Ground truth (`video_path`) is parsed only inside the YouCook2 evaluator;
  retrieval payloads contain only event text and top-K.
- Run the UI tests with:

```bash
.venv/bin/python -m unittest discover -s research_tools/streamlit_ui/tests -p "test_*.py"
```

## Project layout

```text
src/
├── main.py                  FastAPI application assembly (mounts both routers)
├── config.py                Process-wide .env bootstrap
├── adaptive_search/         Adaptive domain: router, schemas, service, algorithms
└── legacy_search/           Pre-adaptive pipeline: router, schemas, service, searchers/
data/                        Sample results and optional object metadata
docs/                        Revised plan and adaptive technical specification
research_tools/           Streamlit debug console and benchmark tooling (dev-only)
```
