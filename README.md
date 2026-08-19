# Temporal Search API

Temporal Search ranks sequences of video keyframes that match a list of text
queries. The repository keeps the original frame-index search API and adds a
versioned adaptive session API with query fusion, boundary proposals, user
constraints, and ordered tuple ranking.

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
ranking via `GET .../video-priorities` (optionally requesting native-FPS
boundary refinement). Precomputed frame-level anchor/pre/post scores can be
ingested via `POST .../artifacts/frame-scores` to drive the separate
dense-refinement/proposal pipeline. `commands/refine` and the `GET .../tuples`
endpoint have been removed - see
[docs/ADAPTIVE_PIPELINE_MIGRATION.md](docs/ADAPTIVE_PIPELINE_MIGRATION.md) for
what replaced them. The system supports deterministic RRF fusion, temporal
regions, budgeted refinement, boundary profiles, temporal NMS, hard
constraints, optimistic revisions, and bounded same-video tuple assembly.

The local `YouCook2FrameProvider` uses PyAV and actual presentation timestamps;
catalog discovery over 1,660 videos and a real frame decode have been verified.
The medium/dense orchestrator, strict API command, persisted run metrics, and
region completion state are covered by end-to-end tests with an injected fake
embedder. `GET /v1/searchers` reports live refinement available only when both
the provider and model are usable, while retaining `frame_provider` as a nested
sub-capability.

Live image encoding is microbatched (default 64) under a hard 4,096-frame
server-side run ceiling. First-load/model-runtime/OOM failures return a
capability `503` without committing partial frame-score artifacts.

The balanced model profile is SigLIP2 Base. Its runtime is lazy: API startup
does not load/download weights, and live inference requires an immutable
`ADAPTIVE_SIGLIP2_REVISION`. Actual SigLIP2 GPU inference has not yet been
verified in this workspace. The optional Qwen3-VL embedding/reranker profile is
specified but still has no executed runtime.

Optional local refinement setup is documented in `.env.example` and
`requirements/live-refinement.txt`. At minimum configure `YOUCOOK2_DATA_ROOT`
and a pinned SigLIP2 revision before starting the API.

`GET /v1/searchers` exposes the complete active `runtime_model_spec`. When a
balanced session omits `refinement.embedding_model`, the API persists that
exact configured spec automatically; an explicit client spec is never silently
rewritten. The ordered-frame reranker and motion scorer are not implemented,
so `use_reranker=true` is rejected and live runs report
`motion_scores_available=false`.

The self-contained [YouCook2 benchmark](research_tools/benchmarks/youcook2/README.md) queries
`http://127.0.0.1:8000/search` and measures corpus Video Recall@K without sending
ground truth to the backend. Current saved runs are smoke/pilot runs only; they
are not full validation or temporal-boundary results.

See [the implementation plan](docs/IMPLEMENTATION_PLAN.md) and the
[complete adaptive module specification](docs/ADAPTIVE_TEMPORAL_SEARCH.md) for
pipeline formulas, defaults, API integration, invalidation rules, limitations,
and paper evaluation guidance.

## Requirements

- Python 3.11 or newer (matches `pyproject.toml`'s `requires-python`)
- An upstream frame-search API running at `http://127.0.0.1:8000/search`

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This installs the runtime dependencies pinned in `requirements/base.txt`
(referenced automatically via `pyproject.toml`'s dynamic dependencies).
For optional local boundary-refinement support, additionally install
`requirements/live-refinement.txt`:

```bash
python -m pip install -r requirements/live-refinement.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
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
pip install -e .
python src/main.py
```

The Temporal Search API will be available at `http://127.0.0.1:8001`. Interactive API documentation is available at `http://127.0.0.1:8001/docs`.

Check that it is running:

```bash
curl http://127.0.0.1:8001/
```

## Rewrite queries with Ollama

`POST /rewrite` analyzes all event queries as one batch and returns structured
video context, standalone target moments, Vietnamese/English retrieval queries,
visible states, temporal boundaries, relations, inferred facts, and ambiguities.

Create a `.env` file in the project root before starting the API:

```dotenv
OLLAMA_API_KEY=your-ollama-api-key
OLLAMA_API_URL=https://ollama.com/api/chat
OLLAMA_MODEL=gpt-oss:20b
```

`OLLAMA_API_KEY` is required. `OLLAMA_API_URL` and `OLLAMA_MODEL` are
optional; they default to `https://ollama.com/api/chat` and `gpt-oss:20b`.
The model is server configuration and cannot be selected in the request. Do not
commit a real API key to source control.

Example request:

```json
{
  "common_query": "Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau",
  "query": [
    "Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất. Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.",
    "Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên."
  ]
}
```

Illustrative response (exact wording depends on the configured model):

```json
{
  "video_context": {
    "scene": "múa lân",
    "main_entities": [
      "một con lân màu vàng, đen và trắng"
    ]
  },
  "events": [
    {
      "event_id": 0,
      "original_query": "Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất. Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.",
      "target_moment_vi": "Trong màn múa lân, khoảnh khắc đầu tiên con lân màu vàng, đen và trắng bắt đầu xoay vòng trên cột số 4 bằng hai chân trước.",
      "retrieval_queries_vi": [
        "Khoảnh khắc bắt đầu xoay vòng của con lân màu vàng, đen và trắng trên cột số 4 bằng hai chân trước.",
        "Con lân vàng, đen và trắng vừa bắt đầu xoay trên cột số 4 trong màn múa lân."
      ],
      "retrieval_queries_en": [
        "The first moment the yellow, black, and white lion starts spinning on pillar number 4 using its two front legs.",
        "A yellow, black, and white lion beginning its spin on pillar number 4 during a lion dance."
      ],
      "retrieval_queries_en_language": ["en", "en"],
      "subject": "con lân màu vàng, đen và trắng",
      "action": "bắt đầu xoay vòng trên cột số 4 bằng hai chân trước",
      "visible_state": "hai chân trước của lân ở trên cột số 4 và cơ thể vừa bắt đầu chuyển động xoay",
      "anchor_query": "Con lân màu vàng, đen và trắng bắt đầu xoay vòng trên cột số 4 bằng hai chân trước trong màn múa lân.",
      "pre_state": "Ngay trước đó, lân đứng yên trên cột số 4, chưa bắt đầu xoay vòng.",
      "post_state": "Ngay sau đó, lân đang xoay vòng trên cột số 4 bằng hai chân trước.",
      "boundary": "start",
      "temporal_relation": {
        "relation": "sequence_start",
        "reference_event_id": null
      },
      "required_entities": [
        "con lân màu vàng, đen và trắng",
        "cột số 4"
      ],
      "soft_context": [
        "màn múa lân"
      ],
      "excluded_context": [
        "khoảnh khắc lân đã tiếp đất"
      ],
      "inferred_information": [
        "Màu sắc của con lân được lấy từ common_query."
      ],
      "ambiguities": []
    },
    {
      "event_id": 1,
      "original_query": "Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.",
      "target_moment_vi": "Trong màn múa lân, khoảnh khắc đầu tiên con lân màu vàng, đen và trắng có cả bốn chân hoàn toàn chạm đất sau khi xoay vòng trên cột số 4.",
      "retrieval_queries_vi": [
        "Khoảnh khắc con lân màu vàng, đen và trắng có bốn chân hoàn toàn chạm đất lần đầu.",
        "Con lân vàng, đen và trắng vừa tiếp đất hoàn toàn bằng cả bốn chân trong màn múa lân."
      ],
      "retrieval_queries_en": [
        "The first moment the yellow, black, and white lion lands with all four feet fully on the ground.",
        "The yellow, black, and white lion fully touching down with all four feet during the lion dance."
      ],
      "retrieval_queries_en_language": ["en", "en"],
      "subject": "con lân màu vàng, đen và trắng",
      "action": "tiếp đất hoàn toàn bằng cả bốn chân",
      "visible_state": "cả bốn chân của lân đang chạm đất",
      "anchor_query": "Con lân màu vàng, đen và trắng có cả bốn chân hoàn toàn chạm đất trong màn múa lân.",
      "pre_state": "Ngay trước đó, lân vẫn đang ở trên không hoặc chỉ một phần chân chạm đất.",
      "post_state": "Ngay sau đó, cả bốn chân của lân đã chạm đất hoàn toàn.",
      "boundary": "end",
      "temporal_relation": {
        "relation": "after",
        "reference_event_id": 0
      },
      "required_entities": [
        "con lân màu vàng, đen và trắng"
      ],
      "soft_context": [
        "màn múa lân"
      ],
      "excluded_context": [
        "khoảnh khắc lân bắt đầu xoay vòng"
      ],
      "inferred_information": [
        "Màu sắc của con lân được lấy từ common_query."
      ],
      "ambiguities": []
    }
  ]
}
```

### Rewrite behavior

- `common_query` is optional. A blank value is treated as omitted.
- The API makes one Ollama request for the complete event batch, allowing the
  model to resolve shared entities and explicit cross-event relations.
- Every `target_moment_vi` and retrieval query is required to be understandable
  on its own. The response preserves the event count, order, IDs, and exact
  `original_query` strings.
- The server validates the complete response against strict Pydantic schemas,
  checks source alignment and distinct retrieval queries, then retries the full
  batch once with validation feedback if needed.
- If both model drafts are structurally valid but a lexical context check is
  still too strict, the server keeps the valid candidate and deterministically
  adds the cleaned `common_query` context instead of returning a false `502`.
- Temporary connection failures and upstream `408`, `425`, or `5xx`
  responses are retried once.
- Each event contains exactly two Vietnamese and two English retrieval queries.
- Ollama Cloud does not currently support the structured-output `format` field,
  so the JSON Schema is included in the prompt and enforced again by the server.

### Rewrite errors

- `422` indicates invalid input. `query` must contain 1-32 non-empty strings.
  Each query is limited to 4,000 characters, all queries together to 24,000,
  and `common_query` to 8,000. Unknown fields, including `modelname` and
  `model_name`, are rejected.
- `500` indicates that `OLLAMA_API_KEY` is not configured.
- `502` indicates a non-recoverable Ollama connection/error response or that
  no structurally valid output was available after the validation retry.
- `503` indicates that Ollama rate-limited the request.
- `504` indicates that the Ollama request timed out.

Errors use FastAPI's standard `detail` field. The endpoint never returns a
partially validated batch. Safe server logs distinguish upstream failures,
structural validation failures, semantic retries, and context fallback without
logging the API key or raw model output.

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
  "boundary_refinement_capability": {
    "requested": false,
    "available": false,
    "reason": null
  },
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
- Live dense refinement is capability-gated (`GET /v1/searchers`); when
  `live_refinement_available=false` you can still ingest precomputed frame
  scores.
- Run the UI tests with:

```bash
.venv/bin/python -m unittest discover -s research_tools/streamlit_ui/tests -p "test_*.py"
```

## Project layout

```text
src/
├── main.py                  FastAPI application assembly (mounts all 3 routers)
├── config.py                Process-wide .env bootstrap
├── adaptive_search/         Adaptive domain: router, schemas, service, algorithms
├── rewrite/                 LLM query-rewrite: router, schemas, service, prompt constants
└── legacy_search/           Pre-adaptive pipeline: router, schemas, service, searchers/
data/                        Sample results and optional object metadata
docs/                        Revised plan and adaptive technical specification
research_tools/           Streamlit debug console and benchmark tooling (dev-only)
```
