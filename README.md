# Temporal Search API

Temporal Search ranks sequences of video keyframes that match a list of text queries. It fetches candidate frames from an upstream frame-search service, groups them by video, and scores tuples using both frame relevance and temporal distance.

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

## Requirements

- Python 3.10 or newer
- An upstream frame-search API running at `http://127.0.0.1:8000/search`

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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
python app.py
```

The Temporal Search API will be available at `http://127.0.0.1:8001`. Interactive API documentation is available at `http://127.0.0.1:8001/docs`.

Check that it is running:

```bash
curl http://127.0.0.1:8001/
```

## Rewrite queries with Ollama

`POST /rewrite` rewrites each event query into a self-contained sentence so it
still carries the video's overall semantic context when used on its own.

Create a `.env` file in the project root before starting the API:

```dotenv
OLLAMA_API_KEY=your-ollama-api-key
OLLAMA_API_URL=https://ollama.com/api/chat
```

`OLLAMA_API_KEY` is required. `OLLAMA_API_URL` is optional and defaults to
`https://ollama.com/api/chat`. Do not commit a real API key to source control.

Example request:

```json
{
  "modelname": "gpt-oss:20b",
  "common_query": "Đoạn video múa lân với một con lân màu vàng, đen và trắng.",
  "query": [
    "E1: Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất. Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.",
    "E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên."
  ]
}
```

Illustrative response (exact wording depends on the selected model):

```json
{
  "modelname": "gpt-oss:20b",
  "common_query": "Đoạn video múa lân với một con lân màu vàng, đen và trắng.",
  "query": [
    "E1: Trong đoạn video múa lân với một con lân màu vàng, đen và trắng, khoảnh khắc đầu tiên con lân bắt đầu xoay vòng trên cột số 4 bằng hai chân trước trước khi tiếp đất.",
    "E2: Trong đoạn video múa lân với một con lân màu vàng, đen và trắng, khoảnh khắc đầu tiên cả bốn chân của con lân chạm hoàn toàn xuống đất."
  ]
}
```

### Rewrite behavior

- `modelname` is passed to Ollama as the model to use.
- `common_query` is optional. A blank value is treated as omitted.
- The API normally makes one Ollama request for each `query` item. Every request
  receives `common_query` and the full input list as context, but instructs the
  model to rewrite only one target event. Calls may run concurrently, up to four
  at once.
- A draft that omits the identifying details from `common_query` is rejected and
  retried once with explicit feedback about the missing context. If both model
  drafts remain too shallow, the cleaned common context is prepended so the
  returned query is still self-contained.
- Each rewritten event must be understandable by itself. Results preserve the
  input order and the response echoes `modelname` and `common_query`.
- Without `common_query`, the other query items still provide context for each
  independent rewrite.

### Rewrite errors

- `422` indicates invalid input. `modelname` and all query strings must be
  non-empty; `query` must contain 1-32 items. Each query is limited to 4,000
  characters, all queries together to 24,000, and `common_query` to 8,000.
- `500` indicates that `OLLAMA_API_KEY` is not configured.
- `502` indicates an Ollama connection failure, error response, or malformed or
  empty response.
- `503` indicates that Ollama rate-limited the request.
- `504` indicates that the Ollama request timed out.

Errors use FastAPI's standard `detail` field. If any individual rewrite fails,
the entire `/rewrite` request fails; the endpoint does not return partial output.

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
  ]
}
```

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
        ├── 000.json
        └── 001.json
```

The CSV must contain a `frame_idx` column. The matching row's zero-based index selects the object JSON filename (`000.json`, `001.json`, and so on). Each JSON file must provide parallel `detection_class_names` and `detection_scores` arrays.

```json
{
  "detection_class_names": ["person", "chair"],
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

A tuple is retained when at least one of its frames contains all requested object classes at or above the threshold.

## Run from Python

The core function can also be called directly:

```python
from temporal_search import temporal_search

results = temporal_search(
    query=["a person enters", "the person sits"],
    top_k_tuple=10,
    top_k_each_query=100,
    gamma=0.05,
    searcher_type="TemporalSearcher",
)
```

## Project layout

```text
app.py                       FastAPI application and request model
temporal_search.py           Search orchestration and result formatting
sendRequests.py              Upstream frame-search client
video_clustering.py          Groups candidate frames by video
video_clustering_schema.py   Internal Pydantic models
response_schema.py           Upstream response models
searcher/
├── TemporalSearcher.py      Ordered temporal tuple search
└── AmbiguousSearcher.py     Order-independent tuple search
data/                        Sample results and optional object metadata
```
