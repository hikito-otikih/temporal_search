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
python -m pip install fastapi uvicorn pydantic pandas
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
