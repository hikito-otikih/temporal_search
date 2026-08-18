# YouCook2 corpus-level Video Recall@K

This package answers one deliberately narrow question:

> Given a text event, does the unique-video ranking returned through the
> frame-search API contain the ground-truth video in its top K?

This first-stage benchmark does **not** evaluate temporal intervals, dense frame
extraction, pre/post states, transition boundaries, or tIoU. Ground-truth
timestamps are parsed and preserved for later stages, but are not used to rank
videos.

The implementation is self-contained. It does not import, copy, or execute any
code from `D:\temporal-benchmark`. Its only runtime dependency is Python's
standard library, and its only backend dependency is the HTTP contract below.

## Backend contract

The default backend is `http://127.0.0.1:8000`.

```http
GET /health
POST /search
Content-Type: application/json

{"query": "cắt hành tây", "top_k": 200}
```

The search response must be a JSON object with a `results` list. The currently
deployed shape is supported directly:

```json
{
  "query": "cắt hành tây",
  "english_query": "cut the onion",
  "top_k": 200,
  "results": [
    {
      "video_name": "0IuQKThr-pM.mp4",
      "frame_index": 12,
      "timestamp": "0:41",
      "video_title": "...",
      "author": "...",
      "watch_url": "...",
      "score": 0.83
    }
  ]
}
```

Extra response fields are ignored. Every result must contain a video identifier
(`video_name` is preferred) and a finite numeric `score`.

`GET /health` must expose both model identity and index identity. At minimum the
current service reports `model`, `store`, and `n_vectors`; paper runs should also
expose immutable `model_revision`, preprocessing version, and `index_hash`.
These retrieval-relevant fields are fingerprinted, and resume is refused if
they change.

The request payload is intentionally constructed from only `query` and `top_k`.
`ground_truth_video`, `video_path`, answer intervals, and query ids are never
sent to the backend, so the benchmark cannot accidentally filter by its answer.

## Quick start in WSL

Run commands from the repository root:

```bash
cd /home/huynhchiton/projects/temporal_search/research_tools

python -m benchmarks.youcook2 health \
  --base-url http://127.0.0.1:8000
```

`127.0.0.1` is local to the OS environment running the CLI. If the backend was
started by Windows and WSL is using NAT networking, run the benchmark with
Windows Python (or pass the Windows host address to `--base-url`):

```powershell
$env:PYTHONPATH='\\wsl.localhost\Ubuntu\home\huynhchiton\projects\temporal_search\research_tools'
python -m benchmarks.youcook2 health --base-url http://127.0.0.1:8000

python -m benchmarks.youcook2 run `
  --query-dir 'C:\Users\huynh\Downloads\youcook2\query' `
  --query-mode event `
  --limit 5 `
  --output-dir 'C:\Users\huynh\Downloads\youcook2_benchmark_smoke'
```

The CLI forces UTF-8 output where the console supports reconfiguration, so
Vietnamese queries are safe on legacy Windows code pages.

Parse the real data and inspect the first three retrieval inputs without making
network calls or creating an output run:

```bash
python -m benchmarks.youcook2 run \
  --query-dir /mnt/c/Users/huynh/Downloads/youcook2/query \
  --query-mode event \
  --dry-run
```

Run a five-query end-to-end smoke test:

```bash
python -m benchmarks.youcook2 run \
  --query-dir /mnt/c/Users/huynh/Downloads/youcook2/query \
  --query-mode event \
  --base-url http://127.0.0.1:8000 \
  --frame-top-k 200 \
  --recall-k 1,5,10,20,50 \
  --aggregation max \
  --limit 5 \
  --output-dir benchmark_runs/youcook2/smoke_max
```

For a full run, remove `--limit`. A killed or interrupted run can be continued:

```bash
python -m benchmarks.youcook2 run \
  --query-dir /mnt/c/Users/huynh/Downloads/youcook2/query \
  --query-mode event \
  --output-dir benchmark_runs/youcook2/siglip_max \
  --resume
```

Successful query ids already present in the JSONL checkpoint are skipped.
Failed requests are attempted again. Resume requires both `run_manifest.json`
and `query_results.jsonl`, and refuses a changed schema, source, configuration,
model, or index identity. `--force-resume` is deliberately disabled because
reusing successful rows from an incompatible run would mix experiments; use a
new output directory instead. A single partial JSONL tail left by a killed
process is truncated safely before append.

Runs created with the earlier `youcook2-video-recall/v1` manifest cannot be
resumed under v2. Keep them as pilot artifacts or start a new v2 output
directory.

The example config provides the same settings:

```bash
python -m benchmarks.youcook2 \
  --config benchmarks/youcook2/example_config.json \
  run
```

Explicit CLI flags override JSON defaults. Use a separate output directory for
every ablation; do not overwrite a run with a different aggregation/model.

## Query inputs

Exactly one source is selected per run.

### Local generated TXT files

`--query-dir` understands the current UTF-8 YouCook2 format:

```text
Đoạn video ..., tìm các sự kiện sau:
E1: cắt hành tây
E2: trộn hành tây với trứng
**Answer
video_path: ".../0IuQKThr-pM.mp4"
E1: 0:41 - 0:53
E2: 1:26 - 1:33
```

Query modes:

- `event` (default): one retrieval request for each `E*` text.
- `event_with_context`: prefix each event with the descriptive header.
- `file`: one combined retrieval request per query file.

### Query manifest

`--query-manifest` accepts JSONL, JSON, or CSV. Each row needs:

- `query_id` (or `id`)
- `query_text` (or `query`/`text`)
- `ground_truth_video` (or `video_id`/`video_name`/`video_path`)

Optional fields are `event_id`, `context`, `start_seconds`, and `end_seconds`.

### Official YouCook2 annotations

Use captions from the official annotation JSON to build an English validation
benchmark without using a generator script:

```bash
python -m benchmarks.youcook2 run \
  --annotations-json /mnt/c/Users/huynh/Downloads/youcook2/annotations/youcookii_annotations_trainval.json \
  --annotation-subset validation \
  --dry-run
```

The expected root is `{"database": {video_id: ...}}`; each annotation's
`sentence` becomes an event query and `segment` is retained as metadata.

### Optional index-coverage audit

`--video-manifest` accepts TXT, CSV, JSONL, or JSON video ids/paths. It verifies
that every ground-truth video is expected in the index before evaluation.
Choose `--missing-ground-truth error` (default), `skip`, or `keep`. If `skip` is
used, the skipped count is recorded in the run manifest; it must be reported
alongside Recall@K.

## Frame hits to unique videos

The API returns frames, whereas the metric ranks videos. Paths and known video
extensions are removed without lowercasing YouTube ids, then frames are grouped
by exact video id. Available video scores are:

- `max`: maximum frame score; recommended first baseline.
- `top_m_mean`: mean of the best `--top-m` frame scores.
- `logsumexp`: stable LogSumExp with `--temperature`; rewards repeated evidence.

Ties are resolved by exact video id for deterministic output. Suggested
ablation commands differ only in output and aggregation:

```bash
python -m benchmarks.youcook2 --config benchmarks/youcook2/example_config.json \
  run --aggregation top_m_mean --top-m 3 \
  --output-dir benchmark_runs/youcook2/siglip_top3_mean

python -m benchmarks.youcook2 --config benchmarks/youcook2/example_config.json \
  run --aggregation logsumexp --temperature 0.1 \
  --output-dir benchmark_runs/youcook2/siglip_lse_t01
```

## Metrics and truncation

The package reports `Recall@K`, MRR, median rank, successful-request coverage,
and mean latency. A missing ground-truth video has reciprocal rank zero.

`POST /search` is truncated by **frame** top-K, not video top-K. Repeated frames
from one video may therefore yield fewer unique videos than requested. Recall on
this returned unique list is exact, but the full-corpus rank of a miss is
unknown. `median_rank` is consequently an optimistic truncated-list imputation:
each miss is assigned `unique_video_count + 1`. It is not a survival-analysis
censored median. `median_rank_found` is also emitted separately.
Always report `frame_top_k` and the distribution of `unique_video_count` with
Recall@K. Raising `frame_top_k`, if the backend permits it, tests truncation
sensitivity.

Primary `recall_at_K` and `mrr` use **all requested queries**; a final API error
counts as a miss. Conditional diagnostics
`recall_at_K_successful_requests` and `mrr_successful_requests` are also emitted
to separate retrieval quality from infrastructure failures. `error_count` and
`completion_rate` must be reported, and a run with any final query error exits
with status code 2.

## Artifacts

Each output directory contains:

- `query_results.jsonl`: append-only per-query checkpoint with the unique-video
  ranking, backend-translated query, rank, latency, and any error.
- `query_results.csv`: one latest row per query for analysis or spreadsheets.
- `metrics.json`: aggregate metrics and their configuration.
- `run_manifest.json`: source SHA-256, API health snapshot, stable backend
  fingerprint, runtime, complete configuration, artifact names, and leakage
  declaration.

The JSONL may contain multiple attempts for a query after resume; the last
attempt is authoritative. The CSV and metrics always use that latest attempt.

## Tests

The tests use a local fake HTTP server and do not require the real API or data:

```bash
python -m unittest discover -s benchmarks/youcook2/tests -v
```

They cover TXT/annotation parsing, case-safe video normalization, all three
aggregation rules, Recall/MRR/median rank, the deployed response schema,
ground-truth request sanitization, artifacts, fail-closed resume compatibility,
backend drift, and partial-checkpoint recovery.
