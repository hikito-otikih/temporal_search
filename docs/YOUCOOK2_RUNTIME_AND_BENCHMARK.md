# YouCook2 runtime và Video Recall@K benchmark

Tài liệu này ghi lại phần đã triển khai và cách chạy trên dataset local. Toàn bộ
code nằm trong repository Temporal Search; không import, copy hoặc execute code
từ `D:\temporal-benchmark`. Dịch vụ bên ngoài duy nhất của benchmark là public
HTTP contract tại `http://127.0.0.1:8000`.

## 1. Trạng thái đã xác nhận

| Thành phần | Trạng thái |
|---|---|
| Recursive YouCook2 catalog | Đã nhận đủ 1.660 video local |
| Mapping `video_id -> asset` | Đã xác nhận trên layout split/recipe |
| Exact PTS decoder | Đã chạy thật bằng PyAV |
| RGB frame output | Đã chạy thật, NumPy `H x W x 3` |
| Budgeted interval sampling | Unit test và provider contract đã pass |
| Medium -> dense orchestration | Đã triển khai và test bằng fake embedder |
| Live refinement API command | Đã triển khai, có capability/error gate |
| SigLIP2 GPU inference | Chưa chạy thật; cần cài ML deps và pin model commit |
| YouCook2 Video Recall@K | Đã chạy live pilot qua API port 8000 |
| Qwen ordered-frame reranker | Chưa có runtime |

Real decode smoke trên `0IuQKThr-pM`:

```text
requested: 0, 1000, 5000, 167000 ms
actual:    0, 1001, 5005, 167000 ms
shape:     (360, 640, 3)
```

Actual PTS được giữ lại; hệ thống không giả định requested timestamp luôn trùng
đúng một decoded frame.

## 2. Cài runtime tùy chọn

Trong WSL, từ repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements-live-refinement.txt
```

Với GPU CUDA, nên cài PyTorch build phù hợp máy trước rồi mới cài các package
còn lại. File trên là bootstrap dependency, không thay cho frozen environment
lock của paper.

Biến môi trường mẫu nằm ở `.env.example`:

```bash
export YOUCOOK2_DATA_ROOT=/mnt/c/Users/huynh/Downloads/youcook2
export YOUCOOK2_METADATA_ROOT=/mnt/c/Users/huynh/Downloads/youcook2/metadata
export ADAPTIVE_SIGLIP2_MODEL=google/siglip2-base-patch16-224
export ADAPTIVE_SIGLIP2_REVISION=<immutable-huggingface-commit>
export ADAPTIVE_SIGLIP2_DIMENSION=768
export ADAPTIVE_DEVICE=cuda
export ADAPTIVE_TORCH_DTYPE=float16
```

Không dùng `main` làm revision. Session model spec và runtime model identity phải
trùng model ID, revision, dimension, preprocess và instruction.

## 3. Probe catalog và decoder

Chỉ scan catalog/metadata, không decode:

```bash
python -m scripts.probe_youcook2_provider \
  --data-root /mnt/c/Users/huynh/Downloads/youcook2 \
  --metadata-root /mnt/c/Users/huynh/Downloads/youcook2/metadata \
  --video-id 0IuQKThr-pM
```

Decode các mốc thật:

```bash
python -m scripts.probe_youcook2_provider \
  --data-root /mnt/c/Users/huynh/Downloads/youcook2 \
  --metadata-root /mnt/c/Users/huynh/Downloads/youcook2/metadata \
  --video-id 0IuQKThr-pM \
  --decode-pts-ms 0,1000,5000,167000
```

Catalog scan qua `/mnt/c` mất khoảng 20--30 giây cho 1.660 video. Đây là startup
cost hiện tại; persistent catalog cache vẫn là optimization sau này.

## 4. Live adaptive refinement

Khởi động API sau khi export environment:

```bash
pip install -e .
python src/main.py
```

Luồng (cập nhật sau khi `commands/refine` và `GET .../tuples` bị xoá - xem
[`ADAPTIVE_PIPELINE_MIGRATION.md`](ADAPTIVE_PIPELINE_MIGRATION.md)):

```text
POST /v1/search-sessions
  -> POST .../artifacts/candidates
  -> GET  .../regions
  -> GET  .../video-priorities
  -> POST .../artifacts/frame-scores
  -> GET  .../proposals
  -> GET  .../runs
```

`GET /v1/searchers` trả `runtime_model_spec` đầy đủ gồm `model_id`,
`revision`, `dimension`, `preprocess` và `instruction`. Khi balanced embedder đã
được cấu hình và request tạo session không gửi
`hyperparameters.refinement.embedding_model`, API tự lưu chính xác spec runtime
này vào session. Nếu client gửi spec rõ ràng thì server giữ nguyên; mismatch sẽ
làm `live_refinement.available=false` thay vì âm thầm đổi model.

Request refinement:

```json
{
  "expected_revision": 0,
  "region_ids": ["region-id"],
  "max_frames": 256
}
```

`region_ids=null` dùng refinement frontier hiện tại. `max_frames` chỉ được giảm
so với hard budget của session, không được vượt `max_frames_per_run`.

Coordinator:

1. chia global frame budget công bằng cho region;
2. sample medium trên toàn region;
3. tìm anchor peak;
4. sample dense quanh peak;
5. deduplicate actual PTS;
6. encode anchor/pre/post theo microbatch (mặc định 64 frame);
7. tạo và calibrate `FrameScoreSample`;
8. tạo proposal/tuple và persist run metrics;
9. đặt region đã xử lý thành `refinement_status=completed`.

Live scorer hiện ghi `motion_scores_available=false` và dùng
`raw_motion_score=0`; không được coi đây là motion ablation đã chạy. Cờ
`use_reranker=true` bị trả `422` cho tới khi ordered-frame reranker thật được
triển khai, tránh một ablation no-op nhưng bị ghi nhãn như đã bật.

Nếu thiếu provider hoặc embedder, endpoint trả `503
live_refinement_unavailable`. Model mismatch hoặc provider data lỗi trả `422`
với error code riêng. Capability tổng hợp được trả bởi `GET /v1/searchers` và
session response.

Tổng frame mặc định là 2.000/run và có hard ceiling 4.096; image microbatch mặc
định 64, tối đa 512. Lỗi first-load/network/dtype/OOM của embedder trả `503` và
không commit frame score.

## 5. Video Recall@K benchmark

Package: `research_tools/benchmarks/youcook2/`.

Mục tiêu hiện tại chỉ là:

> Với một event query, ground-truth video có nằm trong top-K unique videos hay
> không?

Ground-truth video/path/interval không bao giờ được gửi đến backend. Request chỉ
có:

```json
{"query": "event text", "top_k": 200}
```

### Windows command

API port 8000 hiện chạy trong Windows network namespace, do đó chạy bằng Windows
Python là cách trực tiếp nhất:

```powershell
cd \\wsl.localhost\Ubuntu\home\huynhchiton\projects\temporal_search\research_tools

python -m benchmarks.youcook2 health `
  --base-url http://127.0.0.1:8000

python -m benchmarks.youcook2 run `
  --query-dir 'C:\Users\huynh\Downloads\youcook2\query' `
  --query-mode event `
  --frame-top-k 200 `
  --recall-k 1,5,10,20,50 `
  --aggregation max `
  --output-dir 'C:\Users\huynh\Downloads\youcook2_benchmark_max'
```

Thêm `--limit 20` cho pilot. Nếu process dừng:

```powershell
python -m benchmarks.youcook2 run <same arguments> --resume
```

Resume chỉ hợp lệ khi output directory còn đủ `run_manifest.json` và
`query_results.jsonl`, đồng thời schema, source, config, model và index identity
không đổi. `--force-resume` bị vô hiệu hóa để tránh trộn hai thí nghiệm; khi đổi
ablation/backend hãy dùng output directory mới.

Mỗi ablation phải dùng output directory riêng. Xem README trong folder benchmark
cho `top_m_mean`, `logsumexp`, manifest/annotations input và index-coverage
audit.

### Artifacts

- `query_results.jsonl`: append-only checkpoint và unique-video ranking;
- `query_results.csv`: latest result/query;
- `metrics.json`: all-query Recall@K/MRR (request error tính là miss), các
  successful-request diagnostics, median rank, latency, unique-video counts;
- `run_manifest.json`: data fingerprint, API health/model, config, runtime và
  leakage declaration.

## 6. Pilot đã chạy

Pilot live 20 event, `max` aggregation, `frame_top_k=200`:

| Metric | Giá trị |
|---|---:|
| Completion | 20/20, không lỗi |
| Recall@1 | 0.10 |
| Recall@5 | 0.20 |
| Recall@10 | 0.25 |
| Recall@20 | 0.45 |
| Recall@50 | 0.55 |
| MRR | 0.1538 |
| Ground-truth found | 12/20 |
| Unique videos/result | min 39, median 76, mean 80.9, max 147 |
| Mean API latency | khoảng 2.56 giây/query |

Đây chỉ là engineering pilot trên 20 training events với upstream
`google/siglip-base-patch16-224`; không phải kết quả paper và không dùng để tune
threshold/weights. Artifact pilot này dùng manifest v1 và chỉ nên giữ để tham
khảo; runner v2 phải bắt đầu ở output directory mới.

## 7. Việc còn lại

- Pin immutable SigLIP2 commit, cài CUDA-compatible runtime và chạy real GPU
  inference/VRAM-throughput benchmark.
- Sinh held-out validation query hoặc dùng official validation annotations;
  không tune và report trên cùng training queries.
- Chạy full Video Recall@K và các aggregation/language/model ablations.
- Chỉ sau khi video retrieval ổn mới bật temporal metrics/dense benchmark.
- Implement Qwen ordered-frame reranker nếu cần quality profile.
- Persistent artifact/session store, queue, cancellation và SSE vẫn là production
  phase riêng.
