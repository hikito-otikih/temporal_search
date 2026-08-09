# Kế hoạch hoàn thiện Interactive Temporal Video Search

## 1. Kết luận review

Ý tưởng gốc đúng ở ba quyết định: session là source of truth, top-k chỉ là
view có thể dựng lại, và dense refinement phải chạy theo budget. Tuy nhiên,
không thể triển khai nguyên trạng vì plan cũ trộn MVP nghiên cứu với kiến trúc
production và bỏ qua một số contract đang thiếu trong repository.

Các sửa đổi bắt buộc:

1. Không tạo package `app/` song song với `app.py`; adaptive code nằm trong
   `adaptive_search/` và được mount bằng FastAPI router.
2. Giữ nguyên `/rewrite` và `/temporal-search` làm legacy contract. API session
   mới có prefix `/v1`.
3. Dùng `pts_ms` hoặc `timestamp_seconds` làm trục thời gian adaptive;
   `frame_index` chỉ dùng cho legacy/UI.
4. Thêm `FrameProvider` trước khi tuyên bố dense refinement chạy live. Checkout
   hiện chỉ có metadata, không có raw video/frame pixels.
5. Một event có nhiều query variant. Phải fusion/deduplicate theo `event_id`
   trước khi assembly; không được coi mỗi variant là một event khác.
6. Không trộn raw upstream score, cosine embedding và reranker score. Lưu raw
   score riêng, chuẩn hóa theo query/region, rồi mới kết hợp.
7. Cache image embedding theo content hash + PTS + model revision + preprocess;
   sampling policy thuộc refinement artifact, không thuộc image key.
8. Adaptive tuple assembly phải bounded. Heap top-k không làm exhaustive
   backtracking bớt tốn thời gian.
9. Hard constraint phải được filter trước scoring. `constraint_bonus` không
   được dùng để biến một ràng buộc bắt buộc thành soft preference.
10. Model, revision, dimension, preprocessing, prompt/instruction và code
    version phải có trong fingerprint/run manifest để viết paper tái lập được.

## 2. Scope đã chốt

### Vertical slice triển khai trong repository này

- typed domain model và nested hyperparameters;
- region clustering và merge theo thời gian;
- robust calibration cho anchor/pre/post curve;
- multi-window boundary proposal, persistence và temporal NMS;
- video prioritization và bounded refinement frontier;
- ordered same-video tuple assembly với hard constraints và adjacent-gap
  penalty tính theo giây;
- immutable-style artifact fingerprints và executable invalidation rules;
- session API có optimistic revision;
- API ingest sparse candidates và precomputed frame scores để chạy toàn bộ
  phần thuật toán mà không cần GPU/model trong unit test;
- model/provider abstraction, optional SigLIP2 adapter và Qwen3-VL adapter;
- capability response phải nói rõ live refinement chưa khả dụng nếu chưa cấu
  hình frame source;
- regression/unit/API tests và tài liệu frontend/paper.

### Chưa tuyên bố hoàn tất trong vertical slice

- decode raw video thật và batch frame extraction;
- persistent GPU embedding service;
- LLM `/rewrite-v2` sinh pre/post state;
- async distributed job queue, SSE reconnect, multi-worker session store;
- persistent SQLite/Postgres/artifact cache;
- live benchmark/quality claim trên dataset được gán nhãn.

Các mục trên cần raw video hoặc batch-frame API, model weights và validation
dataset. Interface được đóng trước để bổ sung chúng không làm đổi scoring core.

## 3. Kiến trúc mục tiêu

```text
event text / rewritten event
        |
        v
query variants --retrieve--> sparse candidates
        |                         |
        |                    fusion + dedup
        |                         |
        +------------------> temporal regions
                                  |
                         coverage-aware frontier
                                  |
                    medium sample -> candidate boundary
                                  |
                         dense sample near boundary
                                  |
                    anchor/pre/post embedding curves
                                  |
                calibration -> proposals -> temporal NMS
                                  |
                  hard constraints -> bounded assembly
                                  |
                       diversified top-k tuple view
```

Hai baseline vẫn là pipeline độc lập:

```text
queries -> upstream /search -> cluster by video
        -> legacy ordered/ambiguous backtracking -> legacy top-k
```

Legacy score tiếp tục dùng `frame_index` và pairwise-distance formula để giữ
regression. Adaptive score dùng seconds và không được gọi là tương đương score
legacy.

## 4. Model profile

Máy phát triển có RTX 4060 Laptop 8 GB, vì vậy profile mặc định ưu tiên
throughput thay vì chọn model lớn nhất:

| Profile | Dense frame encoder | Final reranker | Mục đích |
|---|---|---|---|
| `balanced` | `google/siglip2-base-patch16-224` | tắt | mặc định trên GPU 8 GB |
| `quality` | `Qwen/Qwen3-VL-Embedding-2B`, 1024d MRL | Qwen3-VL-Reranker-2B cho top 10–20 | demo chất lượng/ablation |
| `paper_full` | Qwen3-VL-Embedding-2B, 2048d | Qwen3-VL-Reranker-2B | accuracy ceiling của model 2B |

Không auto-upgrade model theo thời gian. Deployment phải pin immutable model
revision. SigLIP2 và Qwen3-VL dùng Apache-2.0. Reranker nhận ordered frames để
đánh giá thứ tự; dual encoder độc lập chỉ tạo frame-level similarity curve.

## 5. Seed hyperparameters

Đây là seed mặc định chưa được tune trên validation set:

```yaml
retrieval:
  top_n_per_variant: 50
  top_n_fused: 100
  rrf_k: 60
  query_variants_per_event: 4
clustering:
  gap_seconds: 3.0
  margin_seconds: 3.0
  max_region_seconds: 30.0
refinement:
  max_initial_videos: 5
  max_regions_per_event_per_video: 3
  max_total_regions: 60
  exploration_region_ratio: 0.15
  medium_interval_seconds: 0.5
  dense_interval_seconds: 0.1
  dense_radius_seconds: 1.0
  max_frames_per_run: 2000
boundary:
  window_options_seconds: [0.5, 1.0, 2.0, 3.0]
  min_samples_per_side: 2
  semantic_weight: 0.40
  boundary_weight: 0.30
  pre_weight: 0.10
  post_weight: 0.20
  nms_radius_seconds: 0.50
  max_proposals_per_region: 5
ranking:
  top_k: 20
  max_proposals_per_event_per_video: 8
  max_combinations_per_video: 10000
  max_tuples_per_video: 200
  max_total_tuples: 2000
  default_gap_tau_seconds: 10.0
  gap_lambda: 0.01
```

Threshold không mặc định là `0.5` chỉ vì score nằm trong `[0,1]`. Khi có label,
fit calibration trên validation split và lưu calibration artifact/version.

## 6. Roadmap có acceptance gate

### Phase 0 — Freeze contract và feasibility

- golden regression cho legacy scoring/clustering/object filter;
- canonical timestamp contract;
- `FrameProvider`/model capability probe;
- strict request bounds và explicit searcher enum.

Acceptance: valid legacy fixture giữ nguyên output; adaptive capability không
báo available khi thiếu frame source.

### Phase 1 — Adaptive algorithm core

- typed artifacts;
- clustering/frontier;
- calibration/proposal/NMS;
- bounded tuple assembly;
- executable invalidation và fingerprint.

Acceptance: deterministic synthetic tests cover transition curve, insufficient
window, NMS, constraints và runtime cap.

### Phase 2 — Session vertical slice

- in-memory single-process store;
- optimistic revision (`409` khi stale);
- patch event/hyperparameters và precise invalidation;
- ingest candidate/frame-score artifacts;
- query region/proposal/tuple/run.

Acceptance: ranking-only patch không xóa frame/proposal; sửa một event chỉ xóa
artifact của event đó và toàn tuple view.

### Phase 3 — Live provider/inference

- batch `FrameProvider`, union overlapping intervals;
- local or remote SigLIP2 worker;
- optional Qwen quality/rerank stage;
- image/text embedding cache;
- medium/dense scheduler theo frame budget.

Acceptance: cùng `(video hash, pts, model revision, preprocess)` chỉ encode một
lần; vượt budget phải dừng có lý do, không silently truncate sai event.

### Phase 4 — Production interaction

- SQLite/Postgres session metadata;
- persistent artifact store;
- idempotency key;
- worker queue, cancellation/stale-commit guard;
- SSE có event ID và reconnect;
- pagination/asset authorization.

Acceptance: stale job không publish qua revision mới; restart process không mất
session đã commit.

### Phase 5 — Evaluation và paper

- annotation schema: video, event point/interval, tolerance;
- frozen train/validation/test split;
- baseline và adaptive dùng cùng rewrite/candidate budget;
- metrics: Video Recall@K, Proposal Recall ±1/±2 s, Tuple Recall@K, latency,
  frames encoded, cache hit, clicks/time;
- ablation: anchor-only, +pre/post, +multi-resolution, +reranker, +human feedback;
- run manifest lưu git commit, dataset/index version, seed, hardware và toàn bộ
  model/config fingerprints.

Acceptance: report có confidence interval và không tune threshold trên test set.
