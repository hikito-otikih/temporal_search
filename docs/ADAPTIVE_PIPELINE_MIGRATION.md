# Migration: adaptive_coarse + boundary_refinement lên production service

Ngày: 2026-08-13
Backend: `src/main.py` (port 8001), upstream sparse-search service thật (port 8000,
`172.26.176.1:8000` từ WSL), YouCook2 video thật (`YOUCOOK2_DATA_ROOT`), SigLIP2
dense scoring thật (`google/siglip2-base-patch16-224`, pinned revision, GPU).

Tài liệu này ghi lại việc đưa kết quả benchmark của
[`hyperparameter_sweep_results.md`](../research_tools/benchmarks/youcook2/hyperparameter_sweep_results.md)
và [`boundary_refinement_results.md`](../research_tools/benchmarks/youcook2/boundary_refinement_results.md)
vào service thật: cập nhật hyperparameter mặc định, loại bỏ code path chỉ phục
vụ `adaptive_full`, đưa `boundary_refinement` thành một stage thật (opt-in) cho
cả `adaptive_search` và `legacy_search`, và xác minh lại toàn bộ pipeline bằng
HTTP request thật vào service đang chạy (không phải benchmark bypass-HTTP).

## 1. Trạng thái trước/sau

| Thành phần | Trước | Sau |
|---|---|---|
| `POST /v1/search-sessions/{id}/commands/refine` | Có (frontier-wide decode + dense scoring, orchestrator ~921 dòng) | **Đã xoá** — benchmark xác nhận không bao giờ thắng `adaptive_coarse` trên dữ liệu thật, chi phí GPU cao hơn |
| `GET /v1/search-sessions/{id}/tuples` | Có | **Đã xoá** — HTTP surface duy nhất đọc lại `assemble_ordered_tuples`; nội bộ `bundle.artifacts.tuples` vẫn được `POST .../artifacts/frame-scores` và `commands/fix-frame` cập nhật (không xoá logic, chỉ xoá endpoint đọc) |
| `RefineSessionRequest`, `LiveRefinementOrchestrator` | Có | **Đã xoá** (đổi tên/thu gọn thành `BoundaryRefinementRuntime` trong `refinement_runtime.py`, ~230 dòng) |
| `boundary_refinement.py`, `coarse_anchor.py` (region-seed) | Chỉ có trong `research_tools/benchmarks/youcook2/` | **Port vào `src/adaptive_search/boundary_refinement.py` + `boundary_seeds.py`**, dùng lại `bundle.artifacts.regions/candidates` đã có sẵn trong session thay vì bypass-HTTP |
| `GET /v1/search-sessions/{id}/video-priorities` | Trả `priorities` thô, không refine | Có thêm `apply_boundary_refinement` (query param, **mặc định `true`**), refine **mọi** event của **mọi** video trả về, kèm `boundary_refinement_capability` |
| `POST /temporal-search` (legacy_temporal, legacy_ambiguous) | Không refine | Có thêm `apply_boundary_refinement` (body field, **mặc định `false`**), refine mọi candidate trong mọi tuple trả về |
| `RetrievalHyperparameters` | `top_n_per_variant=200, top_n_fused=500` | `top_n_per_variant=500, top_n_fused=1000` (xem §3) |
| `RefinementHyperparameters` weights | `coverage=.5 mean=.3 min=.2` | `coverage=0 mean=1.0 min=0` (xem §3) |
| `algorithms.py` (clustering/ranking core) | — | **Không đổi** |
| `frontier`, `assemble_ordered_tuples`, `RankingHyperparameters`, `proposal_profiles.py` | — | **Không đổi**, vẫn được dùng (xem §2) |

`adaptive_full` **chưa từng là một module thật**: `SearcherType` trong
`session.py` chỉ có `adaptive_temporal` — "coarse" và "full" chỉ là nhãn của
benchmark runner gọi các tập con khác nhau của cùng một bộ HTTP endpoint. Vì
vậy "loại bỏ adaptive_full" ở đây nghĩa là loại bỏ code path
live-dense-refinement-toàn-frontier (`commands/refine`, `GET /tuples`,
`LiveRefinementOrchestrator`), không phải xoá một module.

## 2. Vì sao `adaptive_full` bị loại bỏ, và điều gì được giữ lại

`hyperparameter_sweep_results.md` (n=50 câu hỏi thật) xác nhận: với retrieval
đủ rộng, không tìm được cấu hình frontier nào (`max_initial_videos`,
`max_total_regions`, `max_regions_per_event_per_video`, `max_frames_per_run`)
mà `adaptive_full` thắng `adaptive_coarse` trên recall thật, trong khi chi phí
GPU decode+embed của nó luôn cao hơn hẳn. `boundary_refinement_results.md`
cho thấy một stage rẻ hơn nhiều — sweep local ±10-native-frame quanh anchor đã
chọn, dùng lại change-point detector sẵn có — cải thiện precision thời điểm
đo được, với chi phí chỉ bằng một phần nhỏ của frontier-wide refine.

**Bị xoá** (`refinement.py`, 921 dòng): `refine_session`, `_refine_video_group`,
`_score_grouped`, budget planning (`_natural_budgets`, `_dense_budget`),
`_select_regions`, `_extract_region_frames`, các dataclass kết quả trung gian.

**Được giữ nguyên, có lý do rõ ràng** (không phải bỏ sót):
`select_refinement_frontier`, `_rebuild_frontier`, `frontier_region_ids`, stage
invalidation `"frontier"`, `ArtifactCounts.frontier_regions`, `GET /regions`'s
`selected_for_refinement`, `POST /artifacts/frame-scores`,
`assemble_ordered_tuples`/`RankingHyperparameters`/`TupleResult`,
`proposal_profiles.py`. Lý do: các phần này là pure/deterministic, không có
chi phí GPU/model (toàn bộ chi phí thật nằm ở vòng lặp decode-and-embed đã bị
xoá), vẫn được test bởi các file không liên quan riêng đến refine
(`test_adaptive_constraints.py`, `test_adaptive_algorithms.py`,
`test_adaptive_invalidation.py`), và `commands/fix-frame` vẫn phụ thuộc hợp lệ
vào `assemble_ordered_tuples`. `proposal_profiles.py` còn được
`boundary_refinement.py` dùng trực tiếp (xem §4), không hề là dead code.

**Được giữ và đổi tên**: `refinement_runtime.py` (trước là `refinement.py`)
giữ lại đúng phần capability-reporting/text-fallback mà stage rẻ mới cần:
`BoundaryRefinementRuntime.capabilities()` (đổi tên từ
`LiveRefinementOrchestrator`), `configured_runtime_spec()`,
`_validate_embedder_identity`, `state_queries()`, `score_frames()`. Class này
cố tình **không** còn phụ thuộc `AdaptiveSearchService` (bản cũ đọc
`service.frame_provider`) — boundary refinement là stateless và được
`legacy_search` dùng lại y hệt, mà `legacy_search` không có session/service
riêng.

## 3. Hyperparameter mặc định mới

Nguồn: sweep n=50 câu hỏi thật, chi tiết đầy đủ trong
`hyperparameter_sweep_results.md` (không lặp lại số liệu ở đây).

| Hyperparameter | Trước | Sau | Lý do |
|---|---|---|---|
| `RetrievalHyperparameters.top_n_per_variant` | 200 | **500** | Retrieval rộng hơn cải thiện recall, không có nhược điểm nào được tìm thấy |
| `RetrievalHyperparameters.top_n_fused` | 500 | **1000** | (như trên) |
| `RetrievalHyperparameters.rrf_k` | 60 | 60 (không đổi) | Sweep không tìm được giá trị tốt hơn |
| `RefinementHyperparameters.video_coverage_weight` | 0.5 | **0.0** | mean-only thắng mọi blend có coverage/min một khi retrieval đã rộng (finding #2) |
| `RefinementHyperparameters.video_mean_weight` | 0.3 | **1.0** | (như trên) |
| `RefinementHyperparameters.video_min_weight` | 0.2 | **0.0** | (như trên) |
| `ClusteringHyperparameters.gap_seconds` | 3.0 | 3.0 (không đổi) | Sweep (n=60) xác nhận `gap_seconds` **vô hiệu** dưới thiết kế hiện tại: `seeds[0]` luôn là candidate có `raw_relevance_score` cao nhất, và candidate đó luôn thuộc region có `raw_coarse_score` cao nhất (vì `raw_coarse_score = max` trong region) — bất biến với việc chia region theo `gap_seconds`. Xác nhận bằng cả đọc code lẫn thực nghiệm (byte-identical trên `gap_seconds` từ 0.5 đến 30.0) |
| `boundary_refinement`'s `radius_frames`/`stride` | (chỉ có trong benchmark) | **10 / 1** | Sweep không tìm được cấu hình nào tốt hơn; port nguyên giá trị |

`validate_refinement_configuration`'s ràng buộc `coverage+mean+min > 0` vẫn
đúng với `0+1.0+0=1.0`.

## 4. `boundary_refinement` như một stage thật

### 4.1 Kiến trúc

`boundary_refinement.py::refine_event_boundary()` (port từ benchmark, một
correctness fix quan trọng — xem §4.2):

1. Nhận một anchor (giây) đã được pipeline khác chọn sẵn cho một event trong
   một video.
2. Lấy `radius_frames=10` frame gốc (native fps) mỗi bên, cách nhau `stride=1`
   frame — qua `FrameProvider.get_frames()` (PyAV, PTS thật).
3. Encode các frame đó cùng `anchor_query`/`pre_state`/`post_state` bằng
   `TextImageEmbedder` (SigLIP2), dùng lại `score_event_frames()`.
4. Gọi `generate_profiled_proposals()` (dispatch theo `boundary_type`, **không
   phải** `generate_boundary_proposals()` thô) để chọn proposal có
   `final_event_score` cao nhất trong cửa sổ vừa sample.
5. Trả `RefinementOutcome(anchor_seconds, refined_seconds, used_fallback,
   sampled_frame_count, top_proposal)`. Nếu provider không trả frame nào
   (`used_fallback=True`), `refined_seconds == anchor_seconds`.

Stage này **stateless và không persist** — không ghi vào session, không có
artifact riêng, không invalidation stage mới. Mỗi request tính lại từ đầu.

Seed nào được refine cho `adaptive_coarse`: `boundary_seeds.select_event_seeds()`
(port từ benchmark's `coarse_anchor.py`) chọn `seeds[0]` — candidate có
`raw_relevance_score` cao nhất trong tối đa 2 region tốt nhất (ngưỡng 10% so
với region tốt nhất, tối đa 6 seed) — **chỉ seed tốt nhất được refine**, không
so sánh `final_event_score` giữa nhiều seed độc lập (mỗi seed được
`robust_sigmoid` calibrate trên population riêng của nó — so sánh trực tiếp
giữa các neighborhood khác nhau là không hợp lệ về mặt toán học).

`legacy_search` **không dùng `boundary_seeds`**: mỗi `ClusteredCandidate` mà
`TemporalSearcher`/`AmbiguousSearcher` chọn đã là một frame duy nhất cho một
event — không có cấu trúc region/candidate-pool để chọn seed như
`adaptive_coarse`. Anchor refine trực tiếp là
`parse_display_timestamp(candidate.timestamp)` (chuỗi `"MM:SS"` upstream trả
về, **không phải** `frame_index` — đó là keyframe index nội bộ của upstream,
không liên quan fps thật).

### 4.2 Correctness fix so với bản benchmark: `pre_state`/`post_state` optional

Bản benchmark của `refine_event_boundary()` bắt buộc `pre_state`/`post_state`
non-null (chỉ gọi cho các câu hỏi "khoảnh khắc đầu/cuối" có state synthesize
sẵn). `legacy_search` không có state semantics nào cho một câu query tự do bất
kỳ. Nếu gán cùng một fallback text cho cả `pre_state` và `post_state`,
`pairwise_softmax(x, x)` trả đúng `(0.5, 0.5)` cho **mọi** frame (xác minh trực
tiếp trong `algorithms.py`) — contrast score của transition-detector suy biến
thành nhiễu thuần, một lỗi correctness âm thầm chứ không phải giả thuyết.

Fix: `generate_profiled_proposals()` (`proposal_profiles.py`) đã sẵn có dispatch
`boundary_type="unknown"` + `pre_state=None`/`post_state=None` sang một profile
khác hẳn ("state" — anchor-persistence re-centering trên `normalized_anchor_score`
riêng, không đụng đến cột pre/post) thay vì transition path suy biến. Production
`boundary_refinement.py` gọi `generate_profiled_proposals()`, không gọi hàm thô.

**Giới hạn thật, ghi rõ chứ không giấu**: `legacy_search`'s boundary refinement
là **anchor-persistence re-centering, không phải true onset/offset detection**.
Chỉ khi caller truyền `boundary_type="onset"/"offset"/"transition"` kèm
`pre_state`/`post_state` thật (như luồng
`POST /v1/search-sessions/from-queries` của `adaptive_coarse`, dùng LLM rewrite
để sinh state text) thì mới có transition detection thật.

## 5. API contract mới

### 5.1 `GET /v1/search-sessions/{id}/video-priorities`

Thêm query param `apply_boundary_refinement: bool = true` (mặc định **BẬT**).
Mỗi item trong `items` có thêm field `boundary_refinement`:

```json
{
  "video_id": "5haTwcEIyE8",
  "priority_score": 0.417977127000812,
  "boundary_refinement": {
    "status": "applied",
    "events": [
      {
        "event_id": "e2",
        "anchor_seconds": 158.0,
        "refined_seconds": 157.96,
        "used_fallback": false,
        "boundary_type": "unknown",
        "sampled_frame_count": 21
      }
    ]
  }
}
```

Response top-level có thêm `boundary_refinement_capability: {requested,
available, reason}`. `status` ∈ `{applied, skipped_runtime_unavailable,
skipped_no_metadata, skipped_video_not_in_catalog, not_requested}`.

### 5.2 `POST /temporal-search`

Request thêm field `apply_boundary_refinement: bool = false` (mặc định
**TẮT** — khác mặc định với adaptive vì đây là endpoint cũ, không được đổi
hành vi mặc định để tránh phá client hiện tại). Mỗi `ClusteredCandidate` trong
`tuple` có thêm `refined_timestamp_seconds` và `boundary_refinement_status`
(cùng tập giá trị §5.1). Response top-level có thêm
`boundary_refinement_capability` y hệt §5.1.

Ví dụ thật (flag bật, group `FTdfwoxgMTU`, xem §7 để có toàn bộ request):

```json
{
  "score": 0.0011807513780098633,
  "video_name": "FTdfwoxgMTU.mp4",
  "tuple": [
    {
      "frame_index": 5754,
      "timestamp": "3:12",
      "score": 0.148848,
      "refined_timestamp_seconds": 191.658,
      "boundary_refinement_status": "applied"
    }
  ]
}
```

### 5.3 Graceful-degradation contract (chung cho cả hai endpoint)

Một lần kiểm tra capability cho mỗi request
(`BoundaryRefinementRuntime.capabilities()`) → nếu không khả dụng, mọi item
được đánh dấu `skipped_runtime_unavailable`, response vẫn `200` với
`boundary_refinement_capability.available=false`. Khi capability khả dụng,
một lần kiểm tra cho mỗi item (thiếu metadata / video không có trong catalog —
cả hai đều là trường hợp thật, không phải giả thuyết, xem §7.2) → item đó
được đánh dấu tương ứng, giữ nguyên kết quả chưa refine, tiếp tục các item
khác. Không có mã lỗi HTTP mới cho "chưa cấu hình" — chỉ lỗi thật giữa chừng
(embedding/runtime error) mới raise loudly (tái dùng
`ERROR_CODE_LIVE_REFINEMENT_*`, `legacy_search`'s `router.py` dùng lại đúng
`adaptive_search.router._raise_api_error`).

## 6. Chi phí/latency khi refine mọi kết quả

Quyết định phạm vi (xác nhận với user trước khi build): refine **mọi** event
của **mọi** video/tuple trả về, không chỉ top-1/top-N. Chi phí worst-case mỗi
request là `O(số video/tuple × số event × (2×radius_frames+1) frame decode +
embed)`. Với `radius_frames=10` → tối đa 21 frame/event. Batch thật trong §7:
`adaptive_coarse` với `top_k=50` retrieve và ~90-105 video/group mất vài giây
đến vài chục giây mỗi group (embedding SigLIP2 trên GPU, batch size 32).
Khuyến nghị vận hành: giữ `top_k_tuple`/`top_k_each_query` (legacy) và số
video trả về (`limit` của `video-priorities`, adaptive) ở mức hợp lý khi bật
flag, vì chi phí tuyến tính theo số kết quả chứ không có cap cứng nào khác.

## 7. Kết quả xác minh sáu chiều (real service, không phải benchmark bypass)

Script: `scripts/verify_migration.py` (ad hoc, không thuộc test suite chính
thức). Chạy thật với `.venv/bin/python3 scripts/verify_migration.py` nhắm vào
`src/main.py` đang chạy port 8001, dùng 5 group câu hỏi YouCook2 thật (2 group
3-event, 3 group 4-event — bộ dữ liệu 203 file query thật trong
`YOUCOOK2_DATA_ROOT/query` chỉ có group 3 hoặc 4-event, không có group 1-event
để test). Toàn bộ 6 kiểm tra: `legacy_temporal`/`legacy_ambiguous` ×
flag on/off, `adaptive_coarse` × flag on/off (mặc định, không truyền query
param) — tổng thời gian chạy thật **7m35s** cho GPU inference + upstream
retrieval thật.

### 7.1 Kết quả tóm tắt (từ `scratch/verify_migration_results.json`)

| Check | Trạng thái | Ghi chú |
|---|---|---|
| 1. `legacy_temporal`, flag off | OK 200, mọi candidate `not_requested` | 60/60 candidate (chỉ 1/5 group có tuple hợp lệ — xem §7.2) |
| 2. `legacy_temporal`, flag on | OK 200, mọi candidate `applied` | 60/60 `applied`, `refined_timestamp_seconds` lệch anchor <=0.4s (xem ví dụ §5.2) |
| 3. `legacy_ambiguous`, flag off | OK 200, mọi candidate `not_requested` | 60/60, byte-identical shape với check 1 ngoại trừ `boundary_refinement_status` |
| 4. `legacy_ambiguous`, flag on | OK 200, mọi candidate `applied` | 60/60 `applied` |
| 5. `adaptive_coarse`, flag off (`apply_boundary_refinement=false`) | OK 200, `capability.requested=false` | 421/421 item `not_requested` trên cả 5 group |
| 6. `adaptive_coarse`, mặc định (không truyền query param → on) | OK 200, `capability.available=true` | 421/421 item `applied`; `refined_seconds` khác `anchor_seconds` rõ ràng ở nhiều event (delta mẫu: -0.33s .. +0.41s), xác nhận seed-selection + refine chạy thật hết pipeline |

### 7.2 Phát hiện thật trong lúc xác minh (không phải bug của migration này)

- **`YOUCOOK2_METADATA_ROOT` chưa được cấu hình trong `.env`** trước khi bắt
  đầu xác minh — khiến `catalog.metadata()` luôn trả `None` cho mọi video
  (endpoint scan `*_keyframes.json`, không phải `<video_id>.json` trơn), tức
  là mọi refine đều bị `skipped_no_metadata` dù capability khả dụng. Đã thêm
  `YOUCOOK2_METADATA_ROOT=/mnt/c/Users/huynh/Downloads/youcook2/metadata` vào
  `.env` (thư mục này đã có sẵn 1660 file `*_keyframes.json`, bao gồm đủ cho
  cả 5 video ground-truth dùng trong xác minh) — đây là một gap cấu hình môi
  trường có sẵn từ trước, không phải lỗi phát sinh từ migration, nhưng **bắt
  buộc phải sửa để "applied" path chạy được thật** trong production.
- **`legacy_temporal`/`legacy_ambiguous` chỉ tìm được tuple hợp lệ cho 1/5
  group** ở `top_k_each_query=50` (VswrGW9b3ck, 0IuQKThr-pM, 0JVmVXLrNZo,
  0Mz4NTozNXw đều trả `results: []`). Hành vi này **giống hệt nhau khi flag
  bật/tắt** — xác nhận `apply_boundary_refinement` không hề ảnh hưởng đến
  retrieval/backtracking, chỉ post-process tuple đã có. Đây là đặc điểm có
  sẵn của `TemporalSearcher`/`AmbiguousSearcher` (cần cùng một video nhận đủ
  hit từ top-50 sparse search cho **mọi** event) kết hợp với recall tiếng Việt
  hạn chế (xem `hyperparameter_sweep_results.md`) chứ không phải regression
  của migration này. Không nằm trong phạm vi báo cáo này để cải thiện recall
  (xem §8).

### 7.3 Hai kiểm tra chế độ suy giảm (degraded mode), qua restart process thật

Không chỉ unit test — tắt hẳn `YOUCOOK2_DATA_ROOT`/`YOUCOOK2_METADATA_ROOT`
trong `.env`, restart `src/main.py` thật (không phải monkeypatch trong cùng
process), xác nhận qua `GET /v1/searchers`:
`live_refinement_available=false`, `reason="no raw video or frame API is
configured"`. Sau đó:

- **legacy_temporal, flag on**: `POST /temporal-search` → 200,
  `boundary_refinement_capability={requested:true, available:false, reason:
  "no raw video or frame API is configured"}`, mọi candidate
  `skipped_runtime_unavailable`, không candidate nào có
  `refined_timestamp_seconds`.
- **adaptive_coarse, mặc định (on)**: full lifecycle
  (`POST /v1/search-sessions` → `POST commands/retrieve` →
  `GET video-priorities`) → 200 ở mọi bước, capability giống hệt trên, mọi
  item `skipped_runtime_unavailable`.

Cả hai đều **200**, không có lỗi 5xx nào — đúng hợp đồng graceful-degradation
qua toàn bộ process startup thật, không chỉ qua test giả lập. Sau khi xác
minh xong, `.env` được khôi phục lại đầy đủ và service được restart lại lần
nữa; `GET /v1/searchers` xác nhận `live_refinement_available=true` trở lại.

## 8. Rủi ro còn lại / phạm vi không bao gồm

- **`GET /tuples` bị xoá** làm giảm khả năng debug/discover tuple qua HTTP
  (trước đây có thể đọc lại `proposal`/`tuple` đã build); `commands/fix-frame`
  vẫn hoạt động đúng vì nó không phụ thuộc endpoint đọc, chỉ phụ thuộc
  `assemble_ordered_tuples` nội bộ (không đổi).
- **`frontier`/tuple internals cố tình không đụng tới** (§2) — không phải bị
  bỏ sót, mà là quyết định phạm vi có lý do rõ ràng.
- **Không có re-validation recall nào mới** trong báo cáo này — recall của
  `adaptive_coarse` đã được benchmark xác nhận riêng trong
  `hyperparameter_sweep_results.md`; báo cáo này chỉ xác nhận **migration**
  (đưa code + default vào service thật) hoạt động đúng, không đo lại chất
  lượng tìm kiếm.
- **3 tài liệu cũ (`ADAPTIVE_TEMPORAL_SEARCH.md`, `STREAMLIT_UI_IMPLEMENTATION_PLAN.md`,
  `YOUCOOK2_RUNTIME_AND_BENCHMARK.md`) từng mô tả `commands/refine`/
  `GET /tuples` cũ như thể còn hoạt động** — đã được sửa (2026-08-18): mỗi file
  thêm banner/ghi chú trỏ về tài liệu này, và các bảng endpoint/luồng cụ thể
  (không phải toàn bộ văn xuôi liên quan) đã được cập nhật để phản ánh đúng
  `GET .../video-priorities` thay cho `GET .../tuples`, và
  `POST .../commands/retrieve` thay cho `POST .../commands/refine`. Tài liệu
  hiện tại (`ADAPTIVE_PIPELINE_MIGRATION.md`) vẫn là nguồn đúng nhất cho trạng
  thái endpoint sau migration.
- **`legacy_search` recall thấp ở `top_k_each_query` nhỏ** (§7.2) không được
  cải thiện trong phạm vi migration này — chỉ đúng khi flag refine bật/tắt
  đều cho kết quả giống hệt nhau về mặt retrieval, đúng như kỳ vọng.

## 9. Cập nhật 2026-08-18: `POST /artifacts/frame-scores` và refinement frontier bị xoá

§2 ở trên từng liệt kê `select_refinement_frontier`, `_rebuild_frontier`,
`frontier_region_ids`, stage `"frontier"`, `ArtifactCounts.frontier_regions`,
`GET /regions`'s `selected_for_refinement`, và `POST /artifacts/frame-scores`
là **"được giữ nguyên, có lý do rõ ràng"** vì chúng pure/deterministic, không
tốn chi phí GPU. Quyết định đó bị đảo ngược hôm nay: sản phẩm xác nhận sẽ
**không bao giờ hỗ trợ một client bên ngoài tự tính điểm frame rồi nộp lại
vào session** — đây là workflow duy nhất mà toàn bộ cụm tính năng này tồn tại
để phục vụ. `apply_boundary_refinement` (§4-§5, live, tự tính điểm mỗi
request) là cơ chế refine đang hoạt động thật và **không bị ảnh hưởng**.

**Đã xoá khỏi `src/adaptive_search/`:**
- `POST /artifacts/frame-scores`, `GET .../frame-scores` (endpoint), `AdaptiveSearchService.replace_frame_scores()`, `FrameScoreIngestRequest`.
- `select_refinement_frontier` (`algorithms/ranking.py`), `_rebuild_frontier`, `_frame_score_acceptance_window` (`service_helpers.py`).
- `ArtifactState.frontier_region_ids` / `.frame_scores`, `ArtifactCounts.frontier_regions` / `.frame_scores`, `GET /regions`'s `selected_for_refinement`.
- Stage `"frontier"` khỏi `PIPELINE_STAGES`; `ClusteringHyperparameters` (toàn bộ class — `gap_seconds`/`margin_seconds`/`max_region_seconds` chỉ còn được `_frame_score_acceptance_window` dùng, giờ cũng bị xoá); `RefinementHyperparameters.max_initial_videos`/`max_regions_per_event_per_video`/`max_total_regions`/`exploration_region_ratio`.

**Vẫn giữ, không đổi** (đúng như lý do đã ghi ở §2 — không phụ thuộc frame-score ingestion): `calibrate_frame_scores`, `generate_boundary_proposals`, `generate_profiled_proposals`, `proposal_profiles.py`, `FrameScoreSample`/`EventProposal` schema, `commands/fix-frame`, `GET /proposals`. `boundary_refinement.py` tự tính `FrameScoreSample` thô mỗi request rồi gọi thẳng `generate_profiled_proposals()` — không hề đọc/ghi `bundle.artifacts.frame_scores`.

**Research tools cũng dọn theo:** `adaptive_full` (benchmark pipeline mode) bị xoá khỏi `tuple_runner.py`/`cli.py` — nó đã phụ thuộc `commands/refine`/`GET /tuples` (xoá ở §1) và giờ thêm cả các hyperparameter frontier vừa xoá, nên không còn cách nào chạy được. `research_tools/benchmarks/youcook2/legacy_clustering.py` (dùng cho ablation "clustered vs atomic" có sẵn từ trước) định nghĩa `ClusteringHyperparameters` cục bộ thay vì import từ `adaptive_search.schemas`.
