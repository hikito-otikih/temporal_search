# Adaptive Temporal Search: thiết kế, contract và hướng dẫn tích hợp

Tài liệu này mô tả đúng vertical slice đang có trong repository. Nó là nguồn
tham chiếu cho frontend, demo ý tưởng và phần phương pháp của paper. Raw-video
provider và live-refinement orchestration đã có; những model/benchmark chưa chạy
thật vẫn được ghi rõ và không nên diễn giải như kết quả paper đã được xác nhận.

## 1. Trạng thái triển khai

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Strict domain schema, timestamp theo giây | Hoàn tất | Pydantic từ chối field lạ, NaN/Inf và giá trị ngoài miền |
| Fusion nhiều query variant bằng RRF | Hoàn tất | Deduplicate theo `(video_id, frame_id)`, vẫn giữ một `event_id` |
| Temporal region, video priority, refinement frontier | Hoàn tất | Deterministic và có budget cứng |
| Anchor/pre/post calibration | Hoàn tất | Raw score được giữ riêng; normalized score được tính cục bộ |
| Transition/state/symmetric-peak proposal và temporal NMS | Hoàn tất | Scorer được dispatch theo `boundary_type` |
| Ordered same-video tuple assembly | Hoàn tất | Hard constraint, adjacent-gap penalty và runtime caps |
| Session API, optimistic revision, scoped invalidation | Hoàn tất | Store hiện là in-memory, single process |
| SigLIP2 embedding boundary, lazy runtime và cache-key helper | Đã nối vào live orchestrator | Startup không tải weights; actual SigLIP2 inference vẫn chưa được chạy/xác nhận với dependency và immutable revision thật |
| YouCook2 raw-video decode theo PTS | Hoàn tất | `YouCook2FrameProvider` dùng PyAV; đã catalog 1.660 video và decode thật một frame với actual PTS |
| Medium/dense sampling và automatic anchor/pre/post scoring | Hoàn tất ở API/orchestrator | `POST .../commands/refine`; fake embedder E2E đã test, GPU SigLIP2 thật chưa xác nhận |
| Qwen ordered-frame reranker | Chưa có runtime | Model spec/instruction đã đóng; `use_reranker=true` hiện bị reject thay vì silent no-op |
| Persistent DB/cache, queue, SSE/cancel | Chưa có | Để cho production phase |
| YouCook2 corpus Video Recall@K | Có package và pilot | Chỉ đo ground-truth video trong top-K; chưa phải full validation/paper result và chưa đo temporal boundary |

Endpoint `GET /v1/searchers` luôn báo `algorithm_core_available=true`; trường
`live_refinement_available` chỉ true khi **cả** dense `FrameProvider` và
text-image embedder đều khả dụng. `frame_provider` vẫn là sub-capability riêng
để frontend giải thích chính xác thiếu decoder hay thiếu model.

## 2. Bài toán và pipeline

Mỗi câu hỏi được biểu diễn thành một chuỗi event có thứ tự. Mỗi event không chỉ
có mô tả khoảnh khắc cần tìm (`anchor_query`) mà còn có thể có trạng thái quan
sát được ngay trước và ngay sau (`pre_state`, `post_state`). Hệ thống tìm một
mốc cho mỗi event trong cùng video, theo đúng thứ tự thời gian.

```text
rewrite/event definitions
          |
          v
4 retrieval variants / event
          |
    upstream retrieval
          |
 RRF fusion + frame dedup
          |
 temporal regions (seconds)
          |
 coverage-aware video priority
          |
 budgeted refinement frontier
          |
 medium/dense frame sampling       [YouCook2 PyAV provider hoặc provider khác]
          |
 anchor / pre / post cosine curves [live command hoặc ingest precomputed]
          |
 local calibration
          |
 profile-aware proposals + NMS
          |
 hard constraints + bounded ordered assembly
          |
 top-k tuple view
```

Legacy `/temporal-search` vẫn là pipeline độc lập và không dùng score adaptive.
Legacy lấy candidate từ `127.0.0.1:8000/search`, nhóm theo video và xếp tuple
bằng frame index. Không so trực tiếp legacy score với adaptive score.

## 3. Contract dữ liệu

### 3.1 EventDefinition

| Field | Ý nghĩa |
|---|---|
| `event_id` | ID ổn định trong session; các query variant dùng chung ID này |
| `original_query` | Câu hỏi gốc, dùng để trace/rewrite |
| `anchor_query` | Mô tả hình ảnh của khoảnh khắc/trạng thái mục tiêu |
| `pre_state` | Trạng thái nhìn thấy trước boundary; nên có cho transition |
| `post_state` | Trạng thái nhìn thấy sau boundary; nên có cho transition |
| `boundary_type` | `onset`, `offset`, `transition`, `plateau_start`, `state`, `symmetric_peak`, hoặc `unknown` |

`unknown` được xử lý như transition nếu cả `pre_state` và `post_state` tồn tại;
nếu không, nó được xử lý như `state`.

### 3.2 Trục thời gian

- Adaptive core dùng `timestamp_seconds: float`.
- Frame provider dùng `pts_ms: int`; đổi sang giây bằng `pts_ms / 1000`.
- `frame_id` là định danh/UI compatibility, không được tự coi là giây.
- `timestamp` dạng chuỗi và `frame_index` của legacy chỉ được giữ ở boundary
  legacy hoặc catalog metadata.

### 3.3 Các artifact chính

| Artifact | Field score quan trọng | Phạm vi |
|---|---|---|
| `SparseCandidate` | `raw_relevance_score`, `normalized_relevance_score` | event/video/frame/query variant |
| `TemporalRegion` | `raw_coarse_score`, `normalized_coarse_score` | event/video/time interval |
| `FrameScoreSample` | raw và normalized anchor/pre/post/motion | event/video/region/frame |
| `EventProposal` | semantic, boundary, pre/post consistency, `final_event_score` | một event tại một timestamp |
| `TupleResult` | event mean, gap penalty, bonus, raw/normalized final score | đủ chuỗi event trong một video |

Raw score luôn được giữ để debug và tái lập. Normalized score trong `[0,1]`
là score xếp hạng cục bộ, **không phải xác suất đúng**.

## 4. Thuật toán theo từng stage

### 4.1 Query-variant fusion

Rewrite hiện sinh hai câu Việt và hai câu Anh cho mỗi event. Candidate của các
variant được xếp hạng riêng rồi hợp nhất bằng Reciprocal Rank Fusion:

```text
RRF(frame) = sum_v 1 / (rrf_k + rank_v(frame))
```

Mặc định `rrf_k=60`. Mỗi variant lấy tối đa 50 kết quả; sau fusion giữ tối đa
100 frame/event. Một event nhận tối đa bốn variant. Cùng một frame trong một
variant chỉ giữ row mạnh nhất; cùng frame giữa các variant cộng RRF. Vì vậy bốn
retrieval query không bị hiểu sai thành bốn event thời gian.

Score gốc từ các ngôn ngữ/model khác nhau không bị cộng trực tiếp; nó chỉ quyết
định rank bên trong từng variant.

### 4.2 Temporal regions

Candidate được nhóm theo `(session, event, video)` và sắp theo giây.

- Bắt đầu region mới nếu khoảng cách với candidate trước lớn hơn 3 s.
- Core span cũng bị giới hạn ở `30 - 2*3 = 24 s` để sau khi thêm margin, region
  không dài hơn 30 s.
- Mở rộng mỗi phía 3 s, chặn start tại 0.
- Chỉ merge region chồng nhau nếu cùng event/video/status và union không vượt
  `max_region_seconds`.
- Coarse score là max RRF trong region. Calibration candidate được làm riêng
  theo từng `(session, event)`, không để phân phối của event khác làm đổi score.

### 4.3 Video priority và refinement frontier

Với mỗi video, lấy evidence tốt nhất cho từng event. Gọi `coverage` là tỷ lệ
event có evidence, `mean` là trung bình vector evidence (event thiếu nhận 0),
và `min` là event yếu nhất:

```text
video_priority = 0.5 * coverage + 0.3 * mean + 0.2 * min
```

Frontier mặc định ưu tiên năm video đầu, tối đa ba region/event/video và 60
region toàn run. Trước khi fill theo rank, thuật toán cố giữ ít nhất một region
độc lập cho mỗi event. 15% budget dành cho exploration có thứ tự ổn định. Region
do người dùng fix được ưu tiên hơn automatic budget.

`medium_interval_seconds=0.5`, `dense_interval_seconds=0.1` và
`dense_radius_seconds=1.0` là contract mặc định. Live orchestrator chia global
frame budget công bằng cho các region, sample medium toàn region, tìm anchor
peak rồi sample dense quanh peak. Nó có thể tự commit `FrameScoreSample` qua
`commands/refine`; ingest precomputed frame scores vẫn được giữ cho offline
experiments và provider ngoài.

### 4.4 Text-image curves và calibration

Với embedding đã L2-normalize, raw curve là cosine:

```text
s_anchor(t) = e(anchor) dot e(frame_t)
s_pre(t)    = e(pre)    dot e(frame_t)
s_post(t)   = e(post)   dot e(frame_t)
```

Anchor và motion được robust-calibrate trong từng event/video/region:

```text
m      = median(scores)
scale  = 1.4826 * median(abs(score - m))
z      = clip((score - m) / scale, -8, 8)
normalized = sigmoid(z)
```

Khi MAD bằng 0, giá trị tại median nhận 0.5; giá trị khác median nhận tail
sigmoid đã clip. Pre/post được chuẩn hóa theo cặp tại từng frame:

```text
p_pre(t)  = exp(s_pre/T)  / (exp(s_pre/T) + exp(s_post/T))
p_post(t) = exp(s_post/T) / (exp(s_pre/T) + exp(s_post/T))
```

Mặc định `T=1`. Do đó `p_pre(t) + p_post(t) = 1`, nhưng đây vẫn không phải
posterior probability đã calibrate bằng label.

### 4.5 Boundary profiles

#### Transition family

Áp dụng cho `onset`, `offset`, `transition`, `plateau_start`, và `unknown` có
đủ pre/post. Với mỗi center, thử mọi cặp left/right window trong
`[0.5, 1, 2, 3]` giây có ít nhất hai mẫu mỗi phía.

```text
post_contrast = mean_R(p_post) - mean_L(p_post)
pre_contrast  = mean_L(p_pre)  - mean_R(p_pre)

b_raw = 0.5*post_contrast + 0.5*pre_contrast + 0.1*motion
        - 0.01*normalized_window_length
        - 0.01*normalized_window_asymmetry
```

Chọn cặp window có `b_raw` lớn nhất, rồi robust-calibrate boundary score giữa
các center trong region.

#### Persistent state

Áp dụng cho `state` và `unknown` thiếu pre/post. Persistence là trung bình
normalized anchor trong lân cận `±max(window_options)=±3 s`. Scorer ưu tiên
center nằm trong một plateau có evidence ổn định thay vì một spike đơn lẻ.

#### Symmetric peak

Với mỗi window đủ mẫu ở hai phía:

```text
prominence = anchor(center) - 0.5*(mean_L(anchor) + mean_R(anchor))
```

Scorer chọn window có prominence lớn nhất, robust-calibrate prominence, đồng
thời dùng `1-mean_L(anchor)` và `1-mean_R(anchor)` để thưởng hai flank thấp.

#### Event score chung

```text
event_score = 0.40*semantic
            + 0.30*boundary
            + 0.10*pre_consistency
            + 0.20*post_persistence
```

Weights được chia cho tổng nên custom weights không cần cộng đúng 1. Với
`state`, ba thành phần ngoài semantic cùng biểu diễn persistence. Sau đó temporal
NMS giữ local maximum trong cùng event/video/region với radius 0.5 s, tối đa năm
proposal/region.

### 4.6 Ordered tuple ranking

Assembly chỉ ghép proposal của cùng video. Hard filter được áp dụng trước score:

- video/region/proposal bị reject;
- fixed video/region/frame/timestamp;
- allowlist video;
- strict chronological order;
- min/max adjacent gap và max tuple span.

Mỗi event/video chỉ đưa tối đa tám proposal vào backtracking. Mỗi video thử tối
đa 10.000 combination, giữ 200 tuple; toàn run giữ 2.000 trước khi trả top 20.

Với gap `g` giữa hai event kề nhau:

```text
gap_penalty(g) = lambda * max(0, g - tau)
```

Mặc định `tau=10 s`, `lambda=0.01`; constraint từng cặp có thể override. Tuple:

```text
raw_tuple_score = mean(event_scores)
                - mean(adjacent_gap_penalties)
                + 0.02 * fixed_event_count / event_count
```

`normalized_final_score` là robust sigmoid trên tập tuple hiện tại. Nó thay đổi
khi candidate set thay đổi và chỉ nên hiển thị như rank score, không phải độ tin
cậy tuyệt đối giữa hai session.

## 5. Hyperparameters mặc định

Đây là seed để chạy/ablation, chưa được tune trên validation set:

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
  quality_profile_enabled: false
  use_reranker: false
  max_initial_videos: 5
  max_regions_per_event_per_video: 3
  max_total_regions: 60
  exploration_region_ratio: 0.15
  max_frames_per_run: 2000
  embedding_batch_size: 64
  medium_interval_seconds: 0.5
  dense_interval_seconds: 0.1
  dense_radius_seconds: 1.0
  video_coverage_weight: 0.5
  video_mean_weight: 0.3
  video_min_weight: 0.2
boundary:
  window_options_seconds: [0.5, 1.0, 2.0, 3.0]
  min_samples_per_side: 2
  pairwise_temperature: 1.0
  anchor_clip_z: 8.0
  post_contrast_weight: 0.5
  pre_contrast_weight: 0.5
  motion_contrast_weight: 0.1
  window_length_regularization: 0.01
  window_asymmetry_regularization: 0.01
  semantic_weight: 0.4
  boundary_weight: 0.3
  pre_weight: 0.1
  post_weight: 0.2
  nms_radius_seconds: 0.5
  max_proposals_per_region: 5
ranking:
  top_k: 20
  default_gap_tau_seconds: 10.0
  gap_lambda: 0.01
  fixed_constraint_bonus: 0.02
  require_strict_order: true
  max_proposals_per_event_per_video: 8
  max_combinations_per_video: 10000
  max_tuples_per_video: 200
  max_total_tuples: 2000
```

`max_frames_per_run` có hard ceiling server-side 4.096 frame; batch encode ảnh
mặc định là 64 và bị chặn ở 512. Hai giá trị này tách biệt: frame budget kiểm
soát tổng công việc của run, còn `embedding_batch_size` kiểm soát peak VRAM.

Không dùng threshold 0.5 như một mặc định chất lượng. Khi có label, threshold
và/hoặc calibrator phải fit trên validation split và được lưu như artifact có
version.

## 6. Lựa chọn model

Người dùng **không cần chỉ định model ngay** để dùng algorithm core. Profile đã
chọn theo GPU RTX 4060 Laptop 8 GB của máy phát triển:

| Profile | Encoder | Dimension | Reranker | Dùng khi |
|---|---|---:|---|---|
| `balanced` | `google/siglip2-base-patch16-224` | 768 | tắt | mặc định thực dụng, chạy frame-level curve |
| `quality` | `Qwen/Qwen3-VL-Embedding-2B` | 1024 MRL | Qwen3-VL-Reranker-2B cho top 10–20 | demo chất lượng/ablation |
| `paper_full` | Qwen3-VL-Embedding-2B | 2048 | cùng reranker | accuracy ceiling của model 2B |

SigLIP2 và Qwen3-VL ở trên đều có license Apache-2.0. Qwen hỗ trợ tiếng Việt,
image/multi-image/video và context dài; reranker phù hợp để kiểm tra ordered
frames ở top nhỏ. Dual encoder vẫn là công cụ tạo curve chính vì rẻ hơn nhiều.

Instruction đã pin ở contract:

```text
Embedding: Retrieve video frames that visually depict the described event.
Reranker:  Judge whether the ordered video frames depict the described events
           in the required chronological order.
```

Với Qwen MRL, adapter yêu cầu vector full chưa normalize, cắt prefix 1024/2048
rồi L2-normalize lại. Không được cắt một vector 2048d đã normalize và coi dot
product mới là cosine.

### Chính sách ngôn ngữ cho anchor/pre/post

SigLIP2 là encoder đa ngôn ngữ, vì vậy `anchor_query`, `pre_state` và
`post_state` tiếng Việt **không bắt buộc** phải dịch sang tiếng Anh trước khi
chấm điểm. Mặc định nên giữ nguyên cả ba state cùng một ngôn ngữ để không làm
lệch quan hệ chuyển trạng thái. Nếu hệ thống rewrite đã sinh bản tiếng Anh ổn
định, hãy đánh giá `vi`, `en` và fusion `vi+en` như ba ablation riêng trên
validation split; lưu language policy vào run config/manifest và không âm thầm
thay câu tiếng Việt bằng bản dịch. Bản dịch sai động từ, trạng thái phủ định hoặc
thứ tự trước/sau có thể làm boundary score sai dù retrieval tổng quát vẫn tốt.

Giá trị `revision: main` trong default schema chỉ là template cấu hình. Adapter
live cố ý từ chối `main`; deployment/paper phải thay bằng immutable model commit.
Không tự động đổi sang model “SOTA mới nhất”, vì việc đó phá cache và tính tái
lập. Chỉ cần người dùng chỉ định model khác nếu có model server sẵn, giới hạn
license, VRAM khác hoặc protocol embedding riêng.

`GET /v1/searchers` công bố `runtime_model_spec` đầy đủ. Nếu client không gửi
`refinement.embedding_model` khi tạo balanced session, API copy đúng spec đang
cấu hình vào session; một spec client gửi rõ ràng luôn được giữ nguyên và
mismatch được thể hiện qua capability.

Nguồn model chính thức:

- <https://github.com/QwenLM/Qwen3-VL-Embedding>
- <https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B>
- <https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B>
- <https://huggingface.co/google/siglip2-base-patch16-224>

## 7. API contract cho frontend

### 7.1 Chọn searcher

`GET /v1/searchers` là capability endpoint. Frontend dùng:

- `legacy_temporal` → POST `/temporal-search` với
  `searcher_type="TemporalSearcher"`;
- `legacy_ambiguous` → POST `/temporal-search` với
  `searcher_type="AmbiguousSearcher"`;
- `adaptive_temporal` → session workflow dưới đây.

Không gửi legacy type vào `/v1/search-sessions`; endpoint này chỉ nhận
`adaptive_temporal`.

### 7.2 Session workflow

| Method | Endpoint | Chức năng |
|---|---|---|
| `POST` | `/v1/search-sessions` | Tạo session/event/config |
| `GET` | `/v1/search-sessions/{id}` | Session, revision, counts, combined live capability |
| `DELETE` | `/v1/search-sessions/{id}?expected_revision=n` | Xóa session |
| `PATCH` | `/v1/search-sessions/{id}/events/{event_id}` | Sửa event và invalidation có scope |
| `PATCH` | `/v1/search-sessions/{id}/hyperparameters` | Deep-patch config |
| `PUT` | `/v1/search-sessions/{id}/constraints` | Thay toàn bộ typed constraints/gap policy |
| `POST` | `/v1/search-sessions/{id}/artifacts/candidates` | Ingest candidates, chạy RRF/region/frontier |
| `POST` | `/v1/search-sessions/{id}/artifacts/frame-scores` | Ingest score curves, tạo proposal/tuple |
| `POST` | `/v1/search-sessions/{id}/commands/refine` | Decode/sample/score selected regions và commit proposal/tuple đồng bộ |
| `POST` | `/v1/search-sessions/{id}/commands/mark-videos` | Đặt video allowlist |
| `POST` | `/v1/search-sessions/{id}/commands/fix-frame` | Fix mốc event |
| `POST` | `/v1/search-sessions/{id}/commands/reject-proposal` | Loại proposal |
| `POST` | `/v1/search-sessions/{id}/commands/clear-event-constraint` | Xóa constraint của event |
| `GET` | `.../regions`, `.../proposals`, `.../tuples`, `.../runs` | Danh sách phân trang |

Tạo session tối thiểu:

```json
{
  "searcher_type": "adaptive_temporal",
  "common_query": "một màn múa lân",
  "events": [
    {
      "event_id": "e1",
      "original_query": "khoảnh khắc lân bắt đầu xoay",
      "anchor_query": "con lân vừa bắt đầu xoay trên cột",
      "pre_state": "con lân đứng ổn định, chưa xoay",
      "post_state": "con lân đang xoay trên cột",
      "boundary_type": "onset"
    }
  ]
}
```

Mọi mutation/run request sau đó gửi `expected_revision`. Nếu session đã bị sửa,
server trả `409 revision_conflict` cùng `current_revision`; frontend phải reload
session rồi áp lại thao tác trên revision mới. Patch/replace giống hệt trạng
thái hiện tại là no-op và không tăng revision.

`PUT .../constraints` thay toàn bộ constraint snapshot. Mỗi adjacent-gap pair
phải tham chiếu hai event kề nhau theo đúng thứ tự session; event ID lạ hoặc cặp
nhảy qua event trung gian nhận `422`.

Response list có `{items,total,offset,limit}`. Region list thêm
`selected_for_refinement`. Frontend nên poll session/runs trong vertical slice;
chưa có SSE hay background job progress.

Live refinement request là strict; field lạ nhận `422`:

```json
{
  expected_revision: 0,
  region_ids: [region-id],
  max_frames: 200
}
```

`region_ids` và `max_frames` đều optional. Khi bỏ `region_ids`, server dùng
frontier hiện tại; `max_frames` chỉ được hạ, không được vượt
`refinement.max_frames_per_run`. Runtime thiếu provider/model trả
`503 live_refinement_unavailable`; cấu hình model/sampling hoặc dữ liệu frame sai
contract trả `422`. Command kiểm tra revision trước decode và kiểm tra lại khi
atomic commit, nên stale work không thể ghi đè session mới.

Ảnh được encode theo microbatch `refinement.embedding_batch_size`. Lỗi load
weights/network/dtype hoặc runtime/OOM trong lần inference được map thành `503`
thay vì raw 500; run lỗi không commit artifact.

Artifact ingest yêu cầu ít nhất một `FrameScoreSample` cho **mỗi** `region_id`
được thay thế; request rỗng/partial nhận `422` và không xóa score cũ hay đánh dấu
region là completed. Live run hiện ghi `motion_scores_available=false`. Reranker
chưa có runtime nên `use_reranker=true` cũng nhận `422`.

### 7.3 Luồng UI đề xuất

1. Gọi `/v1/searchers`, disable nút “run live refinement” nếu capability false.
2. Gọi `/rewrite`, cho người dùng sửa `anchor/pre/post/boundary_type`, rồi tạo
   adaptive session. `/rewrite` hiện chưa tự sinh pre/post theo schema adaptive,
   nên cần mapper/editor ở bước này.
3. Coordinator gọi upstream retrieval cho các query variant và ingest candidate.
4. Hiển thị video coverage và region timeline; dùng `selected_for_refinement`
   để đánh dấu region nằm trong budget.
5. Nếu `live_refinement_available=true`, gọi `commands/refine` cho frontier hoặc
   các region người dùng chọn. Nếu capability false, vẫn có thể ingest
   precomputed frame-score artifact từ worker ngoài.
6. Hiển thị proposal theo từng event và tuple theo video; cho phép mark video,
   fix frame, reject proposal và clear constraint.
7. Sau mỗi command, dùng revision/counts mới để refresh đúng panel bị invalidated.

Adaptive artifacts chỉ có `video_id`; title, watch URL, duration và thumbnail
cần được join từ video catalog/asset service. Không suy ra URL trực tiếp từ ID.

## 8. Constraint precedence và invalidation

Rejection là hard constraint. Fixed frame mạnh hơn fixed region cũ, nhưng không
vượt qua rejected proposal/video. Fixed video/region có thể override allowlist;
global rejected-video vẫn thắng. Hard constraint được filter trước tuple score.

| Thay đổi | Stage đầu bị invalid | Artifact có thể reuse |
|---|---|---|
| `original_query` | rewrite | không có downstream artifact của event đó |
| `anchor_query` | retrieval | artifact event khác |
| `pre_state`, `post_state` | refinement | candidates, regions, frontier |
| `boundary_type` | proposal | frame embeddings/scores |
| retrieval config | retrieval | không reuse downstream |
| clustering config | region | fused candidates |
| frontier budget knobs | frontier | candidates và regions |
| model/sampling refinement config | refinement | candidates, regions, frontier |
| boundary config | proposal | frame scores |
| ranking config | tuple | proposals và mọi stage trước |
| mark videos | frontier | sparse candidates/regions |
| fix/reject/clear event constraint | proposal/tuple | frame scores và stage trước |

`artifact_fingerprint` canonicalize input và hash cùng `artifact_type` và
`code_version`. Image embedding cache key gồm:

```text
video_content_hash + pts_ms + model_id + immutable_revision
+ output_dimension + preprocess_version
```

Sampling policy không nằm trong image key vì cùng một frame/model tạo cùng
embedding. Text key thêm exact text và instruction. Helper và test đã có;
persistent cache/run-manifest storage chưa được nối vào session store.

## 9. Reproducibility và kế hoạch paper

Một run dùng cho paper phải lưu thêm, ngoài snapshot hyperparameters/constraints
đã có:

- git commit và dirty-worktree flag;
- dataset/video-catalog version và video content hash;
- upstream index/retrieval model version;
- rewrite model/prompt/schema version;
- embedding/reranker model ID + immutable revision;
- dimension, preprocess, instruction, dtype và sampling policy;
- seed, hardware, CUDA/PyTorch/Transformers version;
- cache hit/miss, số frame decode/encode/rerank và latency từng stage;
- calibration artifact và threshold version.

Evaluation tối thiểu:

- Video Recall@K;
- Proposal Recall trong tolerance ±1 s và ±2 s;
- mean/median absolute boundary error;
- ordered Tuple Recall@K và success rate đủ mọi event;
- latency p50/p95, frame encoded, peak VRAM và cache-hit rate;
- với interaction: số click, thời gian hoàn thành và correction rate.

Ablation nên dùng cùng retrieval budget và cùng split:

1. legacy temporal;
2. adaptive anchor-only;
3. + pre/post transition score;
4. + multi-window + temporal NMS;
5. + coverage/exploration frontier;
6. + Qwen quality embedding;
7. + ordered-frame reranker;
8. + human constraints.

Không tune threshold trên test set. Báo confidence interval theo video/query,
và tách latency cold-cache/warm-cache. Package `benchmarks/youcook2` đã có để
đo corpus Video Recall@K qua `http://127.0.0.1:8000/search`, chống ground-truth
leakage và xuất run manifest/checkpoint. Run hiện có chỉ là smoke/pilot; chưa đủ
để coi là kết quả full validation hoặc paper, và chưa đánh giá temporal interval.

## 10. Giới hạn cần xử lý tiếp

1. Cài optional live dependencies, pin immutable SigLIP2 commit và chạy inference
   thật trên GPU; đo correctness, VRAM, throughput và cold/warm latency. Lazy
   runtime hiện không tải weights lúc API startup, nhưng actual SigLIP2 inference
   chưa được xác nhận trên máy này.
2. Chạy full YouCook2 validation Video Recall@K và các ablation model/language/
   aggregation; pilot hiện tại chỉ kiểm tra pipeline và không phải paper result.
3. Bổ sung moment/boundary benchmark như DiDeMo hoặc labeled transition subset
   sau khi corpus retrieval ổn định.
4. Implement Qwen ordered-frame reranker cho top tuple nhỏ, chạy tuần tự nếu
   hai model không thể cùng resident trong 8 GB VRAM.
5. Mở rộng `/rewrite` để trả thẳng `anchor_query`, `pre_state`, `post_state`,
   `boundary_type` trong contract versioned.
6. Chuyển session/artifacts sang SQLite/Postgres/object store; thêm idempotency,
   queue, stale-job guard, cancel và SSE reconnect.
7. Thêm persistent video-catalog/embedding cache và run-manifest linkage trước
   các benchmark latency lớn.

Claim chính xác hiện tại là: **YouCook2 PyAV provider đã decode video thật theo
actual PTS; medium/dense orchestrator và `/commands/refine` đã chạy end-to-end
với fake embedder và lưu metrics/region status. Inference SigLIP2 thật vẫn chưa
được xác nhận, Qwen/rewrite-v2/persistent production runtime vẫn chưa có, và
YouCook2 mới chỉ có Video Recall@K pilot**.
