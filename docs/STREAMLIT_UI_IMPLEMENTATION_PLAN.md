# Kế hoạch triển khai Streamlit UI cho Temporal Search

## 1. Mục tiêu

Xây dựng một ứng dụng Streamlit phục vụ ba nhu cầu khác nhau nhưng dùng chung
backend FastAPI hiện có:

1. kiểm tra nhanh legacy temporal search;
2. tương tác đầy đủ với adaptive search theo session;
3. chạy và quan sát benchmark YouCook2 ở mức Video Recall@K trước khi mở rộng
   sang dense temporal localization.

UI là research/debug console, không phải frontend production. Mọi score raw,
normalized score, timestamp, hyperparameter và revision ảnh hưởng đến kết quả
phải xem và xuất được để hỗ trợ debug, ablation và viết paper.

## 2. Nguyên tắc thiết kế

- Backend là source of truth; Streamlit không tự cài lại clustering/scoring.
- Legacy là one-shot request, adaptive là session workflow. Không ép legacy tạo
  session và không giả lập adaptive bằng một giá trị `searcher` trong endpoint cũ.
- Adaptive dùng `timestamp_seconds: float`; `frame_index` và chuỗi `m:ss` chỉ là
  dữ liệu hiển thị/legacy.
- Mọi mutation adaptive gửi `expected_revision`; khi gặp revision conflict, UI
  tải lại session và yêu cầu người dùng áp dụng lại thay đổi.
- Hyperparameter đang chỉnh trong form chưa ảnh hưởng backend cho tới khi người
  dùng bấm **Apply & recompute**.
- Hiển thị riêng raw score và normalized score; normalized score không được ghi
  nhãn là probability/confidence.
- Capability phải lấy từ `GET /v1/searchers`. Khi
  `live_refinement_available=false`, vô hiệu hóa nút chạy dense live nhưng vẫn
  cho phép ingest file frame scores.
- Query hint hoặc ground truth chỉ xuất hiện trong chế độ Evaluation và không
  bao giờ được gửi vào retrieval.

## 3. Phạm vi chức năng

### 3.1 Legacy Search Lab

- Chọn `TemporalSearcher` hoặc `AmbiguousSearcher`.
- Nhập danh sách event query có thể reorder bằng nút lên/xuống.
- Chỉnh `top_k`, `gamma`, object filter và các tham số legacy hợp lệ.
- Gửi `/temporal-search`, xem tuple theo video, score và frame index.
- Hiển thị request/response JSON để regression/debug.
- Không dùng session, region, proposal hoặc adaptive normalized score.

### 3.2 Adaptive Session Lab

- Tạo, mở, refresh và xóa search session.
- Soạn event gồm `original_query`, `anchor_query`, `pre_state`, `post_state`,
  `boundary_type`.
- Ingest sparse candidates từ JSON/CSV hoặc từ adapter upstream.
- Xem video priority, temporal regions và refinement frontier.
- Lọc/allow/reject video.
- Chọn region để xem timeline, keyframe và score curve.
- Pick/fix keyframe hoặc timestamp cho từng event.
- Reject proposal/region, clear event constraint và recompute tuple.
- Ingest precomputed frame scores; chạy dense live chỉ khi provider capability
  cho phép.
- Xem ordered tuples, adjacent gaps, gap penalties và score decomposition.
- Export session snapshot, artifacts, run metrics và reproducibility manifest.

### 3.3 YouCook2 Retrieval Evaluation

MVP chỉ đo corpus-level Video Recall@K:

- chọn query set/split;
- không đưa `video_path` ground truth vào retrieval/filter;
- tìm keyframe toàn corpus, deduplicate thành ranked unique videos;
- hỗ trợ aggregation `max`, mean top-M, LogSumExp hoặc RRF;
- báo Recall@1/5/10/20/50, MRR, median rank và missing-index count;
- mở một query để xem top videos, top evidence frames và vị trí ground truth;
- export per-query ranking và summary CSV/JSON.

Dense localization, tIoU và pre/post benchmark được để sau một feature flag;
không trộn chúng vào Video Recall@K MVP.

## 4. Kiến trúc UI

```text
Streamlit browser
       |
       v
ui/api_client.py --------------> FastAPI :8001
       |                           /v1/searchers
       |                           /temporal-search
       |                           /v1/search-sessions/*
       |
       +------------------------> Frame/Search API :8000
       |                           /health, /search, frame adapter
       |
       +------------------------> local media resolver
                                   YouCook2 video/keyframe paths
```

Streamlit chỉ gọi API qua một typed client. Page/component không gọi `requests`
trực tiếp để thống nhất timeout, error mapping, logging và revision handling.

### Cấu trúc file dự kiến

```text
streamlit_ui/
  Home.py
  pages/
    01_Legacy_Search.py
    02_Adaptive_Session.py
    03_Region_Inspector.py
    04_Tuple_Explorer.py
    05_YouCook2_Evaluation.py
    06_Run_Comparison.py
  components/
    api_status.py
    event_editor.py
    hyperparameter_editor.py
    video_filter.py
    region_timeline.py
    keyframe_picker.py
    score_curves.py
    proposal_table.py
    tuple_viewer.py
    json_inspector.py
  services/
    api_client.py
    media_resolver.py
    benchmark_client.py
    export_service.py
  state/
    keys.py
    session_state.py
  models/
    ui_models.py
  tests/
    test_api_client.py
    test_state_transitions.py
    test_benchmark_metrics.py
    test_smoke_pages.py
```

## 5. Bố cục và luồng tương tác

### 5.1 Sidebar dùng chung

- Backend URL, upstream URL và nút health check.
- Capability badge cho legacy, adaptive core, frame provider và model runtime.
- Chọn workspace/dataset và media root.
- Session ID hiện tại, revision, last run ID và trạng thái dirty.
- Preset: `balanced`, `quality`, `paper_full`, `custom`.
- Nút export diagnostic bundle.

Không cho phép nhập token/secret rồi ghi vào log hoặc export bundle.

### 5.2 Adaptive Session page

Luồng chính là một stepper:

```text
1 Events -> 2 Candidates -> 3 Regions -> 4 Frame scores
         -> 5 Proposals -> 6 Tuples -> 7 Feedback/recompute
```

Mỗi bước hiển thị:

- trạng thái `not_started/running/ready/stale/error/unavailable`;
- artifact count;
- fingerprint/run ID;
- dependency bị invalidated sau lần sửa gần nhất;
- elapsed time và frame/refinement budget nếu backend trả về.

### 5.3 Event editor

Mỗi event là một expandable card:

- stable `event_id` chỉ đọc;
- original/anchor/pre/post text area;
- boundary type selectbox;
- completeness warning khi transition thiếu pre/post;
- duplicate, reorder và delete ở draft trước khi tạo session;
- sau khi đã tạo session, patch từng event qua API và hiển thị artifacts bị
  invalidated.

### 5.4 Hyperparameter tuning

Chia form thành năm tab khớp schema backend:

| Tab | Controls chính |
|---|---|
| Retrieval | top-N/variant, fused top-N, RRF k, variants/event |
| Clustering | gap, margin, max region seconds |
| Refinement | video/region budgets, exploration ratio, medium/dense stride, radius, max frames |
| Boundary | window options, min samples, score weights, NMS radius, proposal cap |
| Ranking | top-K, proposal/combination/tuple caps, gap tau/lambda |

Yêu cầu UX:

- slider chỉ dùng cho miền nhỏ; số chính xác dùng `number_input`;
- hiển thị unit ngay cạnh field;
- validation client-side giống bounds backend;
- **Reset preset**, **Diff from session**, **Apply & recompute**;
- preview invalidation scope trước khi apply;
- lưu một run label/note cho comparison;
- không tự gửi PATCH mỗi lần slider rerun.

### 5.5 Video browser và filter

Hai cột:

- trái: ranked video table gồm coverage, priority, best region/event, status;
- phải: video preview, event coverage matrix và region timeline.

Tương tác:

- search theo video ID;
- sort theo priority/coverage/event score;
- allowlist selected videos;
- reject/unreject video;
- pin video để giữ khi đổi filter;
- compare tối đa 3 video;
- badge phân biệt user constraint với automatic frontier.

Filter chỉ thay view khi chọn **View only**; constraint chỉ gửi backend khi
chọn rõ **Apply as search constraint**.

### 5.6 Temporal region inspector

Timeline là visual chính:

```text
video time ---------------------------------------------------------->
E1       [region A]       [region B]
E2                 [region C]
proposal      ^  ^               ^
fixed frame      *
```

Chức năng:

- zoom/pan và nhập start/end giây;
- overlay nhiều event bằng màu ổn định;
- bật/tắt sparse candidates, regions, sampled frames, proposals, ground truth;
- click region để chọn, reject hoặc force refinement;
- xem start/end/duration/coarse score/status/candidate IDs;
- tải bảng region CSV;
- ground truth overlay chỉ khả dụng trong Evaluation mode và có cảnh báo
  leakage rõ ràng.

Ưu tiên Plotly cho timeline và curve vì cần hover/click/zoom. Nếu event click
không ổn định qua Streamlit rerun, dùng bảng selection làm interaction chính và
chart làm visualization phụ.

### 5.7 Keyframe picker và score curves

Khi chọn region:

- filmstrip keyframe theo timestamp;
- video player seek gần timestamp được chọn;
- pagination/lazy thumbnail để không nạp hàng nghìn ảnh;
- chọn frame rồi **Fix for event**;
- reject proposal hoặc region;
- nhập exact timestamp khi frame chưa có thumbnail;
- hiển thị source: sparse, medium, dense hoặc user-fixed.

Curve chart gồm:

- raw anchor/pre/post/motion;
- normalized anchor/pre/post/motion;
- final semantic/boundary score;
- proposal marker, left/right window và NMS suppression;
- checkbox tách raw/normalized để tránh hai scale chồng nhau.

Frame card phải có `video_id`, `event_id`, `region_id`, `frame_id`, exact
`timestamp_seconds`, display timestamp, scores và model/runtime fingerprint.

### 5.8 Proposal và tuple explorer

Proposal table:

- filter theo event/video/region/status;
- sort final/semantic/boundary/pre/post;
- chọn proposal để seek video;
- fix hoặc reject với confirmation;
- xem score decomposition.

Tuple view:

- một row/card cho mỗi ordered tuple;
- filmstrip theo thứ tự event;
- timestamps và adjacent gaps;
- raw event mean, gap penalty, constraint bonus, raw/normalized final score;
- cảnh báo nếu tuple thiếu preview media;
- export selected tuple để dùng cho demo hoặc annotation.

## 6. Streamlit state và revision model

Chỉ lưu dữ liệu cần cho UI trong `st.session_state`:

```text
backend_url
upstream_url
active_session_id
session_revision
selected_video_id
selected_event_id
selected_region_id
selected_proposal_id
draft_hyperparameters
applied_hyperparameters
view_filters
capabilities
last_error
```

Không lưu toàn bộ artifact lớn nếu có thể tải theo page/limit. Cache:

- `st.cache_resource`: API client và media resolver;
- `st.cache_data`: read-only page responses với key gồm session revision;
- không cache mutation;
- clear artifact cache khi revision thay đổi.

Mutation flow:

```text
read current revision
  -> submit mutation(expected_revision)
  -> success: store new revision, clear dependent cache
  -> conflict: refresh session, show diff, do not auto-retry mutation
```

## 7. API mapping

| UI action | Backend call |
|---|---|
| Discover capability | `GET /v1/searchers` |
| Legacy run | `POST /temporal-search` |
| Create adaptive session | `POST /v1/search-sessions` |
| Refresh session | `GET /v1/search-sessions/{id}` |
| Edit event | `PATCH /v1/search-sessions/{id}/events/{event_id}` |
| Apply hyperparameters | `PATCH /v1/search-sessions/{id}/hyperparameters` |
| Apply video/event constraints | `PUT /v1/search-sessions/{id}/constraints` |
| Ingest sparse candidates | `POST .../artifacts/candidates` |
| Ingest frame scores | `POST .../artifacts/frame-scores` |
| List regions | `GET .../regions` |
| List proposals | `GET .../proposals` |
| List tuples | `GET .../tuples` |
| Fix frame | adaptive fix-frame endpoint |
| Reject proposal | adaptive reject-proposal endpoint |
| Delete session | `DELETE /v1/search-sessions/{id}` |

Client phải đặt connect/read timeout, parse FastAPI validation error và hiển thị
request ID nếu backend bổ sung sau này. Không log raw prompt/model output nếu có
thể chứa dữ liệu nhạy cảm.

## 8. YouCook2 evaluation page

### Input

- query directory/manifest;
- corpus/index name;
- split và số query giới hạn;
- model/language variant;
- frame top-N;
- video aggregation method;
- K list;
- optional run name/seed/note.

### Guard chống leakage

- parser tách `video_path` thành `ground_truth_video_id` chỉ trong evaluator;
- payload `/search` chỉ chứa event text và top-N;
- UI hiển thị payload đã sanitize;
- unit test thất bại nếu `video_path`, answer interval hoặc hint path xuất hiện
  trong retrieval payload.

### Output

- metric cards Recall@K, MRR, median rank;
- recall curve theo K;
- query table: ground-truth rank, hit@K, missing video/index, latency;
- detail drawer: ranked unique videos và evidence keyframes;
- failure buckets: missing corpus item, translation failure, low-score, duplicate
  domination, wrong recipe/visually similar action;
- export `run_manifest.json`, `per_query.csv`, `summary.json`.

Không dùng kết quả từ script localization hiện tại làm Recall@K. Corpus
retrieval phải search toàn index và deduplicate frame hits theo video.

## 9. Error, loading và empty states

- Backend down: hiện URL, lỗi kết nối và nút Retry; không render traceback thô.
- Upstream down: legacy/retrieval disabled, adaptive artifact inspection vẫn mở.
- Live provider unavailable: nút dense live disabled, giải thích và hiển thị
  upload frame-score JSON.
- Empty regions/proposals/tuples: chỉ ra stage trước đó cần chạy hoặc constraint
  nào có thể đã loại hết kết quả.
- Stale artifact: badge màu vàng, dependency/invalidation reason.
- Long run: progress/status polling; không khóa toàn page. Khi backend chưa có
  async job API, mô tả giới hạn thay vì fake progress.
- Media missing: vẫn cho xem metadata/scores và đánh dấu preview unavailable.

## 10. Roadmap triển khai

### Phase UI-0 — Contract và skeleton

- Pin Streamlit/Plotly/http client versions.
- Tạo typed API client, configuration và health/capability panel.
- Thêm `.env.example` không chứa secret.
- Mock backend fixtures từ API schemas.

Acceptance:

- app khởi động khi backend down;
- capability hiển thị đúng;
- client test được success, validation error, timeout và revision conflict.

### Phase UI-1 — Legacy và adaptive session foundation

- Legacy Search Lab.
- Create/load/delete session và event editor.
- Session revision/state model.
- JSON inspector và diagnostic export.

Acceptance:

- legacy chạy không tạo session;
- adaptive mutation luôn gửi expected revision;
- refresh browser không âm thầm tạo session mới.

### Phase UI-2 — Hyperparameters, candidates và regions

- Hyperparameter editor theo nested schema.
- Candidate ingest/upload.
- Video ranking/filter/constraints.
- Region table và Plotly timeline.

Acceptance:

- diff và invalidation preview đúng;
- view filter không thay backend constraint;
- exact timestamp giữ nguyên khi round-trip.

### Phase UI-3 — Frame inspector và feedback loop

- Media resolver/video preview.
- Filmstrip, score curves, proposal table.
- Fix frame/timestamp, reject proposal/region, clear constraint.
- Capability-gated live refinement và precomputed score ingest.

Acceptance:

- click/fix frame tạo đúng constraint cho đúng event;
- stale downstream artifact được đánh dấu sau mutation;
- UI vẫn hoạt động đầy đủ ở metadata mode khi thiếu media.

### Phase UI-4 — Tuple explorer và run comparison

- Ordered tuple cards/timeline.
- Score/gap decomposition.
- So sánh hai hoặc nhiều run/hyperparameter preset.
- Export reproducibility bundle.

Acceptance:

- không so trực tiếp legacy score với adaptive score trên cùng axis;
- comparison ghi đủ model revision, preprocessing và parameters;
- selected tuple export có đủ event/frame/time lineage.

### Phase UI-5 — YouCook2 Recall@K

- Query manifest parser và leakage guard.
- Batch corpus retrieval, video aggregation và metrics.
- Failure browser và run export.
- Resume/checkpoint để không mất batch khi Streamlit rerun.

Acceptance:

- ground-truth video không có trong request/filter;
- frame hits được deduplicate trước Recall@K;
- metric unit tests dùng fixture có rank biết trước;
- training và validation run không bị gộp.

### Phase UI-6 — Dense evaluation sau này

- Nối `FrameProvider`/worker live.
- Ground-truth temporal overlay có feature flag.
- tIoU, start/end error, proposal Recall@K.
- DiDeMo adapter và temporal-language benchmark.

Phase này chỉ bắt đầu sau khi YouCook2 Video Recall@K ổn định.

## 11. Testing strategy

### Unit

- API serialization/deserialization;
- hyperparameter validation/diff;
- timestamp conversion;
- video dedup/aggregation và Recall@K;
- leakage guard;
- revision/state transition.

### Contract/API integration

- chạy FastAPI TestClient/real local server với fixture session;
- mọi UI mutation khớp strict Pydantic schema;
- pagination, empty result và stale revision;
- capability false/true fixtures.

### UI smoke

- import/render từng page;
- widget keys không trùng;
- draft form không auto-submit khi rerun;
- selected video/region tồn tại qua rerun hợp lệ;
- screenshot checklist ở desktop width; mobile không phải acceptance gate.

### End-to-end scenarios

1. Legacy ordered search hoàn chỉnh.
2. Adaptive: create -> candidates -> regions -> frame scores -> proposals -> tuples.
3. Chỉnh clustering gap và quan sát invalidation/recompute.
4. Reject video, fix keyframe, clear constraint.
5. Backend/frame provider unavailable.
6. YouCook2 batch nhỏ có Recall@K biết trước.

## 12. Reproducibility và telemetry

Mỗi exported run bundle gồm:

- UI/backend code version;
- dataset/index fingerprint;
- session/run IDs và revision;
- complete hyperparameters;
- event definitions và constraints;
- model ID, immutable revision, dimension, preprocess/instruction;
- artifact counts, stage latency và budget usage;
- metric summary và per-query results;
- timestamp và optional researcher note.

Telemetry mặc định local/off. Nếu thêm analytics sau này, không gửi query, raw
model output, local path hoặc thumbnail ra dịch vụ ngoài nếu chưa có opt-in.

## 13. Definition of Done

Streamlit UI được coi là hoàn tất cho research MVP khi:

1. legacy và adaptive được gọi đúng hai contract khác nhau;
2. người dùng chỉnh được toàn bộ nested hyperparameters có validation;
3. xem/lọc/constraint video và xem temporal region tương tác được;
4. xem keyframe/curve, fix frame và reject proposal được;
5. xem ordered tuple và score decomposition được;
6. revision conflict và artifact invalidation được xử lý minh bạch;
7. chạy được cả khi live frame provider chưa có bằng precomputed artifacts;
8. YouCook2 Video Recall@K có leakage guard, dedup và export tái lập;
9. unit, contract và E2E scenarios chính đều pass;
10. tài liệu có lệnh chạy, cấu hình, limitation và demo script.

