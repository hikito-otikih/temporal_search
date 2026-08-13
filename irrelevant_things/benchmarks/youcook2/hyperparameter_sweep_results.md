# adaptive_coarse / adaptive_full hyperparameter sweep results

Date: 2026-08-12
Backend: `src/main.py` (port 8001), real YouCook2 videos + real upstream sparse search service.

## Goal

1. Find the best-performing `adaptive_coarse` hyperparameter configuration by sweeping
   `RetrievalHyperparameters` (`top_n_per_variant`, `top_n_fused`, `rrf_k`) and the
   `prioritize_videos()` weights (`video_coverage_weight`, `video_mean_weight`,
   `video_min_weight`).
2. Using that winning config, find the *minimum* `adaptive_full` frontier configuration
   (`max_initial_videos`, `max_total_regions`, `max_regions_per_event_per_video`,
   `max_frames_per_run`) whose results are equal to or better than `adaptive_coarse`'s,
   and characterize how those knobs trade off recall against runtime.

## Table 1 β€” `adaptive_coarse` hyperparameter sweep (n=50 real queries)

Methodology: for each retrieval config, one real `create session -> retrieve -> GET /regions`
round trip per query (needs actual re-retrieval). All 7 weight configs for that retrieval
setting are then evaluated for free by calling `prioritize_videos()` directly in Python on the
already-fetched regions (weights only affect post-retrieval scoring, no extra network calls
needed). `query_variants_per_event` was excluded from the sweep: this benchmark never invokes
the query-rewrite step, so every event always submits exactly one variant regardless of that
setting's cap.

| retrieval (top_n_per_variant / top_n_fused / rrf_k) | weights (coverage / mean / min) | r@1 | r@5 | r@10 | r@20 | r@50 | r@100 | MRR |
|---|---|---|---|---|---|---|---|---|
| orig_default (50/100/60) | current (.5/.3/.2) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 |
| orig_default (50/100/60) | coverage_heavy (.8/.1/.1) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 |
| orig_default (50/100/60) | mean_heavy (.1/.8/.1) | 0.26 | 0.46 | 0.58 | 0.70 | 0.86 | 0.86 | 0.381 |
| orig_default (50/100/60) | min_heavy (.1/.1/.8) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 |
| orig_default (50/100/60) | balanced (1/1/1) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 |
| orig_default (50/100/60) | **coverage_only (1/0/0)** | **0.32** | 0.50 | 0.56 | 0.70 | 0.86 | 0.86 | 0.420 |
| orig_default (50/100/60) | mean_only (0/1/0) | 0.22 | 0.46 | 0.58 | 0.70 | 0.86 | 0.86 | 0.365 |
| current (200/500/60) | current (.5/.3/.2) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 |
| current (200/500/60) | coverage_heavy (.8/.1/.1) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 |
| current (200/500/60) | mean_heavy (.1/.8/.1) | 0.30 | 0.54 | 0.60 | 0.76 | 0.82 | 0.86 | 0.400 |
| current (200/500/60) | min_heavy (.1/.1/.8) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 |
| current (200/500/60) | balanced (1/1/1) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 |
| current (200/500/60) | **coverage_only (1/0/0)** | **0.32** | 0.48 | 0.66 | 0.76 | 0.82 | **0.90** | 0.412 |
| current (200/500/60) | mean_only (0/1/0) | 0.30 | 0.54 | 0.58 | 0.76 | 0.82 | 0.86 | 0.402 |
| wide (500/1000/60) | current (.5/.3/.2) | 0.36 | 0.50 | 0.60 | 0.72 | 0.80 | 0.86 | 0.433 |
| wide (500/1000/60) | coverage_heavy (.8/.1/.1) | 0.36 | 0.50 | 0.60 | 0.72 | 0.80 | 0.86 | 0.433 |
| wide (500/1000/60) | mean_heavy (.1/.8/.1) | 0.36 | 0.52 | 0.64 | 0.76 | 0.80 | 0.86 | 0.447 |
| wide (500/1000/60) | min_heavy (.1/.1/.8) | 0.36 | 0.50 | 0.62 | 0.72 | 0.80 | 0.86 | 0.433 |
| wide (500/1000/60) | balanced (1/1/1) | 0.36 | 0.50 | 0.62 | 0.72 | 0.80 | 0.86 | 0.433 |
| wide (500/1000/60) | coverage_only (1/0/0) | **0.18** | 0.44 | 0.56 | 0.66 | 0.80 | 0.86 | **0.314** |
| **wide (500/1000/60)** | **mean_only (0/1/0)** | **0.38** | **0.52** | **0.64** | **0.76** | **0.82** | 0.86 | **0.461** |
| sharp_rrf (200/500/10) | current (.5/.3/.2) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.372 |
| sharp_rrf (200/500/10) | coverage_heavy (.8/.1/.1) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.372 |
| sharp_rrf (200/500/10) | mean_heavy (.1/.8/.1) | 0.26 | 0.54 | 0.62 | 0.76 | 0.82 | 0.86 | 0.380 |
| sharp_rrf (200/500/10) | min_heavy (.1/.1/.8) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.373 |
| sharp_rrf (200/500/10) | balanced (1/1/1) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.372 |
| sharp_rrf (200/500/10) | coverage_only (1/0/0) | 0.32 | 0.48 | 0.66 | 0.76 | 0.82 | **0.90** | 0.412 |
| sharp_rrf (200/500/10) | mean_only (0/1/0) | 0.26 | 0.52 | 0.62 | 0.76 | 0.82 | 0.86 | 0.381 |
| flat_rrf (200/500/200) | current (.5/.3/.2) | 0.26 | 0.48 | 0.62 | 0.74 | 0.80 | 0.88 | 0.375 |
| flat_rrf (200/500/200) | coverage_heavy (.8/.1/.1) | 0.26 | 0.48 | 0.62 | 0.74 | 0.80 | 0.88 | 0.375 |
| flat_rrf (200/500/200) | mean_heavy (.1/.8/.1) | 0.28 | 0.56 | 0.62 | 0.76 | 0.80 | 0.86 | 0.391 |
| flat_rrf (200/500/200) | min_heavy (.1/.1/.8) | 0.26 | 0.50 | 0.62 | 0.74 | 0.80 | 0.88 | 0.376 |
| flat_rrf (200/500/200) | balanced (1/1/1) | 0.26 | 0.48 | 0.62 | 0.74 | 0.80 | 0.88 | 0.375 |
| flat_rrf (200/500/200) | coverage_only (1/0/0) | 0.32 | 0.48 | 0.66 | 0.76 | 0.82 | **0.90** | 0.412 |
| flat_rrf (200/500/200) | mean_only (0/1/0) | 0.28 | 0.56 | 0.58 | 0.76 | 0.82 | 0.86 | 0.395 |

**Winner: `top_n_per_variant=500, top_n_fused=1000, rrf_k=60` + weights `(coverage=0, mean=1.0, min=0)`**
β€” r@1=0.380, r@5=0.520, r@10=0.640, r@20=0.760, r@50=0.820, r@100=0.860, MRR=0.461.
Compare to the previously-live config (`current` retrieval + `current` weights): r@1=0.260, MRR=0.373
β€” a ~46% relative recall@1 improvement from re-ranking already-available signal, no new data.

### Findings

1. **`rrf_k` doesn't matter.** 10 vs 60 vs 200 (`sharp_rrf`/`current`/`flat_rrf`, all at
   200/500 fan-out) are statistically indistinguishable at every K, for every weight config.
   Consistent with most (event,video) pairs having only 1-2 real candidates β€” there's rarely
   enough rank depth for `rrf_k` to smooth over.
2. **Mean score alone beats every blend that includes coverage or min**, once retrieval is
   wide. Coverage stops being discriminative when most surviving videos already hit every
   event at least weakly; it can no longer separate a good match from a mediocre one.
3. **`coverage_only` + `wide` retrieval is the single worst combination in the whole grid**
   (r@1=0.18, MRR=0.314) β€” worse than every narrow-retrieval config, despite `wide` winning
   with the right weights. Widening retrieval creates more coverage-tied videos; ranking
   purely by coverage can't break those ties by quality. Retrieval breadth and ranking
   weights are not independently tunable.
4. For **narrow** retrieval (`orig_default`, `sharp_rrf`, `flat_rrf`), `coverage_only` is
   consistently the *best* weight choice instead β€” the opposite of finding 2. The right
   weighting is conditional on retrieval width.
5. **r@100 saturates around 0.86-0.90 for every config, largely independent of weighting** β€”
   unlike r@1..r@50, where weight choice swings results by up to 20 points, every weight config
   within a retrieval block converges to nearly the same r@100 (all 7 weight configs land within
   0.86-0.90 of each other at every retrieval width). Weighting mostly reorders *where* the
   correct video lands, not *whether* it's reachable at all - that ceiling is set by retrieval,
   not ranking. Oddly, `wide` retrieval's r@100 ceiling (0.860 for every weight config, even
   `coverage_only`) is *lower* than `current`/`sharp_rrf`/`flat_rrf`'s `coverage_only` row (0.900,
   the single highest value in the whole grid) despite `wide` fetching more fused candidates -
   evidence that `top_n_fused` isn't simply "more reach," it's competing for the same fixed
   `UPSTREAM_TOP_K=500` raw pool differently, not strictly supersetting the narrower configs.

## Table 2 β€” `adaptive_full` frontier-width ladder (n=12, same queries throughout) vs `adaptive_coarse` baseline

Baseline and all four `adaptive_full` configs use Table 1's winning retrieval/weight config
(`top_n_per_variant=500, top_n_fused=1000, rrf_k=60`, weights `coverage=0, mean=1.0, min=0`),
so the only variable below is the `adaptive_full`-specific frontier sizing.

| config | max_initial_videos / max_total_regions | r@1 | r@5 | r@10 | r@20 | r@50 | MRR | found/12 | avg latency/query |
|---|---|---|---|---|---|---|---|---|---|
| **adaptive_coarse baseline** | β€” | **0.417** | **0.417** | **0.583** | **0.667** | **0.833** | **0.446** | 12/12 | 1.4s |
| `old_defaults` (original pre-tuning defaults) | 5 / 60 | 0.417 | 0.417 | 0.417 | 0.417 | 0.417 | 0.417 | 5/12 | 18.4s |
| `small` | 10 / 150 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 4/12 | 40.6s |
| `medium` | 20 / 300 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 4/12 | 74.0s |
| `large` | 40 / 600 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 | 4/12 | 135.3s |

(`max_regions_per_event_per_video=3` for `old_defaults`, `=5` for the other three;
`max_frames_per_run` set generously above the `_natural_budgets()` (2K+1) demand in every
config so it was never the binding constraint.)

### Findings

1. **No config in this ladder matches or beats `adaptive_coarse`, at any K.** `old_defaults`
   (the smallest, cheapest config) comes closest β€” it ties coarse's r@1 but loses at every
   larger K, and every row is flat across K because `unique_video_count_max` never exceeds 2:
   there is essentially never more than 1-2 candidates in the ranked list at all, so K beyond
   that is moot.
2. **Widening the frontier past `old_defaults` bought zero recall and only added latency.**
   `small` -> `medium` -> `large` is a 4x/10x range on `max_initial_videos`/`max_total_regions`
   and a 7.3x range on latency (18.4s -> 135.3s), with identical recall (4/12 found) at every
   step. The frontier-selection knobs are conclusively saturated in this experiment.
3. The likely real bottleneck is retrieval density, not frontier width: most (event,video)
   pairs only ever get 1-2 real candidates (see the companion region-compression finding from
   the same session), and `assemble_ordered_tuples()` requires *complete* per-event coverage
   for a video to produce any tuple at all. Widening the frontier just considers more videos
   that still lack complete coverage; it doesn't fix the underlying sparsity.

## Table 3 β€” `adaptive_coarse` sweep with `cluster_temporal_regions()` skipped (n=50, same queries as Table 1)

Same 5 retrieval configs Γ— 7 weight configs Γ— 50 queries as Table 1, so every row is directly
comparable. The only change: after `fuse_candidates_rrf()`, each fused candidate becomes its
own singleton region (`K=1`, zero-width `start_seconds == end_seconds`) instead of being passed
through `cluster_temporal_regions()` β€” isolating exactly what the clustering step itself buys.
Retrieval was re-run for real (`UpstreamSearchClient.retrieve_candidates()` + `fuse_candidates_rrf()`
called directly in Python, mirroring `service.py`'s `replace_candidates()` up to but not including
clustering); region scores reuse the same `_candidate_normalized_scores_by_event()` calibration
`cluster_temporal_regions()` itself uses, so scores stay numerically consistent with Table 1
wherever a region is naturally a singleton anyway. Ξ” columns are Table 1 βˆ’ Table 3 (positive =
clustering helped).

| retrieval (top_n_per_variant / top_n_fused / rrf_k) | weights (coverage / mean / min) | r@1 | r@5 | r@10 | r@20 | r@50 | r@100 | MRR | Ξ”r@1 | Ξ”MRR | Ξ”r@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orig_default (50/100/60) | current (.5/.3/.2) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 | 0.00 | 0.000 | 0.00 |
| orig_default (50/100/60) | coverage_heavy (.8/.1/.1) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 | 0.00 | 0.000 | 0.00 |
| orig_default (50/100/60) | mean_heavy (.1/.8/.1) | 0.26 | 0.46 | 0.58 | 0.70 | 0.86 | 0.86 | 0.381 | 0.00 | 0.000 | 0.00 |
| orig_default (50/100/60) | min_heavy (.1/.1/.8) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 | 0.00 | 0.000 | 0.00 |
| orig_default (50/100/60) | balanced (1/1/1) | 0.24 | 0.46 | 0.56 | 0.68 | 0.86 | 0.86 | 0.370 | 0.00 | 0.000 | 0.00 |
| orig_default (50/100/60) | **coverage_only (1/0/0)** | **0.32** | 0.50 | 0.56 | 0.70 | 0.86 | 0.86 | 0.420 | 0.00 | 0.000 | 0.00 |
| orig_default (50/100/60) | mean_only (0/1/0) | 0.22 | 0.46 | 0.58 | 0.70 | 0.86 | 0.86 | 0.365 | 0.00 | 0.000 | 0.00 |
| current (200/500/60) | current (.5/.3/.2) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 | 0.00 | 0.000 | 0.00 |
| current (200/500/60) | coverage_heavy (.8/.1/.1) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 | 0.00 | 0.000 | 0.00 |
| current (200/500/60) | mean_heavy (.1/.8/.1) | 0.30 | 0.54 | 0.60 | 0.76 | 0.82 | 0.86 | 0.400 | 0.00 | 0.000 | 0.00 |
| current (200/500/60) | min_heavy (.1/.1/.8) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 | 0.00 | 0.000 | 0.00 |
| current (200/500/60) | balanced (1/1/1) | 0.26 | 0.46 | 0.62 | 0.74 | 0.80 | 0.88 | 0.373 | 0.00 | 0.000 | 0.00 |
| current (200/500/60) | **coverage_only (1/0/0)** | **0.32** | 0.48 | 0.66 | 0.76 | 0.82 | 0.90 | 0.412 | 0.00 | 0.000 | 0.00 |
| current (200/500/60) | mean_only (0/1/0) | 0.30 | 0.54 | 0.58 | 0.76 | 0.82 | 0.86 | 0.402 | 0.00 | 0.000 | 0.00 |
| wide (500/1000/60) | current (.5/.3/.2) | 0.28 | 0.40 | 0.56 | 0.62 | 0.78 | 0.86 | 0.366 | +0.08 | +0.067 | 0.00 |
| wide (500/1000/60) | coverage_heavy (.8/.1/.1) | 0.28 | 0.40 | 0.56 | 0.62 | 0.78 | 0.86 | 0.369 | +0.08 | +0.064 | 0.00 |
| wide (500/1000/60) | mean_heavy (.1/.8/.1) | 0.28 | 0.40 | 0.62 | 0.68 | 0.80 | 0.86 | 0.376 | +0.08 | +0.071 | 0.00 |
| wide (500/1000/60) | min_heavy (.1/.1/.8) | 0.28 | 0.42 | 0.52 | 0.64 | 0.78 | 0.86 | 0.369 | +0.08 | +0.064 | 0.00 |
| wide (500/1000/60) | balanced (1/1/1) | 0.28 | 0.40 | 0.56 | 0.64 | 0.78 | 0.86 | 0.369 | +0.08 | +0.064 | 0.00 |
| wide (500/1000/60) | coverage_only (1/0/0) | 0.20 | 0.44 | 0.50 | 0.62 | 0.80 | 0.88 | 0.319 | **-0.02** | **-0.005** | **-0.02** |
| **wide (500/1000/60)** | **mean_only (0/1/0)** | **0.32** | 0.44 | 0.62 | 0.68 | 0.80 | 0.86 | 0.401 | +0.06 | +0.060 | 0.00 |
| sharp_rrf (200/500/10) | current (.5/.3/.2) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.372 | 0.00 | 0.000 | 0.00 |
| sharp_rrf (200/500/10) | coverage_heavy (.8/.1/.1) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.372 | 0.00 | 0.000 | 0.00 |
| sharp_rrf (200/500/10) | mean_heavy (.1/.8/.1) | 0.26 | 0.54 | 0.62 | 0.76 | 0.82 | 0.86 | 0.380 | 0.00 | 0.000 | 0.00 |
| sharp_rrf (200/500/10) | min_heavy (.1/.1/.8) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.373 | 0.00 | 0.000 | 0.00 |
| sharp_rrf (200/500/10) | balanced (1/1/1) | 0.26 | 0.46 | 0.64 | 0.74 | 0.80 | 0.88 | 0.372 | 0.00 | 0.000 | 0.00 |
| sharp_rrf (200/500/10) | **coverage_only (1/0/0)** | **0.32** | 0.48 | 0.66 | 0.76 | 0.82 | 0.90 | 0.412 | 0.00 | 0.000 | 0.00 |
| sharp_rrf (200/500/10) | mean_only (0/1/0) | 0.26 | 0.52 | 0.62 | 0.76 | 0.82 | 0.86 | 0.381 | 0.00 | 0.000 | 0.00 |
| flat_rrf (200/500/200) | current (.5/.3/.2) | 0.26 | 0.48 | 0.62 | 0.74 | 0.80 | 0.88 | 0.375 | 0.00 | 0.000 | 0.00 |
| flat_rrf (200/500/200) | coverage_heavy (.8/.1/.1) | 0.26 | 0.48 | 0.62 | 0.74 | 0.80 | 0.88 | 0.375 | 0.00 | 0.000 | 0.00 |
| flat_rrf (200/500/200) | mean_heavy (.1/.8/.1) | 0.28 | 0.56 | 0.62 | 0.76 | 0.80 | 0.86 | 0.391 | 0.00 | 0.000 | 0.00 |
| flat_rrf (200/500/200) | min_heavy (.1/.1/.8) | 0.26 | 0.50 | 0.62 | 0.74 | 0.80 | 0.88 | 0.376 | 0.00 | 0.000 | 0.00 |
| flat_rrf (200/500/200) | balanced (1/1/1) | 0.26 | 0.48 | 0.62 | 0.74 | 0.80 | 0.88 | 0.375 | 0.00 | 0.000 | 0.00 |
| flat_rrf (200/500/200) | **coverage_only (1/0/0)** | **0.32** | 0.48 | 0.66 | 0.76 | 0.82 | 0.90 | 0.412 | 0.00 | 0.000 | 0.00 |
| flat_rrf (200/500/200) | mean_only (0/1/0) | 0.28 | 0.56 | 0.58 | 0.76 | 0.82 | 0.86 | 0.395 | 0.00 | 0.000 | 0.00 |

**Best no-clustering config: `orig_default (50/100/60)` + `coverage_only (1/0/0)`**
β€” r@1=0.32, r@5=0.50, r@10=0.56, r@20=0.70, r@50=0.86, r@100=0.86, MRR=0.420 (tied on r@1 with
three other narrow-retrieval + `coverage_only` rows and with `wide`+`mean_only`, but strictly best
on MRR). Table 1's actual winner (`wide`+`mean_only`) only reaches r@1=0.32/MRR=0.401 without
clustering, down from r@1=0.38/MRR=0.461 with it.

### Findings

1. **Clustering has zero measurable effect at every retrieval width except `wide`.** All 28 rows
   under `orig_default`, `current`, `sharp_rrf`, and `flat_rrf` (i.e. every config with
   `top_n_fused` ≀ 500) are identical between Table 1 and Table 3, to the row. This is the direct
   mechanical confirmation of Table 1/2's "most (event,video) pairs have only 1-2 real candidates"
   observation: with that few candidates, `cluster_temporal_regions()` almost never has more than
   one candidate within reach to merge, so each region is already a natural singleton and skipping
   the clustering step changes nothing.
2. **At `wide` retrieval β€” the one place clustering has candidates to actually merge β€” it's a
   consistent net positive.** r@1 improves by +0.06 to +0.08 and MRR by +0.06 to +0.07 for every
   weight config except `coverage_only`. Merging temporally-close candidates into a stronger,
   better-scored region is genuinely more informative than ranking on the best singleton alone,
   but only once retrieval is deep enough to produce candidates worth merging.
3. **`coverage_only` is the one weight config where clustering slightly hurts** (Ξ”r@1=-0.02,
   Ξ”MRR=-0.005) at `wide` retrieval. Consistent with Table 1 finding #3 (`coverage_only`+`wide` is
   already the grid's worst combination): clustering collapses several weak per-event hits into one
   region that still counts as "covering" the event, which makes a coverage-only ranking even less
   able to discriminate a strong match from a mediocre one.
4. **Net effect on the pipeline's best achievable result:** clustering is entirely responsible for
   `wide`+`mean_only` being Table 1's winner. Remove it, and that same config drops from
   r@1=0.38/MRR=0.461 to r@1=0.32/MRR=0.401 (-16% relative r@1, -9% relative MRR), and the
   best-overall config reverts to a narrow-retrieval + `coverage_only` combination β€” the same
   width-dependent weighting pattern Table 1 finding #4 already identified. Clustering doesn't
   improve `adaptive_coarse` in general; it specifically is what makes wide retrieval + mean-score
   ranking pay off.
5. **Clustering's r@1-r@50 advantage at `wide` retrieval fully vanishes by r@100.** Every positive
   Ξ” (current/coverage_heavy/mean_heavy/min_heavy/balanced/mean_only) closes to exactly 0.00 at
   r@100 β€” both pipelines find the same 43/50 targets somewhere in the top 100 once the ranking
   window is wide enough. Clustering only changes *where* a correct video lands near the top of the
   ranking, consistent with Table 1 finding #5; it doesn't expand which videos are reachable at
   all. The one exception is `coverage_only`+`wide`, where clustering's small deficit persists all
   the way to r@100 (-0.02, i.e. one query's worth) rather than closing β€” plausibly because
   `coverage_only` produces heavy score ties (many videos with identical/near-identical coverage),
   and clustering changes which video wins those ties, not merely how quickly they're found.

## Reproducibility

Coarse baseline / any `adaptive_full` config, run via the extended CLI
(`irrelevant_things/benchmarks/youcook2/cli.py`, flags added in this session):

```bash
python -m benchmarks.youcook2 tuple-run \
  --query-dir /mnt/c/Users/huynh/Downloads/youcook2/query \
  --pipeline adaptive_coarse \
  --limit 12 \
  --adaptive-top-k 500 \
  --adaptive-top-n-per-variant 500 --adaptive-top-n-fused 1000 --adaptive-rrf-k 60 \
  --adaptive-video-coverage-weight 0 --adaptive-video-mean-weight 1 --adaptive-video-min-weight 0 \
  --adaptive-ranking-top-k 500 \
  --recall-k 1,5,10,20,50 \
  --output-dir <dir>
```

Table 3 (no clustering) is not CLI-driven β€” it bypasses this repo's HTTP backend entirely and
calls `UpstreamSearchClient.retrieve_candidates()` + `fuse_candidates_rrf()` directly in Python
(same real upstream service, same 50 queries as Table 1), building one singleton `TemporalRegion`
per fused candidate instead of calling `cluster_temporal_regions()`. Standalone script, not
checked into the repo:

```bash
python3 /tmp/coarse_no_cluster_sweep.py   # writes /tmp/coarse_no_cluster_results.json
```

```bash
python -m benchmarks.youcook2 tuple-run \
  --query-dir /mnt/c/Users/huynh/Downloads/youcook2/query \
  --pipeline adaptive_full --limit 12 --timeout 600 --retries 0 \
  --adaptive-top-k 500 --adaptive-top-n-per-variant 500 --adaptive-top-n-fused 1000 --adaptive-rrf-k 60 \
  --adaptive-video-coverage-weight 0 --adaptive-video-mean-weight 1 --adaptive-video-min-weight 0 \
  --recall-k 1,5,10,20,50 \
  --adaptive-max-initial-videos 40 --adaptive-max-total-regions 600 --adaptive-max-regions-per-event 5 \
  --adaptive-max-frames 20000 --adaptive-max-frames-per-run 20000 \
  --output-dir <dir>
```
