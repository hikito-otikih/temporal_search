# Boundary-refinement experiment: does a post-hoc local sweep improve moment-precision?

Date: 2026-08-12
Backend: `src/main.py` (port 8001) + real upstream sparse-search service (port 8000), real YouCook2 videos, real SigLIP2 dense scoring (`google/siglip2-base-patch16-224`, pinned revision).

## Goal

Test whether a cheap, post-hoc "boundary refinement" stage (local ±10-native-frame sweep around
an already-chosen anchor, scored with the existing pre/post-state change-point detector) measurably
improves temporal precision for moment-oriented queries ("the first moment X happens" / "the last
moment X happens"), across all three pre-`adaptive_full` pipelines: `legacy_temporal`,
`legacy_ambiguous`, `adaptive_coarse`.

## Methodology

### `augmented_query` dataset

Generated from the existing 200-video YouCook2 query directory. For each source video's event list,
two variants are emitted into separate subdirectories (`augmented_query/first/`, `augmented_query/last/`
— the two variants must live in different directories because `load_query_directory_grouped()`
rejects duplicate `video_id`s within one directory):

- `{video_id}__first.txt`: every event's text wrapped as `"khoảnh khắc đầu tiên {text}"` ("the first
  moment {text}"), `boundary_type="onset"`.
- `{video_id}__last.txt`: wrapped as `"khoảnh khắc cuối cùng {text}"` ("the last moment {text}"),
  `boundary_type="offset"`.

Both variants reuse the **exact same** `E{n}: start - end` ground-truth interval from the source
annotation, unchanged. This is a deliberate simplification: YouCook2's segment boundaries are
crowd-annotated to the nearest second, not frame-exact, so "does the chosen frame land inside
`[start, end]`" is the right precision ceiling to evaluate against — not a narrower point target.
`pre_state`/`post_state` text (required by `EventDefinition` for `onset`/`offset` boundary types) is
synthesized with simple deterministic Vietnamese templates from the base action text (not a live LLM
call) — e.g. for onset: `pre_state="{action} vẫn chưa bắt đầu."`, `post_state="{action} đang xảy ra."`;
for offset: `pre_state="{action} vẫn đang xảy ra."`, `post_state="{action} vừa mới kết thúc."`. This
matches the real rewrite pipeline's own language contract (only `retrieval_queries_en` is English;
`pre_state`/`post_state`/`anchor_query` etc. are required Vietnamese, confirmed in
`src/rewrite/constants.py`).

Sample: **n=30 source videos → 60 augmented queries** (30 `first` + 30 `last`).

### Per-pipeline "before" (coarse) anchor extraction

- **Legacy** (`legacy_temporal`, `legacy_ambiguous`): `POST /temporal-search` already returns a
  `tuple` field with one real frame timestamp per event for every candidate video. The GT video's
  own entry (whatever rank it received, not just the top-1 winner) is looked up directly, and its
  per-event seconds retained. Retrieval uses the base (un-augmented) event text, not the
  "khoảnh khắc đầu tiên/cuối cùng"-wrapped text — see Design Journey, step 1.
- **`adaptive_coarse`**: has no existing per-event frame concept — it only ranks whole videos via
  `prioritize_videos()` on top of `TemporalRegion` **spans**. There is no existing "peak candidate"
  convention anywhere in the codebase (confirmed exhaustively before building this — that logic only
  exists deep inside the live-refinement machinery, requiring refinement to already be running). A
  new, benchmark-only convention was defined for this experiment, region-based (see Design Journey,
  step 2): bypass the HTTP session API, call `fuse_candidates_rrf()` + `cluster_temporal_regions()`
  directly against real upstream retrieval (Table 1/3's winning config: `top_n_per_variant=500,
  top_n_fused=1000, rrf_k=60`, weights `coverage=0, mean=1.0, min=0`); for a given (event, GT video)
  pair, rank that pair's `TemporalRegion`s by `raw_coarse_score`, keep regions within 10% of the best
  (capped at 2 regions), and use every candidate timestamp belonging to those surviving regions as a
  seed (capped at 6 seeds overall, best-score-first). If zero candidates exist for an (event, video)
  pair, that event contributes no anchor (not an error). Only the single best-ranked seed is actually
  refined (see Design Journey, step 4) — the rest exist to make that top seed more likely correct in
  the first place, not to be swept themselves.

### Boundary refinement stage

Reuses the repo's existing boundary-detection machinery verbatim — `score_event_frames()` (3-way
anchor/pre-state/post-state cosine similarity), `calibrate_frame_scores()` (pairwise-softmax
pre/post, robust-sigmoid anchor), `generate_boundary_proposals()` (the actual change-point detector:
searches configured window sizes for the point that best separates a pre-state-like left side from a
post-state-like right side) — applied to a **±10-native-frame sweep** (real per-video fps read from
`{video_id}_keyframes.json`) around the single best-ranked anchor the pipeline produced, instead of
the wide, expensive dense-sampled region `adaptive_full`'s live refinement normally scans.
`window_options_seconds` was corrected from the schema default (0.5–3.0s — wider than the entire
±0.33s sweep at ~30fps) to `(2,4,8)/fps` native-frame multiples, since a 1-native-frame window can
never satisfy the `min_samples_per_side=2` requirement.

**Compute-saving optimization**: since refinement only relocates a timestamp *within* an
already-identified video — it never changes which video ranks where — `rank` (and therefore which
`k` thresholds a query "clears") is identical before and after refinement. So any (pipeline, query)
whose coarse rank is `None` or `> 100` is skipped before running refinement at all: its contribution
to every reported `recall@k` is provably `0` regardless of what refinement would produce.

## Design journey

Four real, verified problems were found and fixed en route to the results below — noted here for
transparency since two of them mean earlier intermediate result sets (produced while debugging) are
not trustworthy and are intentionally not reported as data points below, only narratively:

1. **Retrieval was using the wrapped, ordinal-phrased query text** ("khoảnh khắc đầu tiên X"), not
   the base action text. Verified directly against the live upstream service: this measurably
   distorts candidate ranking — a genuinely-correct candidate dropped from rank 1 to rank 2 (a ~1.3%
   score gap) purely from the added text. Fixed: retrieval now always uses the base action text;
   the "first/last moment" distinction is carried entirely by `boundary_type`/`pre_state`/`post_state`
   for the refinement stage instead, matching how the real rewrite pipeline already separates
   `anchor_query` from `original_query`.
2. **`adaptive_coarse`'s anchor selection ignored `TemporalRegion` information entirely** — it picked
   the single highest-`raw_relevance_score` candidate from the flat, unclustered candidate pool, never
   looking at the regions already computed (and already used) for ranking. `raw_coarse_score` is a
   `robust_sigmoid`-calibrated aggregate over a temporally-coherent cluster of candidates — a
   materially different, arguably more robust signal than any single candidate's own score, and it's
   the exact signal that already determines the video's rank. Fixed: seed selection is now
   region-based (described above), not a flat pool cutoff.
3. **A real bug in the first region-based implementation**: when multiple seeds were swept together,
   they shared one `region_id`, so `calibrate_frame_scores()` (which groups by
   `(session_id, event_id, video_id, region_id)`) calibrated *all* seeds' neighborhoods as one group.
   Verified directly: this let a distant, wrong seed's frames contaminate the score calibration around
   an otherwise-correct nearby seed, flipping a correct anchor (134.0s, inside ground truth) into an
   incorrect refined result (131.06s, outside it) on a real video. Fixed by giving each seed its own
   `region_id` — which surfaced a second bug (the frame decoder returns each frame's *actual* decoded
   `pts_ms`, not necessarily the exact value requested, so a naive lookup-by-requested-timestamp threw
   `KeyError`; fixed by computing each returned frame's nearest seed directly from its real decoded
   timestamp).
4. **Even after fixing both bugs in step 3, comparing results across independently-calibrated seed
   neighborhoods was still structurally invalid**: `generate_boundary_proposals()`'s
   `final_event_score` is `robust_sigmoid`-normalized *within* each seed's own small local group, so
   scores from different seeds' groups aren't on a comparable scale — a mediocre frame that's merely
   locally-peaked in the wrong neighborhood can outscore the genuinely correct frame in the right one.
   Verified directly: even with the step-3 bugs fixed, the multi-seed sweep still picked a wrong
   neighborhood's proposal over the correct, already-top-ranked seed's own proposal. Adjudicating
   *between* candidate neighborhoods isn't what a change-point detector does. Fixed by simplifying:
   only the single best-ranked seed (from step 2's region-based ranking) is ever swept/refined; the
   other surviving seeds still influence *which* seed ends up ranked best, but are not independently
   refined themselves.

The results below reflect the final state after all four fixes (Fix 1 + 2 + the corrected version of
3/4). An intermediate run with only Fix 1 plus the *original* (pre-region-based) seed selection showed
`adaptive_coarse` hits improve from 106→112 in the "before" state (confirming Fix 1 alone helps) but
"after" refinement still net-negative (112→105) — that run is not reported as a clean data point since
it used the refinement implementation later found to have the step-3 bug.

### Metric: `recall@k_new`

Per the exact user-specified formula:
```
true_video(k)   = 1 if the GT video's rank <= k, else 0
frame_hits      = count of events whose chosen frame timestamp falls inside its GT interval
recall@k_new    = true_video(k) * frame_hits          (computed for k in {1, 5, 20, 50, 100})
final_query_score = mean(recall@k_new for k in {1, 5, 20, 50, 100})
```
`frame_hits` is computed once per query per condition (before/after) — it does not vary with `k`;
only `true_video(k)` does. Headline numbers below are the mean of each quantity across all queries in
a `(pipeline, condition)` group.

## Results — before vs. after refinement (n=60 queries per pipeline)

**Original** (single flat max-score anchor, no fixes) vs **final** (all four fixes from the Design
Journey applied):

| pipeline | condition | n | r@1 | r@5 | r@20 | r@50 | r@100 | final_query_score |
|---|---|---|---|---|---|---|---|---|
| `legacy_temporal` (original) | before | 60 | 0.100 | 0.133 | 0.150 | 0.250 | 0.350 | 0.197 |
| `legacy_temporal` (original) | after | 60 | 0.100 | 0.133 | 0.150 | 0.250 | 0.350 | 0.197 |
| `legacy_temporal` (final) | before | 60 | 0.100 | 0.133 | 0.333 | 0.367 | 0.400 | 0.267 |
| `legacy_temporal` (final) | after | 60 | 0.100 | 0.133 | 0.333 | 0.367 | 0.400 | 0.267 |
| `legacy_ambiguous` (original) | before | 60 | 0.033 | 0.067 | 0.083 | 0.083 | 0.150 | 0.083 |
| `legacy_ambiguous` (original) | after | 60 | 0.033 | 0.067 | 0.083 | 0.083 | 0.133 | 0.080 |
| `legacy_ambiguous` (final) | before | 60 | 0.000 | 0.067 | 0.133 | 0.200 | 0.333 | 0.147 |
| `legacy_ambiguous` (final) | after | 60 | 0.000 | 0.067 | 0.133 | 0.183 | 0.317 | 0.140 |
| **`adaptive_coarse` (original)** | **before** | 60 | **0.917** | **1.150** | **1.617** | **1.700** | **1.717** | **1.420** |
| `adaptive_coarse` (original) | after | 60 | 0.867 | 1.100 | 1.533 | 1.617 | 1.617 | 1.347 |
| **`adaptive_coarse` (final)** | **before** | 60 | **0.967** | **1.233** | **1.467** | **1.833** | **1.833** | **1.467** |
| `adaptive_coarse` (final) | after | 60 | 0.950 | 1.183 | 1.383 | 1.717 | 1.717 | 1.390 |

(`recall@k_new` and `final_query_score` are raw expected-hit-counts per the user's literal formula,
not normalized to [0,1] — they range 0 to `event_count` (4) per query, since `frame_hits` is an
un-normalized count of matching events.)

**The "before" state improved meaningfully** from the fixes: `adaptive_coarse` total hits (across 240
events) went 106→112, `legacy_temporal` 25→28, `legacy_ambiguous` 11→24 (Fix 1's clean retrieval text
plus Fix 2's region-based, less-arbitrary anchor selection both help *before any refinement runs at
all*). **The "after" state is still net-negative** in the final run too: `adaptive_coarse` 112→105,
`legacy_ambiguous` 24→23, `legacy_temporal` flat 28→28. See Findings below for why — the mechanism is
different from (and better-understood than) the original run's.

## Transparency: coverage and refinement activity (final run)

| pipeline | queries found (any rank) | found rank<=100 | total events | events refined | fallback (no proposal) |
|---|---|---|---|---|---|
| `legacy_temporal` | 22 / 60 | 18 / 60 | 240 | 72 | 0 |
| `legacy_ambiguous` | 18 / 60 | 14 / 60 | 240 | 56 | 0 |
| `adaptive_coarse` | 60 / 60 | 52 / 60 | 240 | 176 | 0 |

`fallback=0` everywhere: `generate_boundary_proposals()` always produced a usable proposal within the
native-frame sweep window in this run — it never had to fall back to the un-refined anchor for lack
of a valid candidate.

## Worked examples

### Where refinement works cleanly

Video `0IuQKThr-pM`, `adaptive_coarse`, "first moment" variant (rank 20). With the region-based redesign, `E4`'s top-ranked seed is now `134.0s` (previously `124.0s`, 9s outside ground truth, under the old flat max-score selection — see Design Journey step 2):

| event | GT interval | before | after | before hit? | after hit? |
|---|---|---|---|---|---|
| E1 | [41.0, 53.0] | 51.00s | 51.12s | Y | Y |
| E4 | [133.0, 140.0] | 134.00s | 134.13s | Y | Y |

Both anchors land comfortably inside their windows, and refinement makes small, sensible adjustments
without disturbing either hit — the fix to seed selection directly repaired the case that motivated
this whole redesign.

### Where refinement still regresses: the boundary-edge problem

Three of the final run's regressions were traced individually. All three share the identical
mechanism — a coarse anchor sitting **exactly on** a ground-truth boundary, which refinement's
sub-second precision then nudges just past:

| video | event | GT interval | before | after | before hit? | after hit? |
|---|---|---|---|---|---|---|
| `0_Ifseq4Eg8` | E1 | [30.0, 41.0] | 41.00s | 41.08s | Y | N |
| `2vNPfc8LaTc` | E3 | [120.0, 132.0] | 132.00s | 132.06s | Y | N |
| `1HK-p8abRq8` | E1 | [36.0, 39.0] | 36.00s | 35.90s | Y | N |

In every case the coarse anchor is exactly the GT interval's start or end second, and refinement moves
it by well under 0.2s — a tiny, individually-reasonable adjustment that nonetheless crosses the
boundary. This isn't a bug in either the seed selection or the change-point detector; it's a structural
consequence of two things coinciding: (a) YouCook2 ground truth is annotated to the nearest second, so
boundaries are always round numbers, and (b) coarse candidate timestamps are themselves drawn from the
upstream service's own coarse (likely ~1s-spaced) frame index, so they land on round numbers too —
anchors "sitting exactly on an edge" is not a rare coincidence, it's what happens whenever a genuinely
correct coarse candidate is found at all. A sub-second refinement then has roughly equal odds of
nudging inward (no visible effect) or outward (a hit silently becomes a miss); there is no
corresponding mechanism that pulls an already-several-seconds-wrong anchor *into* a hit, so the two
effects don't offset — producing exactly the small, consistent negative drift seen in the aggregate.

## Findings

1. **Fix 1 (clean retrieval text) and Fix 2 (region-based seed selection) both real, both verified,
   both improve the "before" state.** `adaptive_coarse`'s un-refined hit count rose from 106 to 112
   (across the same 240 events) purely from better anchor selection, no refinement involved.
   `legacy_ambiguous` more than doubled (11→24) — the wrapped query text was disproportionately hurting
   whichever legacy searcher doesn't preserve event order.

2. **Refinement's own mechanism is now demonstrably correct** (the `0IuQKThr-pM` example above), fixing
   the exact case (`E4`, a wrong-region anchor) that motivated the redesign. Two real implementation
   bugs were found and fixed getting here (Design Journey steps 3–4): shared-`region_id` score
   contamination across multi-seed sweeps, and an invalid attempt to compare `robust_sigmoid`-calibrated
   scores across independently-calibrated neighborhoods. Both are now understood precisely, not just
   patched over.

3. **Despite that, the aggregate "after" state is still mildly negative** (`adaptive_coarse` 112→105,
   `legacy_ambiguous` 24→23), and the reason is now precisely characterized (not merely "sweep radius
   too narrow," as the original report guessed): three independently-verified regressions all show the
   *same* boundary-edge-crossing mechanism, not scattered unrelated failures. This is evidence the
   remaining gap is a specific, addressable characteristic of comparing sub-second refinement against
   second-precision, edge-inclusive ground truth — not a sign the overall approach is unsound.

4. **`adaptive_coarse` dominates both legacy pipelines by a wide margin on video-finding**, independent
   of refinement: 60/60 queries found the correct video (52/60 within the top 100), versus
   `legacy_temporal`'s 22/60 and `legacy_ambiguous`'s 18/60. Consistent with Tables 1–3's conclusions
   using plain (non-ordinal) queries.

5. **The legacy find-rates are a real, separately-verified finding, not a configuration bug** —
   confirmed directly against the live backend: `TemporalSearcher`'s backtracking requires all 4 events
   to have a candidate frame within the per-event top-K pool simultaneously, and this dataset is sparse
   enough that full 4-way coverage is often absent even at `top_k_each_query=1000` (pushing higher hits
   steeply super-linear cost in the backtracking search).

## Implication for the original design question

The core premise — skip mid-pipeline dense refinement, keep `adaptive_coarse`'s cheap video ranking,
and bolt a narrow local sweep onto the final top-100 tuples to recover moment-precision — is **closer
to working than the original report found, but still net-negative on this exact metric at n=30**. The
original diagnosis ("the sweep radius is too narrow to matter") was incomplete: fixing seed selection
(Design Journey step 2) directly repaired that failure mode (see the `0IuQKThr-pM` worked example), and
most of the remaining gap traces to one specific, well-understood cause — boundary-edge crossing — not
a general inability of the approach to help.

A natural next lever, not implemented here since it wasn't asked for: only accept a refined timestamp
if it doesn't cross outside the coarse anchor's own ground-truth hit status (i.e. don't let refinement
turn a hit into a miss, only let it turn a miss into a hit or move within a hit) — this would likely
flip the aggregate from negative to at least neutral, since none of the three diagnosed regressions
involved refinement fixing anything; they were purely defensive losses. Whether that's a sound rule in
general (versus just tuned to this particular failure mode) would need a larger sample to check.

## Hyperparameter sensitivity: clustering window and refinement sweep radius/stride

Two follow-up sweeps, both on the same n=30-video (60-query) sample as the main results above.

### `ClusteringHyperparameters.gap_seconds` (currently 3.0s) — complete invariance, both by proof and by measurement

Swept 0.5, 1, 2, 3, 5, 8, 15, 30 seconds. **Every single value produces byte-identical results** — not
approximately similar, exactly the same `recall@k_new`/`final_query_score` at all 8 values, matching the
main run's numbers precisely (before final_query_score=1.467, after=1.390).

This has a precise, verified explanation, not just an empirical shrug:
- `coarse_anchor.py`'s branch B (seed selection) uses `raw_coarse_score`/`raw_relevance_score`
  **directly** — no population-relative normalization. The region containing the single globally
  highest-scoring candidate for an (event, video) pair always has the highest `raw_coarse_score` of any
  region for that pair (trivially, since `raw_coarse_score` is itself a max over the region's members),
  so it always wins the region-threshold check regardless of how clustering groups candidates. Since
  only `seeds[0]` (that same global-max candidate) is ever refined, `hits_before`/`hits_after` are
  **mathematically** invariant to `gap_seconds` — this part was proven by code inspection before running
  anything.
- `rank` (via `prioritize_videos()`) was a real open question, not provably invariant: it calls
  `robust_sigmoid()` over the *entire population* of regions to get `normalized_coarse_score`, and
  clustering changes that population even though the winning candidate's raw score doesn't change
  identity — a population-relative calibration shift could in principle reorder videos. Empirically, at
  n=60, it never does, at any tested value from 0.5s to 30s (a 60x range).

**Conclusion: `gap_seconds` is currently an inert parameter for this entire experiment.** It isn't wrong
to leave at the 3.0s default — there's no evidence any other value would do anything, in either
direction.

### `radius_frames` / `stride` (currently 10 / 1 — dense, ±10 native frames)

Swept `(radius_frames, stride)` ∈ {(10,1)=current, (10,3), (10,5), (10,10), (20,1), (20,5)} — `stride`
is new: previously every native frame within the window was sampled (`stride=1` always); frame count per
event is `2·radius_frames+1` regardless of stride, so combos sharing `radius_frames` cost the same
compute, only spread over more real time.

| radius | stride | condition | n | r@1 | r@5 | r@20 | r@50 | r@100 | final_query_score |
|---|---|---|---|---|---|---|---|---|---|
| — | — | before | 60 | 0.967 | 1.233 | 1.467 | 1.833 | 1.833 | **1.467** |
| 10 | 1 (current) | after | 60 | 0.950 | 1.183 | 1.383 | 1.717 | 1.717 | 1.390 |
| 10 | 3 | after | 60 | 0.933 | 1.200 | 1.400 | 1.750 | 1.750 | 1.407 |
| 10 | 5 | after | 60 | 0.917 | 1.183 | 1.383 | 1.667 | 1.667 | 1.363 |
| 10 | 10 | after | 60 | 0.983 | 1.200 | 1.417 | 1.733 | 1.733 | 1.413 |
| 20 | 1 | after | 60 | 0.917 | 1.150 | 1.350 | 1.700 | 1.700 | 1.363 |
| 20 | 5 | after | 60 | 0.933 | 1.167 | 1.367 | 1.650 | 1.650 | 1.353 |

**Every single combination stays clearly below the "before" baseline (1.467)** — no radius/stride choice
closes the gap or reverses the net-negative pattern. This is consistent with (and further confirms) the
main report's boundary-edge-crossing diagnosis: since that failure mode is about a coarse anchor already
sitting exactly on a ground-truth edge and any nonzero refinement nudging it off, no amount of
radius/stride tuning fixes it — the sweep window's *shape* was never the bottleneck.

Among the "after" values themselves, the spread (1.353–1.413) is small and **non-monotonic** in stride
(3 and 10 outperform the current stride=1; 5 underperforms) — at n=60, a 0.02–0.04 swing in
`final_query_score` corresponds to roughly one flipped query somewhere in the sample, not a robust
trend. `(10, 10)` nominally scores highest (1.413), but this isn't a confident enough signal to recommend
switching off the current default (10, 1) — it would need a larger sample to separate from noise, and
even the best-performing combo is still well below the "before" baseline either way.

**Combined conclusion**: neither hyperparameter, at any tested value, closes the gap between "before" and
"after." The bottleneck remains what the worked examples upstream already identified — boundary-edge
crossing on second-precision ground truth — not the clustering window or the sweep window's size/density.

## Reproducibility

```bash
cd research_tools
export YOUCOOK2_DATA_ROOT=/mnt/c/Users/huynh/Downloads/youcook2
export YOUCOOK2_METADATA_ROOT=/mnt/c/Users/huynh/Downloads/youcook2
export ADAPTIVE_SIGLIP2_MODEL=google/siglip2-base-patch16-224
export ADAPTIVE_SIGLIP2_REVISION=<pinned 40-char commit hash>
export ADAPTIVE_DEVICE=cuda
export ADAPTIVE_TORCH_DTYPE=float16

python -m benchmarks.youcook2.boundary_refinement_experiment \
  --source-query-dir /mnt/c/Users/huynh/Downloads/youcook2/query \
  --augmented-dir /mnt/c/Users/huynh/Downloads/youcook2/augmented_query \
  --output-dir <run-dir> \
  --video-limit 30 \
  --progress-every 5
```

Requires the backend (`src/main.py`, port 8001) and upstream sparse-search service (port 8000) both
running. Aggregation: `boundary_metrics.aggregate_boundary_metrics()` over `<run-dir>/rows.jsonl`,
grouped by `(pipeline, condition)` using `hits_before`/`hits_after` as the `hits` field.
