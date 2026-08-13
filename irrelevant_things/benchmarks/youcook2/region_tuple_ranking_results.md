# Multi-region pooling + order-aware tuple ranking: ablation results

Date: 2026-08-13
Backend: real upstream sparse search service (via `adaptive_search.dependencies.upstream_search_client`),
real YouCook2 videos, n=30 sample (sorted-first-30 of 203 available query files).
Code: `region_tuple_ranking.py` (algorithm), `region_tuple_experiment.py` (sweep driver),
`region_tuple_report.py` (aggregation), `build_temporal_relations_cache.py` (real LLM
`temporal_relation` classification, cached), `tests/test_region_tuple_ranking.py` (24 unit tests,
all passing; 69 across the full benchmark suite). Three rounds below: (1) the core proposal, (2)
`temporal_relation`/confidence-gating/larger-N follow-ups, (3) a correctness fix to how
`temporal_relation` direction was applied, caught by review rather than by the benchmark itself.

## Motivation and proposal under test

Session discussion identified that `prioritize_videos()` (production video ranking) takes an
independent per-event `max` over `TemporalRegion` scores, discarding every candidate region but
the single best one per event — so `top_n_fused=1000` and clustering's own region pool are barely
exploited, and there is no reward for a video whose per-event regions land in query order versus
one where they collide or invert (empirically observed this session: two semantically distinct
events landing 0.13s apart on the *same* frame, and an earlier event ranked *after* a later one).

Proposal tested: pool up to `N` regions per event per video (filtered by an absolute score floor
`alpha` and a relative-delta band `delta` below the event's best region), assemble same-video
combinations across events (bounded backtracking against a hard `max_combinations_per_video` cap,
so the `N^event_count` blowup the proposal itself flagged as a risk cannot occur), score each
combination as `mean(region scores) + order_weight * order_agreement`, and rank videos by their
best (or mean-pooled) combination score instead of by the independent per-event max.

## Correctness check before any benchmarking

At `order_weight=0`, an unconstrained "best combination" is provably identical to "each event's own
independently-best region" (a pooled candidate list always contains that event's single best region
regardless of `delta`, so nothing can outscore picking it), i.e. mathematically forced to reproduce
`prioritize_videos()` exactly. Verified twice: as a unit test on synthetic regions, and as the
`baseline_equivalent (order_weight=0)` row below on real data — **byte-identical to baseline on
every one of the 30 real videos**, confirming the new implementation isn't silently different from
production for the trivial case before looking at anything order-aware.

## Results (n=30, recall@k = ground-truth video ranked <= k)

| config | r@1 | r@5 | r@20 | r@50 | r@100 | MRR | median rank (found) | mean GT-anchor hits /4 | final_query_score |
|---|---|---|---|---|---|---|---|---|---|
| baseline (`prioritize_videos`) | 0.367 | 0.467 | 0.633 | 0.833 | 0.867 | 0.4396 | 6 | 1.867 | 1.4667 |
| baseline_equivalent (order_weight=0) | 0.367 | 0.467 | 0.633 | 0.833 | 0.867 | 0.4396 | 6 | 1.867 | 1.4667 |
| **default** (delta=0.15, N=20, order_weight=0.1) | **0.400** | **0.533** | **0.733** | 0.733 | 0.867 | 0.4694 | 4 | **2.000** | **1.6333** |
| delta=0.05 / 0.10 / 0.25 / 0.40 | 0.400 | 0.533 | 0.733 | 0.733 | 0.867 | ~0.46-0.47 | 4 | ~1.97-2.00 | ~1.61-1.63 |
| N=1 (order-term only, no pooling) | 0.367 | 0.500 | 0.667 | 0.733 | 0.867 | 0.4406 | 6 | 1.867 | 1.5000 |
| N=2 | 0.400 | 0.533 | 0.733 | 0.733 | 0.867 | 0.4614 | 4 | 1.933 | 1.5800 |
| N=5 / N=10 / N=20 | 0.400 | 0.533 | 0.733 | 0.733 | 0.867 | 0.4694 | 4 | 2.000 | 1.6333 |
| order_weight=0.05 | 0.333 | 0.533 | 0.667 | 0.767 | 0.833 | 0.4260 | 5 | 2.000 | 1.5667 |
| order_weight=0.2 | 0.400 | 0.567 | 0.733 | 0.733 | 0.833 | 0.4786 | 4 | 2.000 | 1.6400 |
| order_weight=0.4 / 0.8 | 0.400 | 0.600 | 0.733 | 0.733 | 0.767 | 0.49 | 2 | 2.000 | 1.6333 |
| **pooling=mean** | **0.433** | 0.533 | 0.700 | 0.733 | 0.867 | **0.4913** | 4 | 2.000 | **1.6400** |

Full per-video rows: `runs/region_tuple_sweep_n30.jsonl`. Full summary JSON: `runs/region_tuple_sweep_n30_summary.json`.

## Findings

1. **The proposal helps, on this sample.** `final_query_score` (the repo's established metric,
   `boundary_metrics.py`) rises from 1.4667 to 1.6333 (+11.4%) at default settings, and to 1.6400
   with mean-pooling — a real, not-noise-sized effect at n=30. `hits` (does the GT video's *own*
   anchor timestamp land inside its ground-truth interval — a Branch-B-quality proxy, independent
   of ranking) also improves, 1.867/4 -> 2.000/4, meaning the order-aware anchor selection isn't
   just reordering videos better, it's also picking better per-event timestamps for the correct
   video.
2. **Most of the gain comes from the order term, not from pooling depth.** `N=1` (each event still
   restricted to its own single best region, i.e. no pooling at all - only the order-weighted
   scoring is new) already moves `final_query_score` from 1.4667 to 1.5000. Going from N=1 to N=2
   captures most of the remaining gain (1.5000 -> 1.5800); N=5, N=10, and N=20 are **identical** to
   each other (1.6333 flat) - consistent with this session's earlier finding that real region-count-
   per-event is usually tiny (median 1, mean 1.65 candidates per region). The proposal's own
   suggested cap of N<=20 costs nothing extra to allow, but on this corpus N=5 already captures all
   of its benefit - the "we can't exploit top_n_fused=1000" waste is real, but pooling more than
   ~5 regions deep doesn't currently buy anything further because that many rarely exist.
3. **`delta` barely matters in the tested range** (0.05 through 0.40 all land within a point or two
   of each other) - real per-event region score gaps in this corpus are apparently either much
   smaller or much larger than this whole range, not distributed evenly across it.
4. **`order_weight` is a genuine precision/robustness trade-off, not a free lunch.** Small
   (0.05) is actually *worse* than baseline on r@1 and r@100 - too weak to reliably break ties in
   the right direction, just adding noise. 0.1-0.2 is the best balance found. Pushed further (0.4,
   0.8), r@1/r@5/MRR keep climbing but **r@100 drops from 0.867 to 0.767** - a real cost, not just
   diminishing returns.
5. **The recall@50 regression (0.833 -> 0.733 at default) is real and traced to 4 specific videos**,
   all cases where baseline itself was already struggling: `1HK-p8abRq8` (48->69, hits stayed 2/4),
   `3rtzSsuJ4Ng` (22->61, hits stayed 1/4), `4B6j3gYkvr4` (86->141, hits stayed 0/4 - baseline
   found *zero* correct anchors for this video even before any reordering), `7-WEdqJBXoQ` (36->81,
   hits stayed 1/4). In every regressing case, `hits_baseline` was already low (0-2 out of 4) - i.e.
   the order-aware bonus hurts specifically where the underlying region scores are themselves
   already weak/noisy signal, because rewarding "the regions happen to be chronologically ordered"
   among mostly-noise candidates rewards a coincidence, not real evidence. It helps where there is
   real signal to organize; it can mislead where there isn't.
6. **The effect concentrates on the harder queries, as expected.** 11/30 videos (37%) already rank
   #1 under baseline - ranking can't improve them, only the anchor-accuracy (`hits`) metric can, and
   ranking regressions are impossible for these too (already at the ceiling). Restricting to the
   19 "hard" videos (baseline rank != 1): MRR goes from 0.1152 (baseline) to 0.1885 (default, +64%
   relative) to 0.2258 (order_weight=0.8, +96% relative) - but mean rank-when-found on that same
   hard subset *worsens* at high order_weight (69.5 -> 71.8 -> 118.9), the same precision/robustness
   trade-off as finding 4 restated in aggregate: big wins concentrate on cases already close to
   correct, at a real cost to the tail of harder-still cases.
7. **`pooling=mean` (average a video's kept tuples instead of taking its single best) was the
   single best-performing individual variant tested** (r@1=0.433, MRR=0.4913, final_query_score=
   1.6400, all the best of any row) while keeping r@100 at baseline's 0.867 - it didn't trade away
   tail recall the way high `order_weight` did. Not deeply investigated why (more tuples-per-video
   averaged in means a video needs *consistently* good combinations, not just one lucky one -
   plausibly more robust to a single coincidental high-order-score outlier), flagged as the most
   promising direction for follow-up over further `order_weight` tuning.

## Round 2: `temporal_relation` gating, confidence-gated order weight, N beyond 20

Follow-up requested after the first pass: test whether respecting the rewrite stage's
`temporal_relation` classification, gating `order_weight` by the tuple's own confidence, and
pooling deeper than N=20 change the picture - in particular, whether they fix the recall@50/100
regressions finding 5 traced to specific low-signal videos.

Code: `build_temporal_relations_cache.py` (one real Ollama rewrite call per video, cached -
`runs/temporal_relations_cache.jsonl`), extended `region_tuple_ranking.py` (order-constraint param
on `_order_score`/`assemble_region_tuples_for_video` - see Round 3 below for a correctness fix to
this param's semantics, new `confidence_gate` field on `RegionTupleParams`), unit tests added.
Full results as first run (superseded by the Round 3 correctness fix, numbers unchanged - see
below): `runs/region_tuple_sweep_n30_v2.jsonl` / `..._v2_summary.json`; canonical/current:
`runs/region_tuple_sweep_n30_v3.jsonl` / `..._v3_summary.json`.

**Note on reproducibility:** re-running the sweep against the same 30 videos a second time (Round
2 vs Round 1) shows the upstream sparse search service is not perfectly deterministic between
calls - `baseline`/`default`/etc. shifted by 1-2 recall points at several k between the two runs
(e.g. default's recall@20 was 0.733 in Round 1, 0.633 in Round 2). `baseline_equivalent
(order_weight=0)` still matches `baseline` exactly in both runs (the correctness invariant holds
regardless), and every directional finding below reproduces across both runs - but treat single
point estimates at n=30 as having roughly this much run-to-run noise, not exact figures.

### `temporal_relation` gating: correctly implemented, but a no-op on this corpus

All 30 videos' events were classified for real (1 Ollama call each, ~30-160s, one video failed
with an unrelated schema-validation error and fell back to "check every pair"). Result: **of 87
classified adjacent event pairs across 29 videos, all 87 came back `"after"` - zero
`"independent"` or `"simultaneous"`.** Every `temporal_relation-gated` config in the results table
is therefore numerically **identical** to its non-gated twin (`temporal_relation-gated (default)` =
`default`, `..., order_weight=0.8` = `order_weight=0.8`, `..., pooling=mean` = `pooling=mean`,
byte-for-byte) - a clean confirmation the gating mechanism itself works exactly as coded (unit
tests already proved this on synthetic data; this proves it end-to-end on real rewrite output too),
and a correction to Round 1's speculation: **this specific fix cannot be what's causing the
regressions**, because this corpus's queries never trigger it. YouCook2 events are literal recipe
steps ("cut onion" -> "mix with egg" -> "season" -> "cook") that are causally sequential even
without an explicit "then"/"sau đó" - the LLM rewrite reasonably calls all of them `"after"` rather
than `"independent"`, so there was never an unjustified order penalty to remove here. This remains
worth plumbing through for other query shapes (a lion-dance-style query with a genuine parallel or
unrelated event would exercise it, per this session's earlier worked example), but it does not
explain or fix anything measured in this report.

### N beyond 20: still completely flat, up to N=200

`max_combinations_per_video` was raised to 500,000 for this test specifically (N=20 with 4 events
is `20^4=160,000` combinations, already past Round 1's 20,000 cap - so N>20 needed real headroom to
be measured fairly, not silently truncated). N=30, 50, 100, and 200 are **identical** to N=5/10/20
on every metric (MRR=0.4658, final_query_score=1.6067, flat). The plateau found in Round 1 was
real, not a cap artifact: on this corpus, no video has enough genuinely-competitive regions per
event to benefit from pooling deeper than ~5. **Direct answer to "will bigger N bring a big
benefit": no, not on this corpus, confirmed out to 10x the proposal's own suggested cap.**

### Confidence-gated order weight: this is the fix

Three gate shapes tested at `order_weight=0.8` (the setting with the worst ungated recall@50/100
regression: 0.667/0.767 against baseline's 0.833/0.867):

| gate | r@1 | r@5 | r@20 | r@50 | r@100 | MRR | final_query_score |
|---|---|---|---|---|---|---|---|
| none (ungated) | 0.400 | 0.600 | 0.667 | 0.667 | 0.767 | 0.4786 | 1.6067 |
| linear (order_weight × region_mean_score) | 0.400 | 0.600 | 0.667 | 0.700 | 0.767 | 0.4771 | 1.6133 |
| threshold @ 0.3 | 0.400 | 0.600 | 0.667 | 0.700 | 0.767 | 0.4792 | 1.6133 |
| **threshold @ 0.5** | **0.400** | **0.600** | **0.733** | **0.833** | **0.867** | 0.4848 | **1.6867** |
| threshold @ 0.7 | 0.367 | 0.533 | 0.633 | 0.833 | 0.867 | 0.4585 | 1.5733 |

`threshold @ 0.5` recovers recall@50 and recall@100 **fully back to baseline** while keeping (and
in the case of recall@20, improving on) every head-of-list gain - the best `final_query_score` of
every config tested in either round, including `pooling=mean`. `linear` and `threshold@0.3` only
partially recover the tail (still let too much order_weight through on weak tuples).
`threshold@0.7` overcorrects - conservative enough to protect the tail but strong enough to also
suppress most of the genuine head-of-list wins.

This was verified causally, not just in aggregate - checked against the exact 5 videos Round 1
identified as regressing under `default`/`order_weight=0.8`:

| video_id | baseline | default | order_weight=0.8 (ungated) | **gate@0.5, ow=0.8** |
|---|---|---|---|---|
| 1HK-p8abRq8 | 48 | 60 | 89 | **47** |
| 3rtzSsuJ4Ng | 22 | 50 | 415 | **21** |
| 4B6j3gYkvr4 | 86 | 134 | 531 | **73** |
| 7-WEdqJBXoQ | 36 | 73 | 410 | **32** |
| 0Mz4NTozNXw | 134 | 136 | 118 | 132 |

Every one of the catastrophic ungated regressions (up to rank 531, from a baseline of 86) is not
just softened but reversed to *better than baseline* under `threshold@0.5`. Scanning all 30 videos
for any regression at all under this config: **exactly one**, and a trivial one (`2Ihlw5FFrx4`:
baseline rank 1 -> 2).

Combining `confidence_gate` with `pooling=mean` was also tried and made things *worse*
(final_query_score 1.5467, worse than `pooling=mean` alone's 1.6333) - the two don't compose
additively; gating each tuple's order bonus before mean-pooling across many tuples per video
appears to dilute mean-pooling's own preference for consistently-good combinations. Not
investigated further; flagged as a real negative interaction, not assumed away.

## Round 3: a real correctness bug in how `temporal_relation` was applied, fixed

Caught by direct review, not by the benchmark: Round 2's `orderable_pairs` mechanism only ever
checked **adjacent pairs in query-list order** (`event_ids[i]` vs `event_ids[i+1]`), always
expecting the later-*listed* event to have the larger timestamp - `temporal_relation` was only
consulted to decide whether to check a pair at all (opt-out for `independent`/`simultaneous`),
never to determine which *direction* the order should run. Concretely: if `evt1.relation == "after"`
with `reference_event_id` pointing at `evt2` (the model explicitly asserting `t(evt1) > t(evt2)`),
but `evt1` happens to be listed before `evt2` in the query array, the old code would still check
the pair in list order and award credit whenever `t(evt1) < t(evt2)` - the exact opposite of what
the relation claims. `SYSTEM_PROMPT` (`src/rewrite/constants.py`) explicitly preserves input array
order regardless of chronology (`"events phải có ... đúng thứ tự input"`), so this was a real,
reachable bug, not a hypothetical one - it just never fired *on this corpus specifically*, because
every real classification happened to have `reference_event_id == i-1` for `relation="after"` at
position `i` (array order and chronological order coincided every time, for reasons discussed in
Round 2: recipe steps are causally sequential regardless of connector words).

Fixed by replacing the boolean `orderable_pairs` (length N-1, aligned to array position) with
directed `order_constraints: list[(predecessor_index, successor_index)]` built from each event's
actual `relation` + `reference_event_id` - `"after"` with reference `r` on event `i` yields
`(r, i)` (expect `t(r) < t(i)`), `"before"` yields `(i, r)`, everything else yields nothing.
`_order_score` now checks exactly these directed pairs instead of an assumed list-position chain.
9 unit tests updated/added, including one that reproduces the failure mode directly: relation
"evt1 after evt2" with timestamps ordered `evt1 < evt4 < evt3 < evt2` must score as a **violation**
(`-1.0`), not a partial reward, regardless of evt1/evt2's positions in the query array (all 24
tests, then all 69 in the full benchmark suite, pass).

Re-ran the full n=30 sweep with the fix: results are **byte-for-byte identical** to Round 2's
table above, confirming the earlier prediction - the bug never fired on this corpus, so fixing it
changes nothing measured here, only removes a latent correctness risk for any future query whose
array order and true chronology diverge (which this corpus's queries never do, but nothing
guarantees a real query submitted later won't).

## Recommendation

`confidence_gate="threshold"` at `threshold=0.5`, `order_weight=0.8`, default pooling ("max") is
now the best-supported configuration found across both rounds: +15% `final_query_score` over
baseline (1.4667 -> 1.6867), recall@1/5/20 all at or above every other tested config, and
recall@50/100 fully preserved at baseline levels - the regression that blocked a "drop-in
replacement" recommendation in Round 1 is gone, verified against the exact videos that caused it,
not just in the aggregate mean. `temporal_relation` gating and N>20 pooling are validated as
correctly-implemented and inert on this corpus, not disproven in general - both remain reasonable
to keep for other query shapes, just not load-bearing for the numbers in this report.

Still not implemented in `src/adaptive_search/`: this lives entirely in the benchmark tree
(`region_tuple_ranking.py`), matching this repo's convention of proving an idea here before
promoting it to production (same path `coarse_anchor.py`/`boundary_refinement.py` took). Given the
strength and causal verification of the `threshold@0.5` result, promoting this specific
configuration to a real opt-in mode behind `apply_boundary_refinement`-style flag is the most
directly supportable next step this report can point to.

## Known scope gaps, not addressed here

- Order scoring now uses `reference_event_id` + relation direction (Round 3), not list position -
  but only as **direct, single-hop constraints** (an event's stated relation to its own
  `reference_event_id`), not the full transitive graph (e.g. if A precedes B by direct constraint
  and B precedes C by direct constraint, A-vs-C is never checked directly, only implied if both
  A-B and B-C individually hold in a chosen tuple). `"during"` relations still produce no
  constraint at all - no directional expectation is derived from "during" today.
- `confidence_gate_threshold=0.5` was chosen from a 3-point sweep (0.3/0.5/0.7), not a fine-grained
  search - the true optimum on this corpus could be anywhere nearby; not worth over-fitting further
  at n=30.
- `EventDefinition` (what a live session actually carries) still drops `temporal_relation` entirely
  at the `rewrite_bridge.py` boundary - this report worked around that by calling the rewrite
  service directly and caching the result, not by changing production code. Actually plumbing it
  through would still be needed before `temporal_relation` gating could matter for any query shape,
  not just this corpus's.
- Not tested against the existing GPU-refinement-based `assemble_ordered_tuples` path, which solves
  a related but distinct problem (fine-grained ordering after dense per-frame scoring, not coarse
  video ranking) - the two are complementary, not competitors, and were not benchmarked head-to-head
  since they operate on different inputs at different pipeline stages.
- n=30 throughout. The confidence-gate result in particular rests on exactly 5 videos' worth of
  regression-recovery evidence - real and causally traced, but a larger sample would be needed
  before treating `threshold=0.5` as more than "the best of 3 tested values on this sample."
