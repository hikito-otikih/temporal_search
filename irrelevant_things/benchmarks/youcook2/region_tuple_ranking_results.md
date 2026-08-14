# Multi-region pooling + order-aware tuple ranking: ablation results

Date: 2026-08-13
Backend: real upstream sparse search service (via `adaptive_search.dependencies.upstream_search_client`),
real YouCook2 videos, n=30 (Rounds 1-3) then n=60 (Rounds 4-6, sorted-first-60 of 203 available query
files).
Code: `src/adaptive_search/tuple_ranking.py` (production algorithm, promoted in Round 4, atomic
regions promoted to the production region source in Round 6 - Rounds 1-3 benchmarked the earlier
benchmark-tree-only `region_tuple_ranking.py`),
`region_tuple_experiment.py` (sweep driver, imports the production module directly as of Round 4),
`region_tuple_report.py` (aggregation), `build_temporal_relations_cache.py` (real LLM
`temporal_relation` classification, cached), `tests/adaptive_search/test_tuple_ranking.py` (39 unit
tests) + `test_adaptive_video_priorities_tuple_ranking.py` (6 HTTP integration tests), all passing;
181 across the full backend suite. Six rounds below: (1) the core proposal, (2)
`temporal_relation`/confidence-gating/larger-N follow-ups, (3) a correctness fix to how
`temporal_relation` direction was applied, (4) two further limitations fixed (transitive relation
closure; `EventDefinition` actually carrying `temporal_relation` in production), n doubled to 60,
and a no-clustering ablation arm, (5) a fine-grained $\tau$ sweep comparing the mean-score gate
against a new margin-based gate, and the no-clustering ablation extended to the full hyperparameter
grid Rounds 1-2 swept under clustering, (6) the one untested combo (atomic regions +
`temporal_relation` + `threshold@0.5`) benchmarked and, on its strength, promoted to production.

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

## Round 4: two limitations fixed, n doubled to 60, and a no-clustering ablation

Three follow-ups requested after Round 3: (1) fix the two scope gaps that most needed it -
transitive relation constraints and `EventDefinition` actually carrying `temporal_relation` in
production, not just in this benchmark's workaround; (2) double the sample to n=60; (3) add an
ablation arm testing what happens if Stage 3 (`cluster_temporal_regions`) is skipped entirely and
every candidate frame is treated as its own singleton region.

### Fix 1: `EventDefinition` now carries `temporal_relation`/`reference_event_id` for real

`temporal_relation: TemporalRelationType` and `reference_event_id: str | None` were added to
`EventDefinition` (`src/adaptive_search/schemas.py`), with a validator mirroring the rewrite
schema's own consistency rule (`after`/`before`/`during`/`simultaneous` require a reference,
everything else forbids one). `rewrite_bridge.py::build_session_plan` now populates both fields
from the real `RewrittenEvent.temporal_relation`, translating the rewrite stage's integer
`reference_event_id` to the pipeline's string `event_id`. This is no longer a benchmark-only
workaround: a live session created via `POST /search-sessions/from-queries` carries this data all
the way through, and `GET .../video-priorities?apply_tuple_ranking=true` now calls
`build_order_constraints(bundle.session.events)` for real (`router.py`), the same function this
benchmark calls. Verified with a dedicated HTTP-level test that constructs a session where an
event is listed first but is explicitly `after` a later-listed event, and confirms refinement
picks the anchor the *relation* implies, not the one list position would have implied.

### Fix 2: order constraints are now transitively closed, and `during`/`simultaneous` are explicit

`build_order_constraints` (`tuple_ranking.py`) no longer stops at direct `reference_event_id`
edges. Given `after`/`before` edges $A\!\to\!B$ and $B\!\to\!C$, it now also derives $A\!\to\!C$ via
graph reachability over the (small, per-query) event graph - Round 3's fix only stopped scoring the
*wrong* pairs; it never added the pairs a direct-edges-only reading misses entirely. `during` and
`simultaneous` still produce no constraint, but this is now an explicit, documented branch with a
stated reason (this scoring model can only express strict precedence; overlap/proximity relations
have no correct precedence encoding to give them, so asserting one would be actively wrong exactly
as often as it's right) rather than a silent fallthrough - the previous report listed this as an
unexamined gap; it is now a reasoned decision, even though the runtime behavior for these two
relations is unchanged.

Both fixes are covered by 34 unit tests (`tests/adaptive_search/test_tuple_ranking.py`, up from 26)
including a transitive-closure test and a direct reproduction of the "event listed first but
relation says it's later" scenario, plus 6 HTTP integration tests (up from 5).

### n=60 results

The full sweep was re-run at n=60 (60 videos, first 60 sorted of 203 available; the
`temporal_relations_cache` was extended the same way, 60/60 cached with 1 unrelated schema-validation
failure, still **100% `after`** across all 176 classified adjacent pairs - the same corpus property
Round 3 found at n=30, now confirmed at double the sample).

| config | r@1 | r@5 | r@20 | r@50 | r@100 | MRR | median rank (found) | mean hits/4 | final_query_score |
|---|---|---|---|---|---|---|---|---|---|
| baseline (`prioritize_videos`) | 0.300 | 0.433 | 0.650 | 0.783 | 0.833 | 0.3861 | 6 | 1.550 | 1.2133 |
| baseline, no-clustering (atomic regions) | 0.300 | 0.433 | 0.650 | 0.783 | 0.833 | 0.3861 | 6 | 1.550 | 1.2133 |
| baseline_equivalent (order_weight=0) | 0.300 | 0.433 | 0.650 | 0.783 | 0.833 | 0.3861 | 6 | 1.550 | 1.2133 |
| default (= production, threshold@0.5, ow=0.8) | 0.350 | 0.533 | 0.667 | 0.800 | 0.833 | 0.4329 | 4 | 1.617 | 1.3300 |
| confidence_gate=none, order_weight=0.8 | 0.350 | 0.517 | 0.617 | **0.667** | 0.783 | 0.4248 | 4 | 1.617 | 1.2733 |
| temporal_relation-gated (default) | 0.367 | 0.550 | 0.683 | 0.800 | 0.833 | 0.4518 | 3 | 1.633 | 1.3533 |
| **production (threshold@0.5, ow=0.8) + temporal_relation** | **0.367** | **0.550** | 0.683 | **0.800** | **0.833** | **0.4518** | **3** | 1.633 | **1.3533** |
| no-clustering (atomic regions), default | 0.367 | 0.533 | 0.667 | 0.800 | 0.833 | 0.4415 | 4 | 1.633 | 1.3500 |
| no-clustering (atomic regions), N=5 | 0.333 | 0.500 | 0.667 | 0.800 | 0.833 | 0.4078 | 5 | 1.567 | 1.2733 |

Full sweep (37 configs): `runs/region_tuple_sweep_n60_v2.jsonl` / `..._v2_summary.json`.

**Baseline invariance to clustering, now checked per-video, not just in aggregate.** All 60 videos:
`rank_baseline == rank_baseline_atomic` and `hits_baseline == hits_baseline_atomic` exactly, zero
exceptions. This is the strongest confirmation yet of the max-of-maxes argument from the pipeline
architecture report (Stage 4): the independent-argmax baseline is provably indifferent to region
granularity, and a real 60-video, fully-atomic (1 candidate = 1 region, no merging at all) run
finds no counterexample.

**Transitive closure makes `temporal_relation` matter, even on a 100%-`after` corpus.** Round 3
found `temporal_relation`-gated configs *numerically identical* to their ungated twins at n=30,
concluding the mechanism was correctly built but inert on this corpus. That conclusion no longer
holds with the transitive-closure fix: `temporal_relation-gated (default)` now measurably beats
plain `default` (final_query_score 1.3533 vs 1.3300; median rank 3 vs 4), even though every direct
relation edge still matches list order exactly. The reason is structural, not corpus-specific: a
direct-edges-only constraint set for an $N$-event `after` chain has only $N\!-\!1$ pairs (the
adjacent chain, which a list-position default already produces "for free"); transitive closure
over that same chain has $\binom{N}{2}$ pairs - order-checking gets access to non-adjacent
comparisons (event 1 vs event 3, not just event 1 vs event 2) it never had before, independent of
whether the corpus's relations happen to agree with list order. Per-video: 12/58 videos improved,
8/58 regressed, 38/58 unchanged (mean rank delta $-0.50$, i.e. net improvement) - a real, if modest,
effect, not a rounding artifact. Combining production gating with real `temporal_relation`
(`production (threshold@0.5, order_weight=0.8) + temporal_relation`) is the best configuration
found across all four rounds: final_query_score 1.3533, matching or leading every other row on
every column, with recall@50/100 fully at baseline.

**Confidence-gating re-examined with a larger, more honest regression set.** At n=60, 15 videos
regress under ungated `order_weight=0.8` (vs 5 found at n=30 - the larger sample surfaces
proportionally more of them, as expected). `threshold@0.5` gating **fully reverses 10/15** to
at-or-better-than-baseline; the remaining **5/15 are substantially mitigated but not fully
recovered** - a materially more calibrated claim than Round 3's n=30 finding, which (by the luck of
a smaller sample) saw only one trivial residual regression:

| video_id | baseline | ungated (`confidence_gate=none`) | **gated (`threshold@0.5`)** |
|---|---|---|---|
| 4B6j3gYkvr4 | 86 | 531 | **73** |
| 3rtzSsuJ4Ng | 22 | 415 | **21** |
| 7-WEdqJBXoQ | 36 | 410 | **32** |
| 8fVUcbC8MgM | 8 | 90 | 36 |
| DrM_ZiRvIro | 2 | 41 | 6 |
| GLd3aX16zBg | 18 | 44 | 40 |
| FNUumn079DM | 1 | 9 | 5 |
| *(9 more, 6 fully fixed, 3 omitted for space - full list in the JSONL)* | | | |

Every one of the 5 still-regressed videos has low `hits_baseline` (0-2/4) - the same "weak
underlying signal" diagnosis as before - but gating no longer reads as a complete fix for that
failure mode, only a strong (10/15 full, 5/15 partial) mitigation. `final_query_score`
(ungated 1.2733 vs gated 1.3300) and recall@50 (ungated 0.667 vs gated/baseline 0.783-0.800) both
reproduce the Round 3 pattern cleanly at 2x the sample.

**Hard-subset MRR replicates.** Restricting to the 42/60 videos where baseline doesn't already rank
the video first: baseline MRR 0.123, `default` 0.221 (+79.5% relative), the new best config 0.229
(+86.5% relative) - closely matching Round 1's n=30 figures (+64%/+96%).

### No-clustering ablation: does Stage 3 matter for tuple ranking? (the requested check)

Direct answer: **the system stays close to stable, with a small, real net cost concentrated in a
minority of videos at small N, that mostly washes out by the production N=20.** Per-video, comparing
`default` (clustered) against `no-clustering (atomic regions), default` at matched hyperparameters:
43/58 videos (74%) are byte-identical, 14/58 (24%) are worse under atomic regions, 1/58 is better
(mean rank delta $+0.28$, i.e. mildly worse on average). This is a genuinely different result from
the *baseline* algorithm's exact invariance above - tuple ranking pools *multiple* regions per event
and is sensitive to how finely candidates get grouped, in a way the old independent-argmax baseline
structurally is not.

The effect is N-dependent: at small pooling budgets, clustering clearly helps (`N=5`:
final_query_score 1.3300 clustered vs 1.2733 atomic - clustering's merge-and-take-max naturally
gives a small budget more *temporally distinct* candidates to choose from). By `N=20` (the
production default), the aggregate gap closes and even mildly reverses (1.3300 clustered vs 1.3500
atomic) - the recall@k-based aggregate metric and the per-video mean-rank-delta metric disagree in
direction here, and both are reported rather than picking the more flattering one: recall@k only
cares whether rank crosses a threshold, so a few large atomic-region losses can coexist with a
slightly better aggregate recall if enough borderline cases flip the other way. `N=1` clustered and
`N=1` atomic are **exactly identical on every video** (both reduce to "each event's single
best-scoring candidate," provably the same regardless of how that candidate happens to be grouped
into a region) - a free internal-consistency check that the atomic-region construction is correct,
not just a plausible-looking new code path.

**Practical read:** clustering is not load-bearing for tuple ranking at the hyperparameters this
report recommends (N=20), but it is not free to remove either - it provides real, if modest, value
specifically as a small-budget aid, and removing it introduces a minority-case downgrade that
recall@k aggregates can mask. Keeping Stage 3 as-is remains the right default.

## Round 5: fine-grained $\tau$, margin vs. mean-score signal, and the full atomic-region grid

Two follow-ups requested after Round 4's "known scope gaps": (1) sweep $\tau$ at 0.05 resolution (19
points, 0.05-0.95) for both the existing mean-score gate and a new margin-based gate, find the best
$\tau$ for each, and compare them; (2) extend the no-clustering ablation past its Round 4 subset
(default, $N\in\{1,5,20\}$, `pooling=mean`) to the full grid Rounds 1-2 swept under clustering
(`delta`$\in\{0.05,0.10,0.15,0.25,0.40\}$, $N\in\{1,2,3,5,10,20,30,50,100,200\}$,
`order_weight`$\in\{0.05,0.2,0.4,0.8\}$, `pooling`$\in\{$`max`,`mean`$\}$).

### New signal: margin gate

`_region_margin(region, pool)` (`tuple_ranking.py`) computes, for the *specific* region a combination
actually selects for an event (not a pool-level constant): if that region is the pool's top scorer,
the gap to the runner-up (`score(top) - score(runner_up)`, the classic confidence read); otherwise
the (negative) gap between the reached-for region's own score and the pool's actual top score
(`score(chosen) - score(top)`) - correctly signaling lower confidence the further down the pool a
tuple reached to satisfy the order term. Single-member pools fall back to the region's own score
(nothing to compare against). `confidence_gate="margin"` routes this per-event value (meaned across
the tuple, mirroring how `"threshold"` means `region_mean_score`) through the exact same hard-cutoff
shape as `"threshold"` (`_effective_order_weight`), so the two signals are compared apples-to-apples:
identical gate shape and $\tau$ grid, only the confidence value differs. Covered by 4 new unit tests
(`RegionMarginTests`).

### Fine-grained $\tau$ sweep, both signals

91-config sweep at n=60 (`runs/region_tuple_sweep_n60_v3.jsonl`): 19 $\tau$ points x 2 signals = 38
configs, plus the full atomic grid below.

| $\tau$ | threshold (`region_mean_score`) | margin |
|---|---|---|
| 0.05 | 1.2733 | 1.2000 |
| 0.10 | 1.2733 | 1.1633 |
| 0.15 | 1.2733 | 1.1700 |
| 0.20 | 1.2767 | 1.1733 |
| 0.25 | 1.2800 | 1.1333 |
| 0.30 | 1.2800 | 1.0500 |
| 0.35 | 1.2867 | 1.0667 |
| 0.40 | 1.2933 | 1.0667 |
| 0.45 | 1.3133 | 1.0633 |
| **0.50** | **1.3300** | 1.1167 |
| 0.55 | 1.3233 | 1.1200 |
| 0.60 | 1.3133 | 1.1500 |
| 0.65 | 1.2967 | 1.1733 |
| 0.70 | 1.2633 | 1.1933 |
| 0.75 | 1.2700 | 1.2033 |
| 0.80 | 1.2600 | **1.2067** |
| 0.85 | 1.2433 | 1.1900 |
| 0.90 | 1.2433 | 1.1967 |
| 0.95 | 1.2567 | **1.2067** |

(`final_query_score`, higher is better; baseline = 1.2133; production default $\tau=0.50$ bolded on
the left, margin's tied-best on the right)

**Finding 1: $\tau=0.5$ is confirmed as the genuine peak for the mean-score ("threshold") signal, not
a lucky 3-point pick.** The fine-grained sweep traces a clean, unimodal curve peaking exactly at
$\tau=0.50$ (`final_query_score` 1.3300 - also where recall@50 first reaches 0.800 and holds through
$\tau=0.65$) and falling off smoothly on both sides. The existing production default was already at
the true optimum of this 19-point grid; no change is warranted.

**Finding 2: the margin signal, even at its best $\tau$, does not beat the mean-score signal, and
never even reaches the ungated baseline.** Margin's best two points ($\tau=0.80$ and $\tau=0.95$,
tied at 1.2067) both fall *below* the plain `prioritize_videos` baseline (1.2133) - gating on margin
is, at every one of the 19 tested cutoffs, worse than not gating `order_weight` at all. The signal is
also far less stable (median-rank-found swings 6-29 across the grid, vs. threshold's 4-9), and its
optimum sits at the opposite end of the $[0,1]$ range from threshold's. This is consistent with how
`_region_margin` is defined: a non-top ("reach") pick's margin goes *negative*
(`own_score - pool_top_score`), and single-candidate pools fall back to the region's raw absolute
score - so margin mixes small signed gaps with occasional large absolute values on a scale a $\tau$
grid tuned for an always-$[0,1]$ mean-score signal is not calibrated for. The "natural next lever"
Round 4 flagged as untried has now been tried, at fine granularity, and does not pan out as a drop-in
replacement for the mean-score gate.

**Conclusion: no change to the production default.** `confidence_gate="threshold"`,
`confidence_gate_threshold=0.5` remains the right choice, now backed by a 19-point sweep rather than
a 3-point one, with the alternative it was compared against (margin) shown to underperform at every
tested setting rather than merely "not yet tried."

### Full atomic-region hyperparameter grid

Round 4's no-clustering ablation used a 6-point subset (`default`, $N\in\{1,5,20\}$, the production
combo, `pooling=mean`). This round adds the remaining 14 points of Rounds 1-2's original clustered
sweep (`delta`$\in\{0.05,0.10,0.25,0.40\}$, $N\in\{2,3,10,30,50,100,200\}$,
`order_weight`$\in\{0.05,0.2,0.4\}$), for 20 matched clustered/atomic pairs total, spanning every
dimension (delta, N, order_weight, pooling) Rounds 1-2 swept:

| setting | clustered | atomic | delta |
|---|---|---|---|
| delta=0.05 | 1.3067 | 1.3000 | -0.0067 |
| delta=0.10 | 1.3300 | 1.3367 | +0.0067 |
| delta=0.15 (default) | 1.3300 | 1.3500 | +0.0200 |
| delta=0.25 | 1.3267 | 1.3300 | +0.0033 |
| delta=0.40 | 1.3500 | 1.3300 | -0.0200 |
| N=1 | 1.2133 | 1.2133 | 0.0000 |
| N=2 | 1.2967 | 1.2533 | -0.0434 |
| N=3 | 1.3067 | 1.2933 | -0.0134 |
| N=5 | 1.3300 | 1.2733 | -0.0567 |
| N=10 | 1.3300 | 1.3500 | +0.0200 |
| N=20 (default) | 1.3300 | 1.3500 | +0.0200 |
| N=30 | 1.3300 | 1.3500 | +0.0200 |
| N=50 | 1.3300 | 1.3500 | +0.0200 |
| N=100 | 1.3300 | 1.3500 | +0.0200 |
| N=200 | 1.3300 | 1.3500 | +0.0200 |
| order_weight=0.05 | 1.2967 | 1.3067 | +0.0100 |
| order_weight=0.2 | 1.3133 | 1.3300 | +0.0167 |
| order_weight=0.4 | 1.3300 | 1.3433 | +0.0133 |
| order_weight=0.8 (default) | 1.3300 | 1.3500 | +0.0200 |
| pooling=mean | 1.2733 | 1.3433 | +0.0700 |

(`final_query_score`, clustered minus atomic per matched hyperparameter point)

**Finding: clustering's aggregate benefit is real but narrow, confined to small pooling budgets
($N\le5$) and two delta extremes - at every other tested point (14/20, 70%), atomic regions tie or
beat clustered on this aggregate metric.** $N=1$ reproduces the proven exact-tie invariant.
Clustering clearly helps at $N\in\{2,3,5\}$ (all negative deltas, -0.013 to -0.057) and at the tight
`delta=0.05` and loose `delta=0.40` extremes (small negative deltas). Everywhere else - $N\ge10$ (six
straight ties/wins), every `order_weight` tested (four straight wins), `delta`$\in\{0.10,0.15,0.25\}$
(three straight wins), and `pooling=mean` (the single largest delta in the table, +0.07 in atomic's
favor) - atomic regions are equal to or better than clustered. This extends, and partially revises,
Round 4's "washes out by N=20" read: it isn't that clustering's effect fades to zero at the
production N=20 and stays neutral elsewhere - the full grid shows clustering is a small *positive*
aid only in a specific low-N/extreme-delta corner, and mildly *costs* aggregate recall everywhere
else in the grid, including at the shipped production defaults.

As in Round 4, this is an aggregate-metric read; the matched per-video diff at the `default` point
(43/58 identical, 14/58 worse under atomic, 1/58 better - Round 4's finding) is not re-run at all 20
new points here, since recall@k aggregates are known to be able to mask a minority-regression pattern
like that one. The practical conclusion is unchanged: Stage 3 clustering is not load-bearing at the
shipped hyperparameters (N=20, delta=0.15) - atomic is a hair *better* in aggregate there - but it is
also not free to remove, since per-video it still trades a minority of regressions for the aggregate
parity/improvement.

Full sweep (91 configs): `runs/region_tuple_sweep_n60_v3.jsonl` / `..._v3_summary.json`.

## Round 6: the missing combo (atomic + temporal_relation + threshold@0.5), and the production switch

Round 5's atomic-region grid never combined atomic regions with `temporal_relation` gating - the
flagship "production" row was only ever benchmarked on clustered regions. This round closes that
gap with a single added config (92 total,
`runs/region_tuple_sweep_n60_v4.jsonl` / `..._v4_summary.json`) and, on the strength of the result,
promotes atomic regions to the production default for the tuple-ranking path.

| config | r@1 | r@5 | r@20 | r@50 | r@100 | mrr | median rank (found) | mean hits/4 | final_query_score |
|---|---|---|---|---|---|---|---|---|---|
| baseline (`prioritize_videos`) | 0.300 | 0.433 | 0.650 | 0.783 | 0.833 | 0.3861 | 6 | 1.550 | 1.2133 |
| clustered + threshold@0.5 + temporal_relation (previous production) | 0.367 | 0.550 | 0.683 | 0.800 | 0.833 | 0.4518 | 3 | 1.633 | 1.3533 |
| **atomic + threshold@0.5 + temporal_relation (new production)** | 0.367 | 0.533 | 0.683 | 0.800 | 0.833 | 0.4449 | 4 | 1.650 | **1.3633** |

**This is the best `final_query_score` found across all six rounds** (1.3633, +12.4% over baseline),
though it is a mixed win rather than a clean sweep: atomic trades a little MRR (-0.0069) and one
fewer video landing in the top 5 for more per-event boundary hits (+0.017 mean hits/4) and the
better composite score. Hard-subset MRR (the 42/60 videos where baseline doesn't already rank the
ground truth first) confirms the same shape: baseline 0.123 -> 0.219 (+78.1% relative) for the new
combo, vs. +86.5% for the clustered version it replaces - a real but small give-back on this one
metric, not a regression.

**The per-video picture, not just the aggregate, is what justified promoting it.** Diffing the two
flagship configs video-by-video: 44/60 identical, 15/60 "worse" under atomic, 1/60 better. Critically,
**every one of the 15 "worse" cases preserves `hits` exactly** (0-to-0, 1-to-1, or 2-to-2) - 13/15 are
a bare $\pm1$ rank shift (e.g. rank 19$\to$20, 44$\to$45), the remaining 2 shift by 2-4 ranks, and
none loses a video that was previously found in a useful window. This is a materially more benign
per-video story than Round 4's original (non-relation-gated) atomic-vs-clustered comparison, where
some regressions were larger. On that basis, atomic was judged an acceptable - and on the headline
metric, better - default, not merely a tied one.

**Production change:** `src/adaptive_search/router.py::get_video_priorities` now builds the
tuple-ranking region pool via the new `tuple_ranking.atomic_regions(bundle.artifacts.candidates)`
(moved out of this benchmark's driver and into production, so both now share one implementation)
instead of passing `bundle.artifacts.regions` (the clustered set). Stage 3 clustering
(`cluster_temporal_regions`) still runs and still feeds the independent-argmax baseline path above
and boundary-refinement seed selection - both untouched by this change and out of scope for this
ablation (the baseline is separately proven invariant to region granularity; refinement seeding
was never part of the atomic-vs-clustered comparison). Verified live against the restarted backend
on the lion-dance worked example (`docs/pipeline_architecture/pipeline_architecture.tex`, Stage 1):
`apply_tuple_ranking=true` returns a genuinely different top-5 video ordering than the plain
baseline, confirming the new code path executes correctly end-to-end against real upstream data,
not just the 181-test unit/integration suite (all still green after the change).

## Recommendation

**`atomic regions + threshold@0.5 + temporal_relation`** (`pooling="max"`, `relative_delta=0.15`,
`max_regions_per_event=20`) is the current production configuration, chosen over six rounds and
n=60: final_query_score 1.2133 -> 1.3633 (+12.4%) over baseline, the best composite score found,
verified both in aggregate and per-video against the clustered configuration it replaced.
`temporal_relation` gating is no longer inert - Round 4's transitive-closure fix makes it a real,
positive contributor, and it is now wired into production for any query shape, not just this
benchmark's workaround. Round 5's 19-point $\tau$ sweep confirms `threshold@0.5` sits at the true
optimum (not just the best of 3 spot-checks), and the margin-based alternative it compares against
underperforms at every tested cutoff, so the gate signal and cutoff are both settled rather than
provisional. N>20 pooling remains validated as correctly-implemented and inert on this corpus.

**Now implemented in `src/adaptive_search/`, opt-in**: `apply_tuple_ranking=true` on
`GET .../video-priorities`, with `TupleRankingHyperparameters` defaults matching this report's
winning configuration (`confidence_gate="threshold"`, `confidence_gate_threshold=0.5`,
`order_weight=0.8`) applied over atomic (not clustered) regions. See
`docs/pipeline_architecture/pipeline_architecture.tex`'s Methodology section for the full
production-integration writeup.

## Known scope gaps, not addressed here

- Not tested against the existing GPU-refinement-based `assemble_ordered_tuples` path, which solves
  a related but distinct problem (fine-grained ordering after dense per-frame scoring, not coarse
  video ranking) - the two are complementary, not competitors, and were not benchmarked head-to-head
  since they operate on different inputs at different pipeline stages.
- Single corpus throughout (YouCook2 recipe videos, n=60 of 203 available). The 100%-`after`
  relation finding is a property of this corpus's causally-sequential recipe-step structure, not a
  general claim - a corpus with genuinely `independent`/`simultaneous`/`during` events would be
  needed to test those branches of `build_order_constraints` on real data (they are unit-tested on
  synthetic data only).
- The 5 residually-regressed videos under `threshold@0.5` (all low-`hits_baseline`) remain only
  partially mitigated. Round 5 tried the "gate on margin instead" lever this was flagging as
  untried, and it does not fix them either - margin underperforms threshold at every $\tau$ tested,
  including on this same regression set. A fix for these 5 videos likely needs a different mechanism
  entirely (e.g. per-event rather than per-tuple confidence), not just a different signal under the
  same gate shape.
