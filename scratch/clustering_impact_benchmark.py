"""Does temporal clustering (Stage 3) change ranking or refinement seeds
for the live adaptive_coarse pipeline? Sweeps ClusteringHyperparameters
against one real, already-fetched fused candidate set (from the lion-dance
session) using the actual production functions - no reimplementation.
"""
import json
import sys

sys.path.insert(0, "/home/huynhchiton/projects/temporal_search/src")

from adaptive_search.dependencies import upstream_search_client
from adaptive_search.client import QueryVariant
from adaptive_search.retrieval import fuse_candidates_rrf
from adaptive_search.algorithms import cluster_temporal_regions, prioritize_videos
from adaptive_search.boundary_seeds import select_event_seeds
from adaptive_search.schemas import (
    ClusteringHyperparameters,
    RetrievalHyperparameters,
    RefinementHyperparameters,
    SearchConstraints,
)

with open("scratch/lion_dance_result.json", encoding="utf-8") as f:
    saved = json.load(f)

session = saved["session"]["session"]
event_ids = [e["event_id"] for e in session["events"]]
retrieval_variants = session["retrieval_variants"]

variants: list[QueryVariant] = []
for event_id in event_ids:
    for idx, text in enumerate(retrieval_variants[event_id]):
        variants.append(QueryVariant(event_id=event_id, variant_id=f"v{idx}", text=text))

print(f"=== fetching real candidates: {len(variants)} variants, top_k=50 ===")
raw_candidates = upstream_search_client.retrieve_candidates(
    session_id="bench", variants=variants, top_k=50
)
print(f"raw candidates fetched: {len(raw_candidates)}")

fused = fuse_candidates_rrf(raw_candidates, RetrievalHyperparameters())
print(f"fused candidates: {len(fused)}")
candidates_by_id = {c.id: c for c in fused}

refinement_params = RefinementHyperparameters()
constraints = SearchConstraints()


def fingerprint(gap, margin, max_region, label):
    params = ClusteringHyperparameters(
        gap_seconds=gap, margin_seconds=margin, max_region_seconds=max_region
    )
    regions = cluster_temporal_regions(fused, params)
    priorities = prioritize_videos(regions, event_ids, refinement_params, constraints)
    ranking = [
        (p.video_id, p.event_coverage, round(p.mean_best_event_score, 9), round(p.priority_score, 9))
        for p in priorities[:20]
    ]
    seeds = {}
    for p in priorities[:5]:
        for event_id in event_ids:
            s = select_event_seeds(
                regions=regions,
                candidates_by_id=candidates_by_id,
                event_id=event_id,
                video_id=p.video_id,
            )
            seeds[(p.video_id, event_id)] = s[0] if s else None
    return {
        "label": label,
        "region_count": len(regions),
        "ranking": ranking,
        "seeds_top5": seeds,
    }


configs = [
    (3.0, 3.0, 30.0, "default (baseline)"),
    (0.1, 0.1, 2.0, "near-atomic: almost no clustering at all"),
    (0.5, 0.5, 2.0, "tight"),
    (1.0, 1.0, 10.0, "moderate-tight"),
    (5.0, 5.0, 40.0, "loose"),
    (8.0, 8.0, 60.0, "looser"),
    (15.0, 5.0, 60.0, "wide gap, default-ish margin"),
    (30.0, 10.0, 120.0, "very loose"),
    (60.0, 15.0, 300.0, "extreme: whole-video-ish merging"),
]

results = [fingerprint(*c) for c in configs]
baseline = results[0]

print()
print(f"{'config':45s} {'#regions':>9s}  ranking match  seeds match")
all_identical = True
for r in results:
    ranking_match = r["ranking"] == baseline["ranking"]
    seeds_match = r["seeds_top5"] == baseline["seeds_top5"]
    if not (ranking_match and seeds_match):
        all_identical = False
    print(f"{r['label']:45s} {r['region_count']:9d}  {str(ranking_match):>13s}  {str(seeds_match):>11s}")

print()
print("ALL CONFIGS IDENTICAL TO BASELINE:" , all_identical)

if not all_identical:
    print()
    print("=== first divergence detail ===")
    for r in results[1:]:
        if r["ranking"] != baseline["ranking"]:
            print(f"{r['label']}: ranking differs")
            for a, b in zip(baseline["ranking"], r["ranking"]):
                if a != b:
                    print("  baseline:", a)
                    print("  variant :", b)
        if r["seeds_top5"] != baseline["seeds_top5"]:
            print(f"{r['label']}: seeds differ")
            for k in baseline["seeds_top5"]:
                if baseline["seeds_top5"][k] != r["seeds_top5"].get(k):
                    print("  ", k, "baseline=", baseline["seeds_top5"][k], "variant=", r["seeds_top5"].get(k))

# --- bonus: does clustering granularity matter once a region is rejected? ---
print()
print("=== bonus: region-level reject, does clustering granularity change the outcome? ===")
from adaptive_search.schemas import EventConstraint

top_video = baseline["ranking"][0][0]
tight_regions = cluster_temporal_regions(fused, ClusteringHyperparameters(gap_seconds=0.5, margin_seconds=0.5, max_region_seconds=2.0))
loose_regions = cluster_temporal_regions(fused, ClusteringHyperparameters(gap_seconds=30.0, margin_seconds=10.0, max_region_seconds=120.0))

for label, regions in [("tight", tight_regions), ("loose", loose_regions)]:
    top_regions_for_video_evt0 = [r for r in regions if r.video_id == top_video and r.event_id == event_ids[0]]
    top_regions_for_video_evt0.sort(key=lambda r: -r.raw_coarse_score)
    if top_regions_for_video_evt0:
        best = top_regions_for_video_evt0[0]
        print(f"{label}: best evt0 region for {top_video} spans [{best.start_seconds:.1f},{best.end_seconds:.1f}]s, "
              f"{len(best.candidate_ids)} candidates -> rejecting it removes {len(best.candidate_ids)} candidates at once")
