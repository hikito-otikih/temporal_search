"""Ad hoc six-way verification for the adaptive_coarse + boundary_refinement
production migration (docs/ADAPTIVE_PIPELINE_MIGRATION.md).

Not part of the permanent test suite - issues real HTTP requests against an
already-running `src/main.py` service (default http://localhost:8001) and,
for adaptive_coarse, drives the real session lifecycle
(create -> commands/retrieve against the real upstream sparse-search service
-> GET video-priorities), exactly like a real client would. Needs the real
upstream search service reachable (UPSTREAM_SEARCH_URL) and, to exercise the
"applied" (not just "skipped_runtime_unavailable") branches, a GPU-configured
boundary-refinement runtime (YOUCOOK2_DATA_ROOT + ADAPTIVE_SIGLIP2_REVISION).

Usage:
    .venv/bin/python3 scripts/verify_migration.py [--base-url URL] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://localhost:8001"
DEFAULT_QUERY_DIR = Path("/mnt/c/Users/huynh/Downloads/youcook2/query")
GROUP_FILES = [
    "VswrGW9b3ck.txt",  # 3 events
    "FTdfwoxgMTU.txt",  # 3 events
    "0IuQKThr-pM.txt",  # 4 events
    "0JVmVXLrNZo.txt",  # 4 events
    "0Mz4NTozNXw.txt",  # 4 events
]

EVENT_LINE = re.compile(r"^E\d+:\s*(.+)$")


def parse_query_group(path: Path) -> list[str]:
    events: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "**Answer":
            break
        match = EVENT_LINE.match(line.strip())
        if match:
            events.append(match.group(1).strip())
    if not events:
        raise ValueError(f"{path}: no E<n>: event lines found before **Answer")
    return events


def load_query_groups(query_dir: Path) -> list[tuple[str, list[str]]]:
    return [
        (filename.removesuffix(".txt"), parse_query_group(query_dir / filename))
        for filename in GROUP_FILES
    ]


def legacy_search(
    base_url: str,
    query: list[str],
    *,
    searcher_type: str,
    apply_boundary_refinement: bool,
) -> dict:
    response = requests.post(
        f"{base_url}/temporal-search",
        json={
            "query": query,
            "top_k_tuple": 20,
            "top_k_each_query": 50,
            "searcher_type": searcher_type,
            "apply_boundary_refinement": apply_boundary_refinement,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def adaptive_lifecycle(
    base_url: str, query: list[str], *, apply_boundary_refinement: bool
) -> dict:
    events = [
        {
            "event_id": f"e{index + 1}",
            "original_query": text,
            "anchor_query": text,
        }
        for index, text in enumerate(query)
    ]
    create = requests.post(
        f"{base_url}/v1/search-sessions", json={"events": events}, timeout=30
    )
    create.raise_for_status()
    session_id = create.json()["session"]["id"]
    try:
        retrieve = requests.post(
            f"{base_url}/v1/search-sessions/{session_id}/commands/retrieve",
            json={"top_k": 50},
            timeout=120,
        )
        retrieve.raise_for_status()
        priorities = requests.get(
            f"{base_url}/v1/search-sessions/{session_id}/video-priorities",
            params={"apply_boundary_refinement": str(apply_boundary_refinement).lower()},
            timeout=180,
        )
        priorities.raise_for_status()
        return {
            "retrieve_run": retrieve.json(),
            "video_priorities": priorities.json(),
        }
    finally:
        requests.delete(
            f"{base_url}/v1/search-sessions/{session_id}",
            params={"expected_revision": 0},
            timeout=30,
        )


def summarize_legacy(label: str, groups: list[tuple[str, dict]]) -> dict:
    status_counts: dict[str, int] = {}
    applied_diffs = []
    for _video_id, payload in groups:
        for result in payload["results"]:
            for candidate in result["tuple"]:
                status = candidate["boundary_refinement_status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "applied":
                    applied_diffs.append(candidate["refined_timestamp_seconds"])
    return {
        "label": label,
        "candidate_status_counts": status_counts,
        "applied_refined_seconds_sample": applied_diffs[:10],
    }


def summarize_adaptive(label: str, groups: list[tuple[str, dict]]) -> dict:
    status_counts: dict[str, int] = {}
    deltas = []
    capability_seen = None
    for _video_id, payload in groups:
        vp = payload["video_priorities"]
        capability_seen = vp["boundary_refinement_capability"]
        for item in vp["items"]:
            br = item["boundary_refinement"]
            status_counts[br["status"]] = status_counts.get(br["status"], 0) + 1
            for event in br["events"] or []:
                deltas.append(
                    {
                        "video_id": item["video_id"],
                        "event_id": event["event_id"],
                        "anchor_seconds": event["anchor_seconds"],
                        "refined_seconds": event["refined_seconds"],
                        "delta": event["refined_seconds"] - event["anchor_seconds"],
                    }
                )
    return {
        "label": label,
        "item_status_counts": status_counts,
        "capability": capability_seen,
        "event_deltas_sample": deltas[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--query-dir", type=Path, default=DEFAULT_QUERY_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("scratch/verify_migration_results.json")
    )
    parser.add_argument(
        "--skip-adaptive-refine-off",
        action="store_true",
        help="skip check 5 (adaptive_coarse, flag off) to save time",
    )
    args = parser.parse_args()

    groups = load_query_groups(args.query_dir)
    print(f"Loaded {len(groups)} real YouCook2 query groups: {[g[0] for g in groups]}")

    raw: dict[str, list[tuple[str, dict]]] = {
        "legacy_temporal_off": [],
        "legacy_temporal_on": [],
        "legacy_ambiguous_off": [],
        "legacy_ambiguous_on": [],
        "adaptive_coarse_off": [],
        "adaptive_coarse_on": [],
    }

    for video_id, query in groups:
        print(f"\n=== group {video_id} ({len(query)} events) ===")

        print("  legacy_temporal, flag off ...")
        raw["legacy_temporal_off"].append(
            (video_id, legacy_search(args.base_url, query, searcher_type="TemporalSearcher", apply_boundary_refinement=False))
        )
        print("  legacy_temporal, flag on ...")
        raw["legacy_temporal_on"].append(
            (video_id, legacy_search(args.base_url, query, searcher_type="TemporalSearcher", apply_boundary_refinement=True))
        )
        print("  legacy_ambiguous, flag off ...")
        raw["legacy_ambiguous_off"].append(
            (video_id, legacy_search(args.base_url, query, searcher_type="AmbiguousSearcher", apply_boundary_refinement=False))
        )
        print("  legacy_ambiguous, flag on ...")
        raw["legacy_ambiguous_on"].append(
            (video_id, legacy_search(args.base_url, query, searcher_type="AmbiguousSearcher", apply_boundary_refinement=True))
        )
        if not args.skip_adaptive_refine_off:
            print("  adaptive_coarse, flag off ...")
            raw["adaptive_coarse_off"].append(
                (video_id, adaptive_lifecycle(args.base_url, query, apply_boundary_refinement=False))
            )
        print("  adaptive_coarse, default (no query param -> on) ...")
        raw["adaptive_coarse_on"].append(
            (video_id, adaptive_lifecycle(args.base_url, query, apply_boundary_refinement=True))
        )

    summary = {
        "legacy_temporal_off": summarize_legacy("legacy_temporal_off", raw["legacy_temporal_off"]),
        "legacy_temporal_on": summarize_legacy("legacy_temporal_on", raw["legacy_temporal_on"]),
        "legacy_ambiguous_off": summarize_legacy("legacy_ambiguous_off", raw["legacy_ambiguous_off"]),
        "legacy_ambiguous_on": summarize_legacy("legacy_ambiguous_on", raw["legacy_ambiguous_on"]),
        "adaptive_coarse_on": summarize_adaptive("adaptive_coarse_on", raw["adaptive_coarse_on"]),
    }
    if not args.skip_adaptive_refine_off:
        summary["adaptive_coarse_off"] = summarize_adaptive(
            "adaptive_coarse_off", raw["adaptive_coarse_off"]
        )

    print("\n\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"raw": raw, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nFull raw + summary written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
