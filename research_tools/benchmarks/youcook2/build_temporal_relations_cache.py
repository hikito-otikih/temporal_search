"""One-time, cached step: run the real LLM rewrite for each sample video's
event list and record each event's `temporal_relation` (sequence_start /
after / before / during / simultaneous / independent / unknown) plus
`reference_event_id`. Expensive (one real Ollama call per video, ~90-160s
each observed this session) - cached to JSONL so every downstream sweep
reuses it for free, same "fetch once, sweep cheaply" pattern as candidate
fetching in region_tuple_experiment.py.

Note: `EventDefinition` (what the live session actually carries) drops this
field entirely at the rewrite_bridge.py boundary - it is not available
anywhere else in the pipeline today. This script calls the rewrite service
directly to recover it for benchmarking, matching the exact real
classification a live session would have gotten, not a heuristic guess.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from .core import load_query_directory_grouped
from rewrite.service import rewrite_queries


async def _build_one(video_id: str, queries: list[str], common_query: str | None) -> dict[str, Any]:
    analysis = await rewrite_queries(queries=queries, common_query=common_query)
    return {
        "video_id": video_id,
        "events": [
            {
                "event_id": event.event_id,
                "relation": event.temporal_relation.relation,
                "reference_event_id": event.temporal_relation.reference_event_id,
            }
            for event in analysis.events
        ],
    }


async def run(*, query_dir: Path, output_path: Path, video_limit: int, progress_every: int) -> None:
    groups = load_query_directory_grouped(query_dir)[:video_limit]
    print(f"Loaded {len(groups)} video query groups from {query_dir}")

    done: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["video_id"]] = row
        print(f"Resuming: {len(done)} videos already cached")

    handle = output_path.open("a", encoding="utf-8")
    started = time.monotonic()
    completed = 0
    try:
        for index, group in enumerate(groups, 1):
            if group.video_id in done:
                continue
            queries = [text for _, text in group.events]
            try:
                row = await _build_one(group.video_id, queries, group.context)
            except Exception as exc:  # noqa: BLE001 - record failure, keep going
                row = {"video_id": group.video_id, "error": f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            if index % progress_every == 0 or index == len(groups):
                elapsed = time.monotonic() - started
                print(f"  {index}/{len(groups)} videos processed ({elapsed:.0f}s elapsed)")
    finally:
        handle.close()
    print(f"Done. {completed} new rows written to {output_path}.")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/query")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-limit", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    asyncio.run(run(
        query_dir=Path(args.query_dir),
        output_path=Path(args.output),
        video_limit=args.video_limit,
        progress_every=args.progress_every,
    ))


if __name__ == "__main__":
    main()
