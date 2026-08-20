"""One-time, cached step: run the real LLM rewrite for each sample video's
event list and record the FULL RewriteResponse (not just temporal_relation,
unlike build_temporal_relations_cache.py's narrower cache) - retrieval_
queries_en/vi included, since those are exactly what rewrite_bridge.py turns
into the retrieval variants actually sent to search. Expensive (one real
Ollama call per video) - cached to JSONL so every downstream sweep (a
whole-pipeline benchmark, an old-vs-new validator comparison, ...) reuses it
for free, same "fetch once, sweep cheaply" pattern as
build_temporal_relations_cache.py and region_tuple_experiment.py.

Also records, per retrieval_queries_en entry, a RETROSPECTIVE evaluation
against the OLD (pre-py3langid) Vietnamese-signature-regex validator this
session replaced - reimplemented standalone here from the exact code this
session wrote and then replaced (verified against the conversation's own
edit history, not re-derived from memory), purely for comparison. It does
not affect what gets accepted into the cache; only the live service's
current validator (py3langid + retrieval_queries_en_language self-report)
does that. This lets a later analysis answer "would the old validator have
missed something the new one catches, on real model output" without paying
for a second live LLM pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from .core import load_query_directory_grouped
from rewrite.service import _is_english, rewrite_queries

# Exact old check this session replaced (src/rewrite/constants.py /
# service.py before the py3langid swap) - kept standalone here, not
# imported, since production code no longer has it.
_OLD_VIETNAMESE_SIGNATURE_PATTERN = re.compile('[Ḁ-ỿĂăĐđƠơƯư]')
_OLD_VIETNAMESE_WORD_FRACTION_THRESHOLD = 0.5


def _old_looks_untranslated_to_vietnamese(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    flagged = sum(1 for word in words if _OLD_VIETNAMESE_SIGNATURE_PATTERN.search(word))
    return flagged / len(words) > _OLD_VIETNAMESE_WORD_FRACTION_THRESHOLD


async def _build_one(video_id: str, queries: list[str], common_query: str | None) -> dict[str, Any]:
    analysis = await rewrite_queries(queries=queries, common_query=common_query)
    events = []
    for event in analysis.events:
        en_checks = []
        for en_query, self_report in zip(
            event.retrieval_queries_en, event.retrieval_queries_en_language
        ):
            en_checks.append(
                {
                    "text": en_query,
                    "old_regex_flagged_not_en": _old_looks_untranslated_to_vietnamese(en_query),
                    "new_langid_flagged_not_en": not _is_english(en_query),
                    "self_report": self_report,
                }
            )
        events.append(
            {
                "event_id": event.event_id,
                "original_query": event.original_query,
                "target_moment_vi": event.target_moment_vi,
                "retrieval_queries_vi": event.retrieval_queries_vi,
                "retrieval_queries_en": event.retrieval_queries_en,
                "retrieval_queries_en_checks": en_checks,
                "anchor_query": event.anchor_query,
                "boundary": event.boundary,
                "temporal_relation": {
                    "relation": event.temporal_relation.relation,
                    "reference_event_id": event.temporal_relation.reference_event_id,
                },
            }
        )
    return {"video_id": video_id, "events": events}


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
            call_started = time.monotonic()
            try:
                row = await _build_one(group.video_id, queries, group.context)
            except Exception as exc:  # noqa: BLE001 - record failure, keep going
                row = {"video_id": group.video_id, "error": f"{type(exc).__name__}: {exc}"}
            call_elapsed = time.monotonic() - call_started
            row["_call_seconds"] = round(call_elapsed, 2)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            if index % progress_every == 0 or index == len(groups):
                elapsed = time.monotonic() - started
                print(
                    f"  {index}/{len(groups)} videos processed ({elapsed:.0f}s elapsed, "
                    f"last call {call_elapsed:.1f}s)"
                )
    finally:
        handle.close()
    print(f"Done. {completed} new rows written to {output_path}.")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/query")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-limit", type=int, default=15)
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
