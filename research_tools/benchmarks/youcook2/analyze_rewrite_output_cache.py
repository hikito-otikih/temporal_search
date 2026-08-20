"""Analyze runs/rewrite_output_cache.jsonl: does the new validator (py3langid
+ retrieval_queries_en_language self-report) catch anything the old
Vietnamese-signature-regex validator missed, on real model output - and does
it introduce any new false positives, on the same real text?
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/rewrite_output_cache.jsonl")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    path = Path(args.input)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    ok_rows = [r for r in rows if "error" not in r]
    error_rows = [r for r in rows if "error" in r]

    total_events = 0
    total_en_entries = 0
    old_flagged = 0
    new_langid_flagged = 0
    self_report_flagged = 0
    either_new_flagged = 0  # new_langid OR self_report
    agreements = 0
    disagreements = []
    call_seconds = [r.get("_call_seconds") for r in ok_rows if r.get("_call_seconds") is not None]

    for row in ok_rows:
        for event in row["events"]:
            total_events += 1
            for chk in event["retrieval_queries_en_checks"]:
                total_en_entries += 1
                old = chk["old_regex_flagged_not_en"]
                new_langid = chk["new_langid_flagged_not_en"]
                self_report_not_en = chk["self_report"] == "not_en"
                new_any = new_langid or self_report_not_en

                old_flagged += old
                new_langid_flagged += new_langid
                self_report_flagged += self_report_not_en
                either_new_flagged += new_any

                if old == new_any:
                    agreements += 1
                else:
                    disagreements.append(
                        {
                            "video_id": row["video_id"],
                            "event_id": event["event_id"],
                            "text": chk["text"],
                            "old_flagged": old,
                            "new_langid_flagged": new_langid,
                            "self_report": chk["self_report"],
                        }
                    )

    print(f"videos cached: {len(rows)} (ok: {len(ok_rows)}, errors: {len(error_rows)})")
    for r in error_rows:
        print(f"  ERROR video={r['video_id']}: {r['error']}")

    print(f"\ntotal events: {total_events}, total EN entries checked: {total_en_entries}")
    if call_seconds:
        print(
            f"per-video call time: min={min(call_seconds):.1f}s "
            f"max={max(call_seconds):.1f}s avg={sum(call_seconds)/len(call_seconds):.1f}s "
            f"total={sum(call_seconds):.0f}s"
        )

    print(f"\nold regex flagged not-English:      {old_flagged}/{total_en_entries}")
    print(f"new py3langid flagged not-English:   {new_langid_flagged}/{total_en_entries}")
    print(f"self-report flagged not_en:          {self_report_flagged}/{total_en_entries}")
    print(f"new (langid OR self-report) flagged: {either_new_flagged}/{total_en_entries}")
    print(f"old vs new agreement:                {agreements}/{total_en_entries}")

    if disagreements:
        print(f"\n=== {len(disagreements)} DISAGREEMENTS (old vs new) ===")
        for d in disagreements:
            direction = "NEW catches, OLD missed" if not d["old_flagged"] else "OLD flagged, NEW accepts"
            print(f"  [{direction}] video={d['video_id']} event={d['event_id']}")
            print(f"    text: {d['text']!r}")
            print(
                f"    old_flagged={d['old_flagged']} new_langid_flagged={d['new_langid_flagged']} "
                f"self_report={d['self_report']!r}"
            )
    else:
        print(
            "\nNo disagreements observed - every checked EN entry got the same "
            "verdict from both the old and new validators on this sample. This "
            "means the sample didn't happen to contain a case that exercises "
            "the difference; it does not by itself prove the two are "
            "equivalent (see the earlier 40-case synthetic benchmark for that "
            "comparison, and this session's live self-report test for a "
            "constructed case where they provably diverge)."
        )


if __name__ == "__main__":
    main()
