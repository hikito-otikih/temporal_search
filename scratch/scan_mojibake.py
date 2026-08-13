"""Repo-wide mojibake scanner.

Forensic test, not a heuristic: for every maximal run of consecutive
non-ASCII characters in a line, try encoding it as cp1252 and decoding the
result as UTF-8. Genuine text (Vietnamese diacritics, em dashes, etc.) either
can't be encoded as cp1252 at all, or doesn't happen to form valid UTF-8
when reinterpreted - so a clean round-trip is strong evidence of exactly the
"UTF-8 bytes misread as cp1252, then re-saved as UTF-8" corruption pattern
that hit pages/00_Search.py and pages/04_Tuple_Explorer.py earlier.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE_LIST = ROOT / "tmp_tracked_text_files.txt"


def find_mojibake_runs(text: str):
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        run = ""
        run_start = 0
        for i, c in enumerate(line):
            if ord(c) > 127:
                if not run:
                    run_start = i
                run += c
            else:
                if run:
                    hit = _check_run(run)
                    if hit:
                        hits.append((lineno, run_start, run, hit))
                run = ""
        if run:
            hit = _check_run(run)
            if hit:
                hits.append((lineno, run_start, run, hit))
    return hits


def _check_run(run: str):
    # The actual corruption mechanism (confirmed empirically): correct UTF-8
    # bytes get misdecoded as Windows-1253 (Greek) somewhere in this tool's
    # write path, then the garbled result is saved back as UTF-8. So the
    # reverse test is: encode the *suspect* text back to UTF-8 bytes, decode
    # those bytes as cp1253, and see if that reproduces the run (i.e. the
    # run's own UTF-8 bytes, reinterpreted as cp1253, equal itself - meaning
    # it's a fixed point of the corruption, not proof either way) - instead,
    # the real test is the forward direction: does *this* run equal what you
    # get by taking some plausible original character's UTF-8 bytes through
    # cp1253? We can't invert cp1253->UTF-8 bytes->UTF-8 text directly since
    # cp1253-encoding the run and UTF-8-decoding it is the actual reverse of
    # the corruption (corruption was UTF8-bytes -decoded-as-> cp1253; reverse
    # is cp1253-encode the corrupted text back to bytes, then UTF-8-decode).
    try:
        original_bytes = run.encode("cp1253")
        decoded = original_bytes.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if decoded == run:
        return None
    return decoded


def main():
    paths = [
        ROOT / line.strip()
        for line in Path("/tmp/tracked_text_files.txt").read_text().splitlines()
        if line.strip()
    ]
    total_hits = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            print(f"SKIP (read error) {path}: {exc}")
            continue
        hits = find_mojibake_runs(text)
        for lineno, col, run, decoded in hits:
            total_hits += 1
            print(f"{path}:{lineno} col{col}  {run!r} -> should be {decoded!r}")
    print(f"\nTotal likely-mojibake spots: {total_hits}")


if __name__ == "__main__":
    main()
