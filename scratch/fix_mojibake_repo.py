"""Auto-fix every mojibake spot found by scan_mojibake.py, repo-wide.

Confirmed reversal: corrupted_text.encode('cp1253').decode('utf-8') recovers
the original (the corruption was original_utf8_bytes decoded as cp1253).
Applied per maximal run of consecutive non-ASCII characters, same detection
as scan_mojibake.py, so only genuine round-trippable corruption is touched -
real Vietnamese/Greek/etc. text that doesn't happen to be valid cp1253->UTF-8
is left completely alone.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fix_run(run: str) -> str | None:
    try:
        original_bytes = run.encode("cp1253")
        decoded = original_bytes.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return None if decoded == run else decoded


def fix_line(line: str) -> tuple[str, int]:
    out = []
    run = ""
    fixes = 0

    def flush():
        nonlocal run
        if run:
            fixed = fix_run(run)
            out.append(fixed if fixed is not None else run)
            if fixed is not None:
                nonlocal fixes
                fixes += 1
            run = ""

    for c in line:
        if ord(c) > 127:
            run += c
        else:
            flush()
            out.append(c)
    flush()
    return "".join(out), fixes


def fix_text(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    total = 0
    fixed_lines = []
    for line in lines:
        # splitlines(keepends=True) keeps the newline as part of the ascii tail
        fixed, n = fix_line(line)
        fixed_lines.append(fixed)
        total += n
    return "".join(fixed_lines), total


def main():
    paths = [
        ROOT / line.strip()
        for line in Path("/tmp/tracked_text_files.txt").read_text().splitlines()
        if line.strip()
    ]
    total_files = 0
    total_fixes = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        fixed_text, n = fix_text(text)
        if n:
            path.write_text(fixed_text, encoding="utf-8")
            print(f"fixed {n} spot(s) in {path}")
            total_files += 1
            total_fixes += n
    print(f"\n{total_fixes} spot(s) fixed across {total_files} file(s)")


if __name__ == "__main__":
    main()
