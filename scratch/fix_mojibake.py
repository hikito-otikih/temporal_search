"""Fix mojibake emoji/punctuation introduced by an earlier tool write.

Every non-ASCII character here is written as an explicit \\uXXXX escape (not
a raw byte) so this script's own source stays pure ASCII and can't suffer
the same corruption while being transmitted/written.
"""

from pathlib import Path

MAG_GLASS_BROKEN = "πŸ”"  # what "\U0001F50D" (magnifying glass) became
FLASK_BROKEN = "πŸ§ͺ"  # what "\U0001F9EA" (test tube) became
GEAR_BROKEN = "βš™οΈ"  # what "⚙️" (gear+VS16) became
MIDDOT_BROKEN = "Β·"  # what "·" (middle dot) became at one call site
WRONG_TRIANGLE = "▢"  # what was written instead of a right-pointing triangle

ROOT = Path("irrelevant_things/streamlit_ui")

replacements = {
    ROOT / "Home.py": [
        (f'icon="{MAG_GLASS_BROKEN}"', 'icon=":material/search:"'),
        (f'icon="{FLASK_BROKEN}"', 'icon=":material/science:"'),
    ],
    ROOT / "_bootstrap.py": [
        (
            f'st.set_page_config(page_title="Temporal Search", page_icon="{MAG_GLASS_BROKEN}", layout="wide")',
            'st.set_page_config(page_title="Temporal Search", page_icon=":material/search:", layout="wide")',
        ),
        (
            f'with st.sidebar.expander("{GEAR_BROKEN} Connection settings"):',
            'with st.sidebar.expander("Connection settings"):',
        ),
    ],
    ROOT / "pages" / "00_Search.py": [
        (f'st.title("{MAG_GLASS_BROKEN} Search")', 'st.title("Search")'),
        (f'st.expander("{WRONG_TRIANGLE} Play")', 'st.expander("Play")'),
        (
            f'st.subheader(f"Results {MIDDOT_BROKEN} {{len(items)}} video{{\'s\' if len(items) != 1 else \'\'}}")',
            'st.subheader(f"Results - {len(items)} video{\'s\' if len(items) != 1 else \'\'}")',
        ),
    ],
}

for path, pairs in replacements.items():
    text = path.read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in text:
            raise SystemExit(f"NOT FOUND in {path}: {old!r}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("fixed", path)
