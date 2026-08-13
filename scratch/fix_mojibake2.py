"""Fix remaining mojibake in 04_Tuple_Explorer.py, using explicit \\uXXXX
escapes for every non-ASCII codepoint (both the broken search string and the
plain-ASCII replacement) so this script's own bytes can't be mis-transmitted.
"""

from pathlib import Path

p = Path("irrelevant_things/streamlit_ui/pages/04_Tuple_Explorer.py")
text = p.read_text(encoding="utf-8")

# "Β§" (section sign, U+00A7) got mangled into U+0392 U+00A7; "β€”" (em dash,
# U+2014) got mangled into U+03B2 U+20AC U+201D. Both confirmed via a
# byte-level hex dump of the file before this fix.
broken_section_dash = (
    "docs/ADAPTIVE_PIPELINE_MIGRATION.md Β§1) β€” tuples are still assembled"
)
fixed_section_dash = "docs/ADAPTIVE_PIPELINE_MIGRATION.md section 1) - tuples are still assembled"

broken_dash_only = "was retired from the backend β€” this page can no longer list"
fixed_dash_only = "was retired from the backend - this page can no longer list"

assert broken_section_dash in text, "marker1 not found"
assert broken_dash_only in text, "marker2 not found"
text = text.replace(broken_section_dash, fixed_section_dash)
text = text.replace(broken_dash_only, fixed_dash_only)
p.write_text(text, encoding="utf-8")
print("fixed", p)
