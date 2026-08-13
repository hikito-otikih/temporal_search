import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "irrelevant_things" / "streamlit_ui"))

from streamlit.testing.v1 import AppTest

PAGE = str(
    Path(__file__).resolve().parent.parent
    / "irrelevant_things"
    / "streamlit_ui"
    / "pages"
    / "00_Search.py"
)


def dump(at, label):
    print(f"\n=== {label} ===")
    if at.exception:
        for exc in at.exception:
            print("EXCEPTION:", exc.value if hasattr(exc, "value") else exc)
    for e in at.error:
        print("st.error:", e.value)


PASTED_BLOCK = (
    "Doan video huong dan nau cho ngo, tim cac su kien sau:\n"
    "E1: nhet xien vao con cho\n"
    "E2: danh mot qua trung va them no vao hon hop va danh deu de tron deu moi thu\n"
    "E3: chien cho trong chao dau\n"
    "E4: phuc vu banh ngo voi sot ca chua va mu tat"
)

at = AppTest.from_file(PAGE, default_timeout=180)
at.run()
at.text_area[0].set_value(PASTED_BLOCK).run()
at.radio[0].set_value("Adaptive (recommended)").run()
at.checkbox[0].set_value(True).run()
at.button[0].click().run(timeout=180)
dump(at, "after search")

captions = [c.value for c in at.caption]
not_found = [c for c in captions if "not found in this video" in c]
print("captions with 'not found in this video':", len(not_found))
print("sample:", not_found[:3])

fix_buttons_all = [b for b in at.button if b.label == "Fix"]
print("total Fix buttons (found + not-found combined):", len(fix_buttons_all))

if not not_found:
    print("No missing-event case surfaced in this run's top results - trying step navigation on a found moment instead.")
    fix_buttons_all[0].click().run(timeout=60)
    dump(at, "after opening a found-moment fixer")
else:
    # Find which button index corresponds to a missing-event row by matching
    # caption order isn't trivial via AppTest's flat element lists, so just
    # click the LAST Fix button (missing-event rows render after found ones
    # for each item) and confirm navigation works from whatever anchor it opens at.
    fix_buttons_all[-1].click().run(timeout=60)
    dump(at, "after opening a missing-event fixer")

# Exercise free-form navigation: step forward 10s twice, then jump to an
# arbitrary timestamp via the number input, and confirm the frame grid
# updates to the new center each time (not stuck at the original anchor).
back_buttons = [b for b in at.button if b.label == "+ 10s"]
print("'+ 10s' nav buttons found:", len(back_buttons))
if back_buttons:
    caption_before = [c.value for c in at.caption if "frames around" in c.value]
    print("frame-grid caption before nav:", caption_before[:1])
    back_buttons[0].click().run(timeout=60)
    caption_after = [c.value for c in at.caption if "frames around" in c.value]
    print("frame-grid caption after +10s:", caption_after[:1])
    print("center moved:", caption_before[:1] != caption_after[:1])

    number_inputs = list(at.number_input)
    jump_inputs = [n for n in number_inputs if "Jump to" in (n.label or "")]
    print("jump-to number inputs found:", len(jump_inputs))
    if jump_inputs:
        jump_inputs[0].set_value(120.0).run(timeout=60)
        caption_jumped = [c.value for c in at.caption if "frames around" in c.value]
        print("frame-grid caption after jump to 120s:", caption_jumped[:1])
