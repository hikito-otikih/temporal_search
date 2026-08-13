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


at = AppTest.from_file(PAGE, default_timeout=180)
at.run()
at.text_area[0].set_value("cut an onion\nfry the onion in a pan").run()
at.radio[0].set_value("Adaptive (recommended)").run()
at.checkbox[0].set_value(True).run()  # "Refine timestamps" - needed to get event_id-bearing moments
at.button[0].click().run(timeout=180)
dump(at, "after search")

captions_before = [c.value for c in at.caption]
print("some moment captions:", [c for c in captions_before if "moment" in c][:6])

fix_buttons = [b for b in at.button if b.label == "Fix"]
print("Fix button count:", len(fix_buttons))
if not fix_buttons:
    print("NO FIX BUTTONS FOUND - aborting")
    sys.exit(1)

# Click the first "Fix this frame" button - should reveal the frame grid.
fix_buttons[0].click().run(timeout=60)
dump(at, "after clicking Fix (grid should be open)")

images = list(at.get("image"))
print("image count after opening grid:", len(images))

# Find the "Choose" buttons - they're labeled with a timestamp like "0:51.00",
# optionally with a " *" suffix on the center/anchor frame.
choose_buttons = [
    b for b in at.button
    if b.label not in ("Fix", "Search", "Not this one") and ":" in (b.label or "")
]
print("choose-frame button count:", len(choose_buttons))
if not choose_buttons:
    print("NO CHOOSE BUTTONS FOUND - aborting")
    sys.exit(1)

# Pick the center one (marked with " *") if present, else the first.
center = next((b for b in choose_buttons if b.label.endswith(" *")), choose_buttons[0])
print("picking frame labeled:", center.label)
center.click().run(timeout=180)
dump(at, "after choosing a frame")

captions_after = [c.value for c in at.caption]
fixed_captions = [c for c in captions_after if "[fixed]" in c]
print("captions containing [fixed]:", fixed_captions)
