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


def video_ids(at):
    return [m.value for m in at.markdown if m.value.startswith("**") and m.value.endswith("**")]


print("### Adaptive: search, then reject the top result ###")
at = AppTest.from_file(PAGE, default_timeout=120)
at.run()
at.text_area[0].set_value("cut an onion\nfry the onion in a pan").run()
at.radio[0].set_value("Adaptive (recommended)").run()
at.button[0].click().run(timeout=120)
dump(at, "after search")
before = video_ids(at)
print("videos before reject:", before)
print("subheaders before:", [s.value for s in at.subheader])

# The reject buttons are all labeled "Not this one" - the first one
# corresponds to the first (top-ranked) result.
reject_buttons = [b for b in at.button if b.label == "Not this one"]
print("reject button count:", len(reject_buttons))
reject_buttons[0].click().run(timeout=60)
dump(at, "after rejecting top result")
after = video_ids(at)
print("videos after reject:", after)
print("subheaders after:", [s.value for s in at.subheader])
print("top result changed:", before[0] != after[0] if before and after else "N/A")
print("same count:", len(before) == len(after))
print("rejected video no longer present:", before[0] not in after if before and after else "N/A")


print("\n### Legacy: search, then reject the top result ###")
at2 = AppTest.from_file(PAGE, default_timeout=60)
at2.run()
at2.text_area[0].set_value("cut an onion").run()
at2.radio[0].set_value("Legacy — ordered").run()
at2.button[0].click().run(timeout=60)
dump(at2, "after legacy search")
before2 = video_ids(at2)
print("videos before reject:", before2)

reject_buttons2 = [b for b in at2.button if b.label == "Not this one"]
print("reject button count:", len(reject_buttons2))
if reject_buttons2:
    reject_buttons2[0].click().run(timeout=30)
    dump(at2, "after rejecting top legacy result")
    after2 = video_ids(at2)
    print("videos after reject:", after2)
    print("rejected item no longer first:", before2[0] not in after2[:1] if before2 and after2 else "N/A")
