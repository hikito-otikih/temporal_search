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


print("### Legacy: search, then reject the top result (longer timeout) ###")
at2 = AppTest.from_file(PAGE, default_timeout=120)
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
    reject_buttons2[0].click().run(timeout=120)
    dump(at2, "after rejecting top legacy result")
    after2 = video_ids(at2)
    print("videos after reject:", after2)
    print("same length:", len(before2) == len(after2))
    print("rejected item's video not first anymore:", before2[0] != after2[0])
