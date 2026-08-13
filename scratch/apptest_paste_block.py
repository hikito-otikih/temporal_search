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

PASTED_BLOCK = (
    "Đoạn video hướng dẫn nấu chó ngô, tìm các sự kiện sau:\n"
    "E1: nhét xiên vào con chó\n"
    "E2: đánh một quả trứng và thêm nó vào hỗn hợp và đánh đều để trộn đều mọi thứ\n"
    "E3: chiên chó trong chảo dầu\n"
    "E4: phục vụ bánh ngô với sốt cà chua và mù tạt"
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
at.text_area[0].set_value(PASTED_BLOCK).run()
print("captions after typing (before clicking Search):")
for c in at.caption:
    print(" ", repr(c.value))

at.radio[0].set_value("Adaptive (recommended)").run()
at.button[0].click().run(timeout=180)
dump(at, "after adaptive search with pasted block")
print("subheaders:", [s.value for s in at.subheader])
print("captions (first 6):", [c.value for c in at.caption][:6])
