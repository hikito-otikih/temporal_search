"""Ad hoc AppTest-based verification for pages/00_Search.py against the real,
already-running backend (localhost:8001). Not a browser, but a real headless
Streamlit script run with real widget interaction and real HTTP calls -
closer to true verification than the static smoke tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "irrelevant_things" / "streamlit_ui"))

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parent.parent / "irrelevant_things" / "streamlit_ui" / "pages" / "00_Search.py")


def dump(at, label):
    print(f"\n=== {label} ===")
    if at.exception:
        for exc in at.exception:
            print("EXCEPTION:", exc.value if hasattr(exc, "value") else exc)
    for e in at.error:
        print("st.error:", e.value)
    for w in at.warning:
        print("st.warning:", w.value)
    for i in at.info:
        print("st.info:", i.value)


print("### Test 1: adaptive_coarse search ###")
at = AppTest.from_file(PAGE, default_timeout=120)
at.run()
dump(at, "initial load")
at.text_area[0].set_value("cut an onion\nfry the onion in a pan").run()
at.radio[0].set_value("Adaptive (recommended)").run()
at.button[0].click().run(timeout=180)
dump(at, "after adaptive search")
print("subheaders:", [s.value for s in at.subheader])
print("markdowns (first 5):", [m.value for m in at.markdown][:5])
print("captions (first 10):", [c.value for c in at.caption][:10])

print("\n### Test 2: legacy ordered search ###")
at2 = AppTest.from_file(PAGE, default_timeout=120)
at2.run()
at2.text_area[0].set_value("cut an onion").run()
at2.radio[0].set_value("Legacy — ordered").run()
at2.button[0].click().run(timeout=60)
dump(at2, "after legacy search")
print("subheaders:", [s.value for s in at2.subheader])
print("captions (first 10):", [c.value for c in at2.caption][:10])

print("\n### Test 3: empty query validation ###")
at3 = AppTest.from_file(PAGE, default_timeout=30)
at3.run()
at3.button[0].click().run()
dump(at3, "after empty search")
