import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "irrelevant_things" / "streamlit_ui"))

from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent.parent / "irrelevant_things" / "streamlit_ui" / "pages"


def dump(at, label):
    print(f"\n=== {label} ===")
    if at.exception:
        for exc in at.exception:
            print("EXCEPTION:", exc.value if hasattr(exc, "value") else exc)
    for e in at.error:
        print("st.error:", e.value)


print("### Search page: adaptive search, inspect thumbnails/video ###")
at = AppTest.from_file(str(BASE / "00_Search.py"), default_timeout=180)
at.run()
at.text_area[0].set_value("cut an onion\nfry the onion in a pan").run()
at.radio[0].set_value("Adaptive (recommended)").run()
at.button[0].click().run(timeout=180)
dump(at, "adaptive search")
captions = [c.value for c in at.caption]
print("no-thumbnail count:", sum(1 for c in captions if c == "no thumbnail"))
print("video-not-available count:", sum(1 for c in captions if c == "Video not available locally."))
print("total result captions:", len(captions))
print("images found:", len(at.get("image")))
for img in list(at.get("image"))[:3]:
    print("  image path:", getattr(img.proto, "url", None) or img.value)

print("\n### Developer Tools: 02_Adaptive_Session.py loads without crashing ###")
at02 = AppTest.from_file(str(BASE / "02_Adaptive_Session.py"), default_timeout=30)
at02.run()
dump(at02, "02 initial load")

print("\n### Developer Tools: 04_Tuple_Explorer.py loads without crashing ###")
at04 = AppTest.from_file(str(BASE / "04_Tuple_Explorer.py"), default_timeout=30)
at04.run()
dump(at04, "04 initial load (no active session)")
print("info:", [i.value for i in at04.info])

print("\n### Developer Tools: 01_Legacy_Search.py still works ###")
at01 = AppTest.from_file(str(BASE / "01_Legacy_Search.py"), default_timeout=30)
at01.run()
dump(at01, "01 initial load")
