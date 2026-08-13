import json
import time
import urllib.request

BASE = "http://localhost:8001/v1"

# Real Vietnamese text with diacritics, written via escapes to avoid editor mojibake risk.
COMMON_VI = (
    "Đoạn video múa lân một con lân màu vàng "
    "đen trắng, tìm các sự kiện sau:"
)
Q1 = (
    "Lân quay vòng trên cột số 4 bằng 2 chân "
    "trước rồi tiếp đất. Khoảnh khắc "
    "đầu tiên mà lân bắt đầu xoay vòng."
)
Q2 = (
    "Khoảnh khắc 4 chân hoàn toàn chạm đất "
    "đầu tiên."
)
Q3 = (
    "Khoảnh khắc đầu tiên 2 người biểu "
    "diễn lân cúi chào ban giám khảo."
)
Q4 = (
    "Sau đó lân tiến lại chào một con rồng. "
    "Khoảnh khắc đầu tiên con rồng cử động đầu."
)


def call(method, path, payload=None):
    url = BASE + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    payload = {"common_query": COMMON_VI, "query": [Q1, Q2, Q3, Q4]}
    print("=== creating session (LLM rewrite) ===")
    t0 = time.time()
    session = call("POST", "/search-sessions/from-queries", payload)
    print(f"elapsed: {time.time()-t0:.1f}s")
    session_id = session["session"]["id"]
    print("session_id:", session_id)
    out = {"session": session}

    print("=== retrieving (RRF fusion + clustering) ===")
    t0 = time.time()
    retrieve = call(
        "POST",
        f"/search-sessions/{session_id}/commands/retrieve",
        {"top_k": 50},
    )
    print(f"elapsed: {time.time()-t0:.1f}s")
    out["retrieve"] = retrieve

    print("=== regions ===")
    regions = call("GET", f"/search-sessions/{session_id}/regions")
    out["regions"] = regions

    print("=== video-priorities (no refinement) ===")
    priorities = call(
        "GET", f"/search-sessions/{session_id}/video-priorities?limit=10"
    )
    out["priorities"] = priorities

    print("=== video-priorities (with refinement) ===")
    t0 = time.time()
    priorities_refined = call(
        "GET",
        f"/search-sessions/{session_id}/video-priorities?limit=10&apply_boundary_refinement=true",
    )
    print(f"elapsed: {time.time()-t0:.1f}s")
    out["priorities_refined"] = priorities_refined

    with open("scratch/lion_dance_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote scratch/lion_dance_result.json")


if __name__ == "__main__":
    main()
