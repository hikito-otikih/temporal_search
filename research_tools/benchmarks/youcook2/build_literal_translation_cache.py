"""One-time, cached step: for each video's minimal_context VI text (prefix
pulled from common_query + original_query, unmodified - see
run_rewrite_pipeline_benchmark.py's _reconstruct_analysis_minimal_context),
get a LITERAL English translation - word-for-word meaning, no paraphrasing,
no elaboration, no added narrative framing. This is deliberately NOT the
existing retrieval_queries_en in rewrite_output_cache.jsonl, which is a
translation of a DIFFERENT (elaborately rewritten) VI text - it exists to
test a specific question: does a second, purely-translated variant help
recall/MRR beyond the single-VI-variant minimal_context result (measured:
recall@1 0.317, MRR 0.398, beating baseline on every metric), or does even
literal translation reintroduce the "any deviation from ground-truth
phrasing loses precision" problem measured earlier in this session?

Uses a simple, direct Ollama call (not rewrite_queries()'s full schema/
validation machinery) - just: given N Vietnamese sentences, return exactly
N literal English translations as a JSON array, same order.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rewrite.config import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_API_URL,
    OLLAMA_TIMEOUT_SECONDS,
    load_ollama_env,
)
from .core import load_query_directory_grouped  # noqa: E402
from .run_rewrite_pipeline_benchmark import _prefix_terms  # noqa: E402

TRANSLATE_SYSTEM_PROMPT = (
    "Bạn dịch câu tiếng Việt sang tiếng Anh. Dịch NGUYÊN VĂN, sát nghĩa "
    "từng từ nhất có thể - không diễn giải, không thêm bối cảnh, không thêm "
    "hoặc bớt thông tin, không viết lại thành câu tường thuật đầy đủ hơn. "
    "BẮT BUỘC dịch TOÀN BỘ câu sang tiếng Anh - tuyệt đối không được để sót "
    "bất kỳ từ tiếng Việt nào trong kết quả, kể cả tên món ăn/nguyên liệu "
    "không có bản dịch tiếng Anh chuẩn (ví dụ: dịch 'bánh xèo' thành "
    "'banh xeo pancake', không giữ nguyên 'bánh xèo'; dịch 'rau diếp' thành "
    "'lettuce', không bỏ sót). Nếu không chắc bản dịch tiếng Anh chính xác "
    "nhất, vẫn phải chọn một bản dịch hợp lý thay vì để nguyên tiếng Việt. "
    "Chỉ trả về đúng một JSON array of strings, cùng thứ tự và cùng số "
    "phần tử với input. Không markdown, không giải thích."
)


def _minimal_context_texts(cached_row: dict[str, Any], common_query: str | None) -> list[str]:
    prefix = ' '.join(_prefix_terms(common_query, 2))
    texts = []
    for event in cached_row["events"]:
        text = f"{prefix}, {event['original_query']}" if prefix else event['original_query']
        texts.append(text)
    return texts


async def _translate_one(client: httpx.AsyncClient, texts: list[str]) -> list[str]:
    payload = {
        "model": os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    response = await client.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    content = response.json()["message"]["content"].strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start, end = content.find("["), content.rfind("]")
    translations = json.loads(content[start : end + 1])
    if len(translations) != len(texts):
        raise ValueError(f"expected {len(texts)} translations, got {len(translations)}")
    return [str(t) for t in translations]


async def run(*, query_dir: Path, cache_path: Path, rewrite_cache_path: Path, video_limit: int) -> None:
    load_ollama_env()
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not configured")

    groups_by_id = {g.video_id: g for g in load_query_directory_grouped(query_dir)}
    rewrite_rows = [
        json.loads(line) for line in rewrite_cache_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rewrite_rows = [r for r in rewrite_rows if "error" not in r][:video_limit]
    print(f"{len(rewrite_rows)} videos to translate")

    done: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["video_id"]] = row
        print(f"Resuming: {len(done)} videos already cached")

    handle = cache_path.open("a", encoding="utf-8")
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    started = time.monotonic()
    completed = 0
    try:
        async with httpx.AsyncClient(headers=headers, timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            for index, rewrite_row in enumerate(rewrite_rows, 1):
                video_id = rewrite_row["video_id"]
                if video_id in done:
                    continue
                group = groups_by_id.get(video_id)
                if group is None:
                    continue
                vi_texts = _minimal_context_texts(rewrite_row, group.context)
                call_started = time.monotonic()
                try:
                    en_texts = await _translate_one(client, vi_texts)
                    row = {"video_id": video_id, "vi_texts": vi_texts, "en_texts": en_texts}
                except Exception as exc:  # noqa: BLE001
                    row = {"video_id": video_id, "vi_texts": vi_texts, "error": f"{type(exc).__name__}: {exc}"}
                row["_call_seconds"] = round(time.monotonic() - call_started, 2)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                if index % 5 == 0 or index == len(rewrite_rows):
                    print(f"  {index}/{len(rewrite_rows)} videos ({time.monotonic()-started:.0f}s elapsed)")
    finally:
        handle.close()
    print(f"Done. {completed} new rows written to {cache_path}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/query")
    parser.add_argument("--rewrite-cache", default="runs/rewrite_output_cache.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-limit", type=int, default=60)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    asyncio.run(run(
        query_dir=Path(args.query_dir), cache_path=Path(args.output),
        rewrite_cache_path=Path(args.rewrite_cache), video_limit=args.video_limit,
    ))


if __name__ == "__main__":
    main()
