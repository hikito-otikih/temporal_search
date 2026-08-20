"""Whole-pipeline comparison: does routing through the (now-fixed) rewrite
stage produce video rankings as good as feeding the adaptive pipeline
ground-truth event text directly?

Two runs over the SAME videos:
  - "baseline": _run_adaptive_coarse() as tuple_runner.py already does it -
    ground-truth event text straight into the adaptive session, no rewrite.
  - "rewrite": the SAME videos' events routed through the real rewrite
    stage first, via POST /v1/search-sessions/from-rewrite fed from the
    cached RewriteResponse in runs/rewrite_output_cache.jsonl (built by
    build_rewrite_output_cache.py) - zero additional LLM calls, reusing the
    expensive live Ollama pass already paid for.

Placeholder fields: the cache only stores original_query, target_moment_vi,
retrieval_queries_vi/en(+language), anchor_query, boundary, and
temporal_relation - not subject/action/visible_state/pre_state/post_state/
required_entities/soft_context/excluded_context/inferred_information/
ambiguities/video_context, which RewriteResponse's schema requires but
rewrite_bridge.build_session_plan() never reads (verified by grep: those
fields only feed boundary-refinement code, which _run_adaptive_coarse's
plain retrieve()+get_video_priorities() call never triggers). Reconstructed
here with clearly-labeled non-empty placeholders so the schema validates;
they do not participate in the coarse ranking being measured. If this
benchmark is ever extended to a boundary-refinement-aware pipeline, the
cache needs pre_state/post_state added for real - placeholders would matter
there.

Both runs use core.py's own compute_metrics() (recall@K, MRR, exactly as
tuple_runner.py's CLI does), so the numbers are directly comparable to any
other benchmark run through this package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .core import VideoQueryGroup, compute_metrics, load_query_directory_grouped
from .tuple_client import TemporalSearchBackendClient
from .tuple_runner import TupleRunConfig, _adaptive_hyperparameters, _build_event_payload, _rank_of_video

_PLACEHOLDER = "unspecified (not read by build_session_plan's coarse-ranking path)"

# YouCook2-specific annotation boilerplate, not general Vietnamese stopwords
# (so not added to rewrite.service.CONTEXT_STOPWORDS, which is production
# code used on arbitrary real common_query phrasing). Every video's
# common_query starts "Đoạn video hướng dẫn nấu <dish>, tìm các sự kiện
# sau:" - "đoạn"/"video" are already in CONTEXT_STOPWORDS, but "hướng",
# "dẫn", "nấu" are not, so _extract_context_terms(...)[:2] returned the
# literal constant ["hướng", "dẫn"] ("instructions") for every single one
# of 60 videos - zero discriminating power, never reaching the actual dish
# name ("hành tây chiên", "salad Waldorf", ...) the minimal_context
# hypothesis was actually about.
_DATASET_PREFIX_FILLER_WORDS = {"hướng", "dẫn", "nấu"}


def _prefix_terms(common_query: str | None, count: int) -> list[str]:
    from rewrite.service import _extract_context_terms

    terms = [t for t in _extract_context_terms(common_query) if t not in _DATASET_PREFIX_FILLER_WORDS]
    return terms[:count]


def _reconstruct_analysis(cached_row: dict[str, Any]) -> dict[str, Any]:
    events = []
    for event in cached_row["events"]:
        events.append(
            {
                "event_id": event["event_id"],
                "original_query": event["original_query"],
                "target_moment_vi": event["target_moment_vi"],
                "retrieval_queries_vi": event["retrieval_queries_vi"],
                "retrieval_queries_en": event["retrieval_queries_en"],
                "retrieval_queries_en_language": [
                    chk["self_report"] for chk in event["retrieval_queries_en_checks"]
                ],
                "subject": _PLACEHOLDER,
                "action": _PLACEHOLDER,
                "visible_state": _PLACEHOLDER,
                "anchor_query": event["anchor_query"],
                "pre_state": _PLACEHOLDER,
                "post_state": _PLACEHOLDER,
                "boundary": event["boundary"],
                "temporal_relation": event["temporal_relation"],
                "required_entities": [],
                "soft_context": [],
                "excluded_context": [],
                "inferred_information": [],
                "ambiguities": [],
            }
        )
    return {
        "video_context": {"scene": _PLACEHOLDER, "main_entities": [_PLACEHOLDER]},
        "events": events,
    }


def _reconstruct_analysis_single_variant(cached_row: dict[str, Any]) -> dict[str, Any]:
    """Same as _reconstruct_analysis, except every event's
    retrieval_queries_vi/en are forced to reduce to exactly ONE retrieval
    variant (anchor_query) once build_session_plan() dedupes them.

    Diagnostic-only construction, not a realistic rewrite response: sets
    every one of the 4 slots (2 "vi" + 2 "en") to the identical
    anchor_query string. build_session_plan() computes
    dict.fromkeys([*retrieval_queries_vi, *retrieval_queries_en]) - purely
    textual dedup, language-blind - so 4 identical strings collapse to the
    single variant baseline's _build_event_payload() also produces (which
    has no retrieval_variants at all, falling back to anchor_query alone).
    This isolates variant COUNT from rewrite TEXT QUALITY: comparing this
    condition against the baseline holds "one variant, whose text" as the
    only free variable, since both use the exact same anchor_query field
    the rewrite stage produced - just with fusion across 4 candidate
    variants removed."""
    analysis = _reconstruct_analysis(cached_row)
    for event in analysis["events"]:
        anchor = event["anchor_query"]
        event["retrieval_queries_vi"] = [anchor, anchor]
        event["retrieval_queries_en"] = [anchor, anchor]
        event["retrieval_queries_en_language"] = ["en", "en"]
    return analysis


def _reconstruct_analysis_minimal_context_bilingual(
    cached_row: dict[str, Any], translation_row: dict[str, Any], common_query: str | None,
    prefix_term_count: int = 2,
) -> dict[str, Any]:
    """Same minimal_context VI text as _reconstruct_analysis_minimal_context,
    plus a second variant: a LITERAL (not paraphrased) English translation
    of that exact VI text, from runs/literal_translation_cache.jsonl
    (build_literal_translation_cache.py) - not the existing
    retrieval_queries_en in rewrite_output_cache.jsonl, which translates a
    different (elaborately rewritten) VI text. Tests whether a second,
    purely-translated variant adds cross-lingual recall on top of the
    single-VI-variant result, or whether even literal translation
    reintroduces precision loss."""
    prefix = ' '.join(_prefix_terms(common_query, prefix_term_count))
    analysis = _reconstruct_analysis(cached_row)
    vi_texts = translation_row["vi_texts"]
    en_texts = translation_row["en_texts"]
    for i, event in enumerate(analysis["events"]):
        vi_text = f"{prefix}, {event['original_query']}" if prefix else event['original_query']
        assert vi_text == vi_texts[i], f"cache/translation VI text mismatch: {vi_text!r} != {vi_texts[i]!r}"
        en_text = en_texts[i]
        event["original_query"] = vi_text
        event["anchor_query"] = vi_text
        event["retrieval_queries_vi"] = [vi_text, vi_text]
        event["retrieval_queries_en"] = [en_text, en_text]
        event["retrieval_queries_en_language"] = ["en", "en"]
    return analysis


def _run_minimal_context_bilingual(
    backend: TemporalSearchBackendClient,
    group: VideoQueryGroup,
    cached_row: dict[str, Any],
    translation_row: dict[str, Any],
    config: TupleRunConfig,
) -> dict[str, Any]:
    analysis = _reconstruct_analysis_minimal_context_bilingual(cached_row, translation_row, group.context)
    session_id = backend.create_session_from_rewrite(analysis, common_query=group.context)
    backend.retrieve(session_id, top_k=config.adaptive_top_k)
    priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)
    rank = _rank_of_video(priorities, "video_id", group.video_id)
    return {
        "status": "ok", "rank": rank, "unique_video_count": len(priorities), "session_id": session_id,
    }


def _run_rewrite_single_variant(
    backend: TemporalSearchBackendClient,
    group: VideoQueryGroup,
    cached_row: dict[str, Any],
    config: TupleRunConfig,
) -> dict[str, Any]:
    analysis = _reconstruct_analysis_single_variant(cached_row)
    session_id = backend.create_session_from_rewrite(analysis, common_query=group.context)
    backend.retrieve(session_id, top_k=config.adaptive_top_k)
    priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)
    rank = _rank_of_video(priorities, "video_id", group.video_id)
    return {
        "status": "ok", "rank": rank, "unique_video_count": len(priorities), "session_id": session_id,
    }


def _reconstruct_analysis_context_aware(
    cached_row: dict[str, Any], common_query: str | None, prefix_term_count: int = 2
) -> dict[str, Any]:
    """Only inject the disambiguating prefix into events that actually
    lack it - leave already-sufficient events as bare original_query,
    byte-identical to baseline. Directly targets the mechanism the
    unconditional version got wrong: YouCook2's own event lines usually
    already name their dish/subject ("cắt hành tây..." already contains
    "hành tây"), so prepending the same term again is pure duplication that
    over-weights an already-present signal, pulling a video's distinct
    moments toward each other rather than disambiguating them - that's why
    the real-dish-name version measured *worse* than baseline. This uses
    rewrite.service._matched_context_terms (the same term-overlap check
    _validate_standalone_context already runs) to detect "does this event
    already contain the topic term" and only prefixes when it doesn't -
    exactly "Múa lân, Khoảnh khắc 4 chân..." for a query that's genuinely
    missing its subject, vs. leaving "cắt hành tây..." untouched since it
    already has one."""
    from rewrite.service import _matched_context_terms

    prefix_terms = _prefix_terms(common_query, prefix_term_count)
    prefix = ' '.join(prefix_terms)
    analysis = _reconstruct_analysis(cached_row)
    for event in analysis["events"]:
        original = event['original_query']
        already_sufficient = bool(prefix_terms) and bool(_matched_context_terms(original, prefix_terms))
        text = original if already_sufficient or not prefix else f"{prefix}, {original}"
        event["original_query"] = text
        event["anchor_query"] = text
        event["retrieval_queries_vi"] = [text, text]
        event["retrieval_queries_en"] = [text, text]
        event["retrieval_queries_en_language"] = ["en", "en"]
    return analysis


def _run_context_aware(
    backend: TemporalSearchBackendClient,
    group: VideoQueryGroup,
    cached_row: dict[str, Any],
    config: TupleRunConfig,
) -> dict[str, Any]:
    analysis = _reconstruct_analysis_context_aware(cached_row, group.context)
    session_id = backend.create_session_from_rewrite(analysis, common_query=group.context)
    backend.retrieve(session_id, top_k=config.adaptive_top_k)
    priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)
    rank = _rank_of_video(priorities, "video_id", group.video_id)
    return {
        "status": "ok", "rank": rank, "unique_video_count": len(priorities), "session_id": session_id,
    }


def _reconstruct_analysis_minimal_context(
    cached_row: dict[str, Any], common_query: str | None, prefix_term_count: int = 2
) -> dict[str, Any]:
    """original_query, byte-for-byte, with just a short disambiguating
    prefix pulled from common_query - not a paraphrase, not a full
    self-contained sentence. Tests a specific real-world hypothesis: a
    terse ground-truth-style query only needs to be self-contained when the
    query itself is ambiguous without video-level context (e.g. "the first
    moment all 4 legs touch the ground" could be any four-legged subject in
    any video) - YouCook2's own ground truth rarely has this problem
    (its lines already name their own subject/object), which is exactly
    why the earlier baseline comparison couldn't surface this effect. Uses
    rewrite.service._extract_context_terms - the same extraction already
    used to build the LLM prompt's REQUIRED_STANDALONE_CONTEXT block - as a
    mechanical stand-in for hand-picking a keyword like "Múa lân". See
    _DATASET_PREFIX_FILLER_WORDS for why raw _extract_context_terms output
    is filtered before use here."""
    prefix = ' '.join(_prefix_terms(common_query, prefix_term_count))
    analysis = _reconstruct_analysis(cached_row)
    for event in analysis["events"]:
        text = f"{prefix}, {event['original_query']}" if prefix else event['original_query']
        # original_query itself must also become `text` - build_session_plan()
        # unconditionally includes event.original_query as its own variant
        # (see that function's own comment), so leaving it as the bare
        # cache text here would silently fuse it back in alongside the
        # prefixed version - two variants, not the one this condition is
        # meant to test in isolation.
        event["original_query"] = text
        event["anchor_query"] = text
        event["retrieval_queries_vi"] = [text, text]
        event["retrieval_queries_en"] = [text, text]
        event["retrieval_queries_en_language"] = ["en", "en"]
    return analysis


def _run_minimal_context(
    backend: TemporalSearchBackendClient,
    group: VideoQueryGroup,
    cached_row: dict[str, Any],
    config: TupleRunConfig,
) -> dict[str, Any]:
    analysis = _reconstruct_analysis_minimal_context(cached_row, group.context)
    session_id = backend.create_session_from_rewrite(analysis, common_query=group.context)
    backend.retrieve(session_id, top_k=config.adaptive_top_k)
    priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)
    rank = _rank_of_video(priorities, "video_id", group.video_id)
    return {
        "status": "ok", "rank": rank, "unique_video_count": len(priorities), "session_id": session_id,
    }


def _run_baseline(
    backend: TemporalSearchBackendClient, group: VideoQueryGroup, config: TupleRunConfig
) -> dict[str, Any]:
    session_id = backend.create_adaptive_session(
        _build_event_payload(group), common_query=group.context,
        hyperparameters=_adaptive_hyperparameters(config),
    )
    backend.retrieve(session_id, top_k=config.adaptive_top_k)
    priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)
    rank = _rank_of_video(priorities, "video_id", group.video_id)
    return {
        "status": "ok", "rank": rank, "unique_video_count": len(priorities), "session_id": session_id,
    }


def _run_rewrite(
    backend: TemporalSearchBackendClient,
    group: VideoQueryGroup,
    cached_row: dict[str, Any],
    config: TupleRunConfig,
) -> dict[str, Any]:
    analysis = _reconstruct_analysis(cached_row)
    session_id = backend.create_session_from_rewrite(analysis, common_query=group.context)
    backend.retrieve(session_id, top_k=config.adaptive_top_k)
    priorities = backend.get_video_priorities(session_id, limit=config.adaptive_ranking_top_k)
    rank = _rank_of_video(priorities, "video_id", group.video_id)
    return {
        "status": "ok", "rank": rank, "unique_video_count": len(priorities), "session_id": session_id,
    }


def run(
    *, query_dir: Path, cache_path: Path, base_url: str, recall_ks: tuple[int, ...],
    limit: int | None = None, conditions: tuple[str, ...] = ("baseline", "rewrite"),
    translation_cache_path: Path | None = None,
    top_k: int = 20, top_n_per_variant: int | None = None, ranking_top_k: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    groups_by_id = {g.video_id: g for g in load_query_directory_grouped(query_dir)}
    cached_rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cached_rows = [row for row in cached_rows if "error" not in row]
    translations_by_id: dict[str, dict[str, Any]] = {}
    if "minimal_context_bilingual" in conditions:
        if translation_cache_path is None:
            raise ValueError("minimal_context_bilingual requires --translation-cache")
        for line in translation_cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    translations_by_id[row["video_id"]] = row
    if limit is not None:
        cached_rows = cached_rows[:limit]
    print(
        f"{len(cached_rows)} cached rewrite rows (errors excluded); conditions={conditions}; "
        f"top_k={top_k} top_n_per_variant={top_n_per_variant} ranking_top_k={ranking_top_k}"
    )

    config = TupleRunConfig(
        backend_base_url=base_url, pipeline="adaptive_coarse", top_k_tuple=100,
        top_k_each_query=100, gamma=0.05, adaptive_top_k=top_k, recall_ks=recall_ks,
        timeout_seconds=120.0, retries=2, retry_backoff_seconds=0.5,
        adaptive_ranking_top_k=ranking_top_k, adaptive_top_n_per_variant=top_n_per_variant,
    )
    backend = TemporalSearchBackendClient(base_url=base_url, timeout_seconds=config.timeout_seconds, retries=config.retries)

    rows: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
    started = time.monotonic()
    for index, cached_row in enumerate(cached_rows, 1):
        video_id = cached_row["video_id"]
        group = groups_by_id.get(video_id)
        if group is None:
            print(f"  skip {video_id}: not found in query_dir")
            continue
        translation_row = translations_by_id.get(video_id)
        for condition, runner in (
            ("baseline", lambda: _run_baseline(backend, group, config)),
            ("rewrite", lambda: _run_rewrite(backend, group, cached_row, config)),
            ("rewrite_single_variant", lambda: _run_rewrite_single_variant(backend, group, cached_row, config)),
            ("minimal_context", lambda: _run_minimal_context(backend, group, cached_row, config)),
            (
                "minimal_context_bilingual",
                lambda: _run_minimal_context_bilingual(backend, group, cached_row, translation_row, config),
            ),
            ("context_aware", lambda: _run_context_aware(backend, group, cached_row, config)),
        ):
            if condition not in conditions:
                continue
            if condition == "minimal_context_bilingual" and translation_row is None:
                print(f"  skip {video_id}: no translation cached")
                continue
            try:
                rows[condition].append(runner())
            except Exception as exc:  # noqa: BLE001
                rows[condition].append({"status": "error", "rank": None, "error": f"{type(exc).__name__}: {exc}"})
        if index % 10 == 0 or index == len(cached_rows):
            print(f"  {index}/{len(cached_rows)} videos evaluated ({time.monotonic()-started:.0f}s elapsed)")

    metrics = {c: compute_metrics(rows[c], recall_ks) for c in conditions}
    for condition in conditions:
        print(f"\n=== {condition} ===")
        print(json.dumps(metrics[condition], indent=2))

    print("\n=== SIDE BY SIDE ===")
    for k in sorted(recall_ks):
        key = f"recall_at_{k}"
        line = "  ".join(f"{c}={metrics[c][key]!s:6}" for c in conditions)
        print(f"  recall@{k:<3d} {line}")
    line = "  ".join(f"{c}={metrics[c]['mrr']!s:6}" for c in conditions)
    print(f"  mrr       {line}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default="/mnt/c/Users/huynh/Downloads/youcook2/query")
    parser.add_argument("--cache", default="runs/rewrite_output_cache.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--recall-k", default="1,5,10,20,50")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--translation-cache", default="runs/literal_translation_cache.jsonl")
    parser.add_argument(
        "--conditions", default="baseline,rewrite",
        help=(
            "comma-separated subset of: baseline, rewrite, rewrite_single_variant, "
            "minimal_context, minimal_context_bilingual, context_aware"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20, help="network-level top_k per retrieval variant")
    parser.add_argument("--top-n-per-variant", type=int, default=None, help="None keeps the server default")
    parser.add_argument("--ranking-top-k", type=int, default=None, help="video-priorities limit; None keeps the server default (100)")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run(
        query_dir=Path(args.query_dir), cache_path=Path(args.cache), base_url=args.base_url,
        recall_ks=tuple(int(k) for k in args.recall_k.split(",")), limit=args.limit,
        conditions=tuple(args.conditions.split(",")),
        translation_cache_path=Path(args.translation_cache),
        top_k=args.top_k, top_n_per_variant=args.top_n_per_variant, ranking_top_k=args.ranking_top_k,
    )


if __name__ == "__main__":
    main()
