"""Executable dependency rules and stable artifact fingerprints."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


PIPELINE_STAGES = (
    "retrieval",
    "region",
    "refinement",
    "tuple",
)


# Longest-prefix match is used. SearchHyperparameters has exactly one field
# (tuple_ranking) - "retrieval"/"refinement"/"embedding"/"boundary" prefixes
# were removed alongside the SearchHyperparameters fields they used to
# match (RetrievalHyperparameters.retrieval, VideoPriorityHyperparameters's
# session-level refinement/embedding_model, BoundaryHyperparameters.boundary
# - none of these are reachable from a hyperparameters patch any more).
PARAMETER_STAGE_PREFIXES: dict[str, str] = {
    "tuple_ranking": "tuple",
}


EVENT_STAGE_PREFIXES: dict[str, str] = {
    "anchor_query": "retrieval",
    "boundary_type": "tuple",
    "temporal_relation": "tuple",
}


def changed_paths(before: Any, after: Any, prefix: str = "") -> set[str]:
    """Return leaf paths whose JSON-compatible values differ."""

    left = _to_plain(before)
    right = _to_plain(after)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right), key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return set()
        return {prefix or "$"}
    return set() if left == right else {prefix or "$"}


def invalidated_for_parameters(paths: Iterable[str]) -> list[str]:
    first_stages = [
        _stage_for_path(path, PARAMETER_STAGE_PREFIXES) for path in set(paths)
    ]
    return _downstream(first_stages)


def invalidated_for_event(paths: Iterable[str]) -> list[str]:
    normalized = {
        path.removeprefix("event.").split(".", 1)[0] for path in set(paths)
    }
    first_stages = [
        _stage_for_path(path, EVENT_STAGE_PREFIXES) for path in normalized
    ]
    return _downstream(first_stages)


def invalidated_for_constraint(command: str) -> list[str]:
    """Return MVP invalidation; live providers may prepend scoped refinement."""

    if command in {"mark_videos", "clear_allowed_videos"}:
        return ["refinement", "tuple"]
    if command in {"fix_frame", "clear_constraint"}:
        return ["tuple"]
    raise ValueError(f"unknown constraint command: {command}")


def artifact_fingerprint(
    *,
    artifact_type: str,
    inputs: Any,
    code_version: str,
) -> str:
    """Hash semantic inputs; callers should not include timestamps or run IDs."""

    payload = {
        "artifact_type": artifact_type,
        "code_version": code_version,
        "inputs": _to_plain(inputs),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _stage_for_path(path: str, mapping: Mapping[str, str]) -> str:
    matches = [
        (prefix, stage)
        for prefix, stage in mapping.items()
        if path == prefix or path.startswith(f"{prefix}.")
    ]
    if not matches:
        raise ValueError(f"no invalidation rule for path: {path}")
    return max(matches, key=lambda item: len(item[0]))[1]


def _downstream(first_stages: Iterable[str]) -> list[str]:
    stages = list(first_stages)
    if not stages:
        return []
    start = min(PIPELINE_STAGES.index(stage) for stage in stages)
    return list(PIPELINE_STAGES[start:])


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_plain(item) for item in value)
    return value
