"""Reproducibility bundle and CSV/JSON export helpers.

``build_diagnostic_bundle`` deliberately excludes secrets: it never reads the
environment, only the sanitized values the UI has already stored in
``st.session_state`` (backend URL, upstream URL, workspace, capability flags).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Iterable


def json_bytes(payload: Any, *, indent: int = 2) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=indent, default=str).encode("utf-8")


def csv_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    rows = list(rows)
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_diagnostic_bundle(
    *,
    app_version: str,
    backend_url: str,
    upstream_url: str,
    workspace: str,
    capabilities: dict[str, Any],
    session_id: str | None,
    session_revision: int | None,
    hyperparameters: dict[str, Any] | None,
    artifacts: dict[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    """Build a safe, secret-free reproducibility manifest."""
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "app_version": app_version,
        "backend_url": backend_url,
        "upstream_url": upstream_url,
        "workspace": workspace,
        "capabilities": capabilities,
        "session": {
            "id": session_id,
            "revision": session_revision,
        },
        "hyperparameters": hyperparameters or {},
        "artifact_counts": artifacts,
        "researcher_note": note,
    }


def build_run_manifest(
    *,
    run_summary: dict[str, Any],
    per_query: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_manifest_version": 1,
        "summary": run_summary,
        "config": config,
        "per_query": per_query,
    }
