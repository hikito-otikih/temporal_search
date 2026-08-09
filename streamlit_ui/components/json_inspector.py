"""JSON inspector and diagnostic export helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from services.export_service import json_bytes


def render_json_inspector(payload: Any, *, title: str = "Payload") -> None:
    st.subheader(title)
    st.json(payload)
    st.download_button(
        f"Download {title} JSON",
        data=json_bytes(payload),
        file_name=f"{_slugify(title)}.json",
        mime="application/json",
        key=f"dl_{_slugify(title)}",
    )


def render_sanitized_payload(payload: dict[str, Any]) -> None:
    st.caption("Sanitized retrieval payload — ground truth is never included.")
    st.code(__import__("json").dumps(payload, indent=2, ensure_ascii=False))


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower() or "payload"
