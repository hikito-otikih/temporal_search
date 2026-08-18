"""UI smoke tests (static).

Pages are not executed here (executing them requires a Streamlit script
runtime), so the smoke checks are static:

1. every page imports and calls ``bootstrap()``;
2. widget ``key=`` values are unique within each page;
3. the entrypoint ``Home.py`` exists and is discoverable.

If Streamlit is installed, an additional import-level check is attempted.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"
HOME = Path(__file__).resolve().parent.parent / "Home.py"

WIDGET_FUNCS = {
    "button",
    "checkbox",
    "selectbox",
    "multiselect",
    "text_input",
    "number_input",
    "text_area",
    "radio",
    "slider",
    "select_slider",
    "color_picker",
    "file_uploader",
    "date_input",
    "time_input",
    "download_button",
    "toggle",
    "camera_input",
    "plotly_chart",
}

PAGE_FILES = sorted(PAGES_DIR.glob("*.py"))


class PageSmokeTests(unittest.TestCase):
    def test_home_exists(self):
        self.assertTrue(HOME.is_file())

    def test_pages_are_discovered(self):
        self.assertGreaterEqual(len(PAGE_FILES), 6, "expected at least six pages")

    def test_every_page_calls_bootstrap(self):
        for path in PAGE_FILES:
            with self.subTest(page=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn("bootstrap()", source)
                self.assertIn("from _bootstrap import", source)

    def test_widget_keys_are_unique_per_page(self):
        for path in [HOME, *PAGE_FILES]:
            with self.subTest(page=path.name):
                self._assert_unique_keys(path)

    @staticmethod
    def _assert_unique_keys(path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        seen: dict[str, str] = {}

        def visit_node(node: ast.AST) -> None:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "st"
                    and node.func.attr in WIDGET_FUNCS
                ):
                    for keyword in node.keywords:
                        if keyword.arg == "key" and isinstance(keyword.value, ast.Constant):
                            key = keyword.value.value
                            if not isinstance(key, str):
                                continue
                            location = f"{path.name}:{node.lineno}"
                            if key in seen:
                                raise AssertionError(
                                    f"duplicate widget key {key!r} at {location} "
                                    f"(first seen at {seen[key]})"
                                )
                            seen[key] = location
            for child in ast.iter_child_nodes(node):
                visit_node(child)

        visit_node(tree)

    def test_no_raw_requests_import_in_pages(self):
        for path in PAGE_FILES:
            with self.subTest(page=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("import requests", source)
                self.assertNotIn("import httpx", source)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "streamlit not installed")
class StreamlitInstalledChecks(unittest.TestCase):
    def test_streamlit_is_importable(self):
        import streamlit  # noqa: F401

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
