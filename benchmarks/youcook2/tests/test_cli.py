from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.youcook2 import cli


class _RecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.encoding_requested: str | None = None
        self.errors_requested: str | None = None

    def reconfigure(self, *, encoding: str, errors: str) -> None:
        self.encoding_requested = encoding
        self.errors_requested = errors


class CliEncodingTests(unittest.TestCase):
    def test_console_streams_are_reconfigured_to_utf8_when_supported(self) -> None:
        stdout = _RecordingStream()
        stderr = _RecordingStream()
        with patch.object(cli.sys, "stdout", stdout), patch.object(cli.sys, "stderr", stderr):
            cli._configure_utf8_console()
        self.assertEqual(stdout.encoding_requested, "utf-8")
        self.assertEqual(stderr.encoding_requested, "utf-8")
        self.assertEqual(stdout.errors_requested, "backslashreplace")

    def test_dry_run_supports_stringio_and_vietnamese(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            query_dir = Path(temporary)
            (query_dir / "video.txt").write_text(
                "Tìm sự kiện:\n"
                "E1: cắt hành tây\n"
                "**Answer\n"
                'video_path: "video.mp4"\n'
                "E1: 0:01 - 0:02\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.object(cli.sys, "stdout", stdout), patch.object(cli.sys, "stderr", stderr):
                exit_code = cli.main(["run", "--query-dir", str(query_dir), "--dry-run"])
        self.assertEqual(exit_code, 0)
        self.assertIn("cắt hành tây", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
