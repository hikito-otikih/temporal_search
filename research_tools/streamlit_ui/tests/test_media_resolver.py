import json
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.media_resolver import MediaResolver


class MediaResolverNearestKeyframeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "metadata").mkdir()
        (self.root / "keyframes" / "video1").mkdir(parents=True)
        for name in ("video1_f000000.webp", "video1_f000150.webp", "video1_f000600.webp"):
            (self.root / "keyframes" / "video1" / name).write_bytes(b"")
        (self.root / "metadata" / "video1_keyframes.json").write_text(
            json.dumps(
                {
                    "video_id": "video1",
                    "keyframes": [
                        {"index": 0, "timestamp": 0.0, "file": "video1_f000000.webp"},
                        {"index": 150, "timestamp": 5.0, "file": "video1_f000150.webp"},
                        {"index": 600, "timestamp": 20.0, "file": "video1_f000600.webp"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.resolver = MediaResolver(self.root)

    def test_picks_the_closest_timestamp(self):
        path = self.resolver.resolve_keyframe_near("video1", 6.0)
        self.assertIsNotNone(path)
        self.assertEqual(path.name, "video1_f000150.webp")

    def test_exact_match(self):
        path = self.resolver.resolve_keyframe_near("video1", 20.0)
        self.assertEqual(path.name, "video1_f000600.webp")

    def test_missing_manifest_returns_none(self):
        self.assertIsNone(self.resolver.resolve_keyframe_near("unknown_video", 1.0))

    def test_unconfigured_root_returns_none(self):
        resolver = MediaResolver(None)
        self.assertIsNone(resolver.resolve_keyframe_near("video1", 1.0))

    def test_malformed_manifest_returns_none_not_raise(self):
        (self.root / "metadata" / "video2_keyframes.json").write_text("not json", encoding="utf-8")
        self.assertIsNone(self.resolver.resolve_keyframe_near("video2", 1.0))


if __name__ == "__main__":
    unittest.main()
