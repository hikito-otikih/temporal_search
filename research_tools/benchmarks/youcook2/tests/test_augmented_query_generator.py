from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.youcook2.augmented_query_generator import generate_augmented_queries
from benchmarks.youcook2.core import DatasetFormatError, load_query_directory_grouped


def _write_query_file(directory: Path, name: str, video_path: str) -> None:
    (directory / name).write_text(
        "Hướng dẫn nấu ăn, tìm các sự kiện sau:\n"
        "E1: cắt hành tây\n"
        "E2: chiên hành tây\n"
        "**Answer\n"
        f'video_path: "{video_path}"\n'
        "E1: 0:10 - 0:20\n"
        "E2: 0:30 - 0:40\n",
        encoding="utf-8",
    )


class AugmentedQueryGeneratorTests(unittest.TestCase):
    def test_round_trips_events_and_answers_into_first_and_last_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as output_dir:
            source = Path(source_dir)
            _write_query_file(source, "Video-A.txt", "Video-A.mp4")
            summary = generate_augmented_queries(source, output_dir)
            self.assertEqual(summary, {"video_count": 1})

            first_groups = load_query_directory_grouped(Path(output_dir) / "first")
            last_groups = load_query_directory_grouped(Path(output_dir) / "last")

        self.assertEqual(len(first_groups), 1)
        self.assertEqual(len(last_groups), 1)
        first, last = first_groups[0], last_groups[0]

        # Same video_id, same answers (interval reused verbatim), same event ids in order.
        self.assertEqual(first.video_id, "Video-A")
        self.assertEqual(last.video_id, "Video-A")
        self.assertEqual(first.answers, {"E1": (10.0, 20.0), "E2": (30.0, 40.0)})
        self.assertEqual(last.answers, first.answers)
        self.assertEqual([eid for eid, _ in first.events], ["E1", "E2"])
        self.assertEqual([eid for eid, _ in last.events], ["E1", "E2"])

        # Event text got wrapped, and first/last differ from each other.
        first_text = dict(first.events)["E1"]
        last_text = dict(last.events)["E1"]
        self.assertIn("cắt hành tây", first_text)
        self.assertIn("cắt hành tây", last_text)
        self.assertNotEqual(first_text, last_text)

    def test_same_video_first_and_last_variants_do_not_collide_within_a_directory(self) -> None:
        # The whole reason for separate first/ and last/ subdirectories:
        # load_query_directory_grouped rejects duplicate video_id within one dir.
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as output_dir:
            source = Path(source_dir)
            _write_query_file(source, "Video-A.txt", "Video-A.mp4")
            generate_augmented_queries(source, output_dir)

            # Sanity: if both variants were dumped into one flat directory it
            # would raise DatasetFormatError. Confirm that scenario really
            # would fail, so this test documents *why* the split matters.
            output = Path(output_dir)
            flat = output / "flat"
            flat.mkdir()
            for variant_dir in ("first", "last"):
                for file in (output / variant_dir).glob("*.txt"):
                    (flat / f"{variant_dir}_{file.name}").write_text(
                        file.read_text(encoding="utf-8"), encoding="utf-8"
                    )
            with self.assertRaises(DatasetFormatError):
                load_query_directory_grouped(flat)

    def test_multiple_videos_produce_one_file_pair_each(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as output_dir:
            source = Path(source_dir)
            _write_query_file(source, "one.txt", "Video-A.mp4")
            _write_query_file(source, "two.txt", "Video-B.mp4")
            summary = generate_augmented_queries(source, output_dir)
            self.assertEqual(summary, {"video_count": 2})

            first_files = sorted(p.name for p in (Path(output_dir) / "first").glob("*.txt"))
            last_files = sorted(p.name for p in (Path(output_dir) / "last").glob("*.txt"))
        self.assertEqual(first_files, ["Video-A__first.txt", "Video-B__first.txt"])
        self.assertEqual(last_files, ["Video-A__last.txt", "Video-B__last.txt"])


if __name__ == "__main__":
    unittest.main()
