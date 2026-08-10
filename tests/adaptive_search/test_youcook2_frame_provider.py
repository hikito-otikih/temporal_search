from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_search.providers import (
    DecodedVideoFrame,
    FrameDecodeError,
    FrameProvider,
    FrameProviderConfigurationError,
    FrameRequestError,
    VideoAssetNotFoundError,
    YouCook2FrameProvider,
    _frame_pts_ms,
    _pyav_duration_ms,
)


class FakeDecoder:
    name = "fake-decoder"

    def __init__(self, *, available=True, actual_pts=None):
        self.available = available
        self.actual_pts = actual_pts or {}
        self.calls = []

    def availability(self):
        if self.available:
            return True, None
        return False, "synthetic decoder is unavailable"

    def decode(self, video_path, pts_ms):
        targets = list(pts_ms)
        self.calls.append((Path(video_path), targets))
        return [
            DecodedVideoFrame(
                pts_ms=self.actual_pts.get(target, target),
                frame_index=index,
                image=f"rgb-{target}",
            )
            for index, target in enumerate(targets)
        ]


class IncompleteDecoder(FakeDecoder):
    def decode(self, video_path, pts_ms):
        frames = super().decode(video_path, pts_ms)
        return frames[:-1]


class EmptyImageDecoder(FakeDecoder):
    def decode(self, video_path, pts_ms):
        frames = super().decode(video_path, pts_ms)
        first = frames[0]
        frames[0] = DecodedVideoFrame(
            pts_ms=first.pts_ms,
            frame_index=first.frame_index,
            image=None,
        )
        return frames


class YouCook2FrameProviderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temporary_directory.name)
        self.video_path = (
            self.dataset_root
            / "videos"
            / "training"
            / "101"
            / "video-a.mp4"
        )
        self.video_path.parent.mkdir(parents=True)
        self.video_path.write_bytes(b"synthetic video placeholder")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def provider(self, decoder=None, **kwargs):
        return YouCook2FrameProvider(
            self.dataset_root,
            decoder=decoder or FakeDecoder(),
            **kwargs,
        )

    def test_recursive_catalog_accepts_dataset_root_and_reports_capability(self):
        provider = self.provider()

        capability = provider.capabilities()

        self.assertTrue(capability.available)
        self.assertTrue(capability.supports_batch)
        self.assertTrue(capability.supports_dense_sampling)
        self.assertFalse(capability.supports_thumbnails)
        self.assertEqual(provider.catalog.asset_count, 1)
        self.assertEqual(provider.catalog.resolve("video-a"), self.video_path.resolve())
        self.assertIsInstance(provider, FrameProvider)

    def test_get_frames_sorts_requested_pts_and_deduplicates_actual_pts(self):
        decoder = FakeDecoder(actual_pts={80: 83, 81: 83})
        provider = self.provider(decoder)

        frames = provider.get_frames("video-a", [81, 0, 80, 0])

        self.assertEqual(decoder.calls[0][1], [0, 80, 81])
        self.assertEqual([frame.pts_ms for frame in frames], [0, 83])
        self.assertTrue(all(frame.image is not None for frame in frames))
        self.assertTrue(all(frame.video_id == "video-a" for frame in frames))

    def test_interval_grid_is_inclusive_when_end_is_not_aligned(self):
        decoder = FakeDecoder()
        provider = self.provider(decoder)

        frames = provider.sample_interval(
            "video-a", 0, 950, 300, max_frames=10
        )

        self.assertEqual([frame.pts_ms for frame in frames], [0, 300, 600, 900, 950])

    def test_budget_downsamples_evenly_over_the_whole_interval(self):
        provider = self.provider()

        frames = provider.sample_interval(
            "video-a", 0, 1000, 100, max_frames=3
        )

        self.assertEqual([frame.pts_ms for frame in frames], [0, 500, 1000])

    def test_single_frame_budget_samples_the_region_midpoint(self):
        provider = self.provider()

        frames = provider.sample_interval(
            "video-a", 0, 950, 100, max_frames=1
        )

        self.assertEqual([frame.pts_ms for frame in frames], [475])

    def test_provider_batch_ceiling_is_also_applied_to_interval_sampling(self):
        provider = self.provider(max_batch_frames=3)

        frames = provider.sample_interval(
            "video-a", 0, 1000, 100, max_frames=20
        )

        self.assertEqual([frame.pts_ms for frame in frames], [0, 500, 1000])
        with self.assertRaises(FrameRequestError):
            provider.get_frames("video-a", [0, 1, 2, 3])

    def test_metadata_rejects_pts_beyond_duration_before_decode(self):
        metadata_root = self.dataset_root / "metadata"
        metadata_root.mkdir()
        (metadata_root / "video-a_keyframes.json").write_text(
            json.dumps(
                {
                    "video_id": "video-a",
                    "duration": 1.25,
                    "fps": 25.0,
                    "frame_count": 32,
                }
            ),
            encoding="utf-8",
        )
        decoder = FakeDecoder()
        provider = self.provider(decoder, metadata_root=metadata_root)

        provider.get_frames("video-a", [1250])
        with self.assertRaises(FrameRequestError):
            provider.get_frames("video-a", [1251])

        self.assertEqual(len(decoder.calls), 1)

    def test_missing_root_and_decoder_are_reported_without_import_crash(self):
        missing = YouCook2FrameProvider(
            self.dataset_root / "missing",
            decoder=FakeDecoder(),
        )
        missing_capability = missing.capabilities()
        self.assertFalse(missing_capability.available)
        self.assertIn("does not exist", missing_capability.reason)

        unavailable_decoder = self.provider(FakeDecoder(available=False))
        decoder_capability = unavailable_decoder.capabilities()
        self.assertFalse(decoder_capability.available)
        self.assertIn("synthetic decoder", decoder_capability.reason)

    def test_duplicate_video_stems_make_catalog_explicitly_unavailable(self):
        duplicate = (
            self.dataset_root
            / "videos"
            / "validation"
            / "102"
            / "video-a.mkv"
        )
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(b"duplicate placeholder")

        provider = self.provider()

        capability = provider.capabilities()
        self.assertFalse(capability.available)
        self.assertIn("duplicate YouCook2 video ID", capability.reason)

    def test_video_id_is_never_concatenated_into_a_path(self):
        provider = self.provider()

        with self.assertRaises(VideoAssetNotFoundError):
            provider.get_frames("../video-a", [0])
        with self.assertRaises(VideoAssetNotFoundError):
            provider.get_frames("unknown", [0])

    def test_invalid_interval_values_fail_before_decoder_use(self):
        decoder = FakeDecoder()
        provider = self.provider(decoder)

        invalid_calls = [
            lambda: provider.sample_interval("video-a", -1, 10, 1, max_frames=2),
            lambda: provider.sample_interval("video-a", 10, 9, 1, max_frames=2),
            lambda: provider.sample_interval("video-a", 0, 10, 0, max_frames=2),
            lambda: provider.sample_interval("video-a", 0, 10, 1, max_frames=0),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(FrameRequestError):
                    call()
        self.assertEqual(decoder.calls, [])

    def test_decoder_partial_output_and_empty_images_are_rejected(self):
        with self.assertRaises(FrameDecodeError):
            self.provider(IncompleteDecoder()).get_frames("video-a", [0, 100])
        with self.assertRaises(FrameDecodeError):
            self.provider(EmptyImageDecoder()).get_frames("video-a", [0])

    def test_environment_factory_uses_configurable_roots(self):
        metadata_root = self.dataset_root / "metadata"
        metadata_root.mkdir()
        with patch.dict(
            os.environ,
            {
                "CUSTOM_YC_ROOT": str(self.dataset_root / "videos"),
                "CUSTOM_YC_METADATA": str(metadata_root),
            },
            clear=False,
        ):
            provider = YouCook2FrameProvider.from_environment(
                decoder=FakeDecoder(),
                data_root_env="CUSTOM_YC_ROOT",
                metadata_root_env="CUSTOM_YC_METADATA",
            )

        self.assertTrue(provider.capabilities().available)
        self.assertEqual(provider.catalog.resolve("video-a"), self.video_path.resolve())
    def test_fractional_pts_and_duration_helpers_keep_full_denominator(self):
        self.assertEqual(
            _frame_pts_ms(SimpleNamespace(pts=1), Fraction(1, 30)),
            33,
        )
        self.assertEqual(
            _frame_pts_ms(SimpleNamespace(pts=1), Fraction(1, 2000)),
            1,
        )
        with self.assertRaises(FrameDecodeError):
            _frame_pts_ms(SimpleNamespace(pts=None), Fraction(1, 30))

        stream = SimpleNamespace(duration=1, time_base=Fraction(1, 30))
        container = SimpleNamespace(duration=None)
        self.assertEqual(_pyav_duration_ms(container, stream), 34)

        stream_without_duration = SimpleNamespace(duration=None, time_base=None)
        microsecond_container = SimpleNamespace(duration=1_000_001)
        self.assertEqual(
            _pyav_duration_ms(microsecond_container, stream_without_duration),
            1001,
        )


if __name__ == "__main__":
    unittest.main()
