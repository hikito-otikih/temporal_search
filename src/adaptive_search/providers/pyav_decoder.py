"""PyAV-backed local video decoder - lazily imports `av` so importing the
API never pulls in or downloads the optional multimedia runtime."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from .base import DecodedVideoFrame, FrameDecodeError, FrameProviderConfigurationError, FrameRequestError


class PyAVFrameDecoder:
    """Lazy decoder that preserves real stream presentation timestamps."""

    @property
    def name(self) -> str:
        return "pyav"

    def availability(self) -> tuple[bool, str | None]:
        try:
            self._import_av()
        except FrameProviderConfigurationError as exc:
            return False, str(exc)
        return True, None

    def decode(
        self,
        video_path: Path,
        pts_ms: Sequence[int],
    ) -> list[DecodedVideoFrame]:
        av = self._import_av()
        results: list[DecodedVideoFrame] = []
        try:
            with av.open(str(video_path), mode="r") as container:
                if not container.streams.video:
                    raise FrameDecodeError(
                        f"video has no video stream: {video_path.name}"
                    )
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                duration_ms = _pyav_duration_ms(container, stream)
                for target_ms in pts_ms:
                    if duration_ms is not None and target_ms > duration_ms:
                        # Region boundaries can come from upstream keyframe
                        # timestamps that slightly exceed the real media
                        # duration; decode the last frame instead of aborting.
                        target_ms = duration_ms
                    frame = self._decode_nearest(container, stream, target_ms)
                    if frame is None:
                        raise FrameDecodeError(
                            f"could not decode a frame near {target_ms}ms from "
                            f"{video_path.name}"
                        )
                    results.append(
                        DecodedVideoFrame(
                            pts_ms=_frame_pts_ms(
                                frame,
                                stream.time_base,
                                getattr(stream, "start_time", None),
                            ),
                            frame_index=None,
                            image=frame.to_ndarray(format="rgb24"),
                        )
                    )
        except (FrameDecodeError, FrameRequestError):
            raise
        except Exception as exc:
            raise FrameDecodeError(
                f"failed to decode {video_path.name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return results

    @staticmethod
    def _decode_nearest(container: Any, stream: Any, target_ms: int) -> Any | None:
        if stream.time_base is None:
            raise FrameDecodeError("video stream does not expose a time base")
        time_base = Fraction(stream.time_base)
        start_ticks = getattr(stream, "start_time", None) or 0
        target_ticks = start_ticks + int(Fraction(target_ms, 1000) / time_base)
        container.seek(
            max(0, target_ticks),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual_ms = _frame_pts_ms(frame, stream.time_base, start_ticks)
            if actual_ms < 0:
                continue
            if actual_ms >= target_ms:
                if previous is None:
                    return frame
                previous_ms = _frame_pts_ms(
                    previous,
                    stream.time_base,
                    start_ticks,
                )
                if target_ms - previous_ms <= actual_ms - target_ms:
                    return previous
                return frame
            previous = frame
        # The nominal duration can be just beyond the final frame PTS.
        return previous

    @staticmethod
    def _import_av() -> Any:
        try:
            import av
        except Exception as exc:  # pragma: no cover - optional dependency
            raise FrameProviderConfigurationError(
                "PyAV is unavailable; install the optional 'av' package to "
                "enable local YouCook2 frame decoding"
            ) from exc
        return av


def _frame_pts_ms(
    frame: Any,
    time_base: Any,
    start_time: int | None = None,
) -> int:
    if frame.pts is None:
        raise FrameDecodeError(
            "decoded frame does not expose a presentation timestamp"
        )
    origin = 0 if start_time is None else int(start_time)
    milliseconds = Fraction(frame.pts - origin) * Fraction(time_base) * 1000
    return (milliseconds.numerator * 2 + milliseconds.denominator) // (
        2 * milliseconds.denominator
    )


def _pyav_duration_ms(container: Any, stream: Any) -> int | None:
    if stream.duration is not None and stream.time_base is not None:
        duration = Fraction(stream.duration) * Fraction(stream.time_base) * 1000
        return (
            duration.numerator + duration.denominator - 1
        ) // duration.denominator
    if container.duration is not None:
        # PyAV exposes container duration in AV_TIME_BASE microseconds.
        return (int(container.duration) + 999) // 1000
    return None
