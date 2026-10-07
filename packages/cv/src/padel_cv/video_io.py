"""Reading one exact frame, and video writing that browsers can actually play.

OpenCV's VideoWriter emits MPEG-4 Part 2 (mp4v/FMP4), which HTML5 <video> does
not decode. This writer encodes H.264 (yuv420p) with a moved MOOV atom
(faststart) via the ffmpeg binary bundled by imageio-ffmpeg, so results play
inline in the browser without any system dependency.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import cast

import cv2
import imageio.v2 as imageio

from padel_cv.arrays import ImageArray


def read_frame(video: Path, index: int) -> ImageArray:
    """Frame `index` of `video` (BGR), read sequentially from the start.

    Seeking in an MP4 lands on the nearest keyframe rather than on `index`, so
    a still meant to match the analysis of that frame is read in order."""
    capture = cv2.VideoCapture(str(video))
    frame = None
    for _ in range(index + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"{video.name} has no frame {index}")
    capture.release()
    return cast(ImageArray, frame)


class H264VideoWriter:
    """Frame sink that accepts BGR frames and writes a browser-ready H.264 mp4."""

    def __init__(self, path: Path, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            str(path),
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-movflags", "+faststart"],
            # imageio's default of 16 stretches 1080p to 1088 rows; 1080 is a
            # multiple of 8, and H.264 in yuv420p only needs even sizes.
            macro_block_size=8,
        )

    def write(self, bgr_frame: ImageArray) -> None:
        self._writer.append_data(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

    def release(self) -> None:
        self._writer.close()

    def __enter__(self) -> H264VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
