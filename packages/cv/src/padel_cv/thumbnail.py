"""Extract a thumbnail frame from a video for the dashboard match cards.

A still from the middle of the (processed) video makes each match card
recognisable at a glance — skeletons and overlays included, so the card already
shows what the analysis found. Written as a JPEG next to the structured data.
"""

from __future__ import annotations

from pathlib import Path

import cv2


def write_thumbnail(video_path: Path, output_path: Path, width: int = 480) -> Path | None:
    """Grab a mid-video frame, downscale to `width`, save as JPEG.

    Returns the output path, or None if the video could not be read.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None

    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return output_path
