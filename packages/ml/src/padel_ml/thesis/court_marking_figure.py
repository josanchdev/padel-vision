"""The court marking figure of section 4.1: the six clicks and the lines they give.

The frame a court was marked on, with the six points saved in its court file
drawn as the marking tool draws them (`padel_cv.court_annotator`): a numbered
cross per point, in click order, joined by the court outline and the net. Over
them, the court lines that the homography fitted to those six points projects
back onto the image: the two service lines, which the tool also draws to check
the clicks, and the centre line between them, which completes the T on each
side. If the six points are right, the projected lines fall on the painted ones.

Drawn at the frame's own resolution, with thicker strokes than the tool's 1600 px
window, so that they stay visible at the width of the text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from padel_cv.arrays import ImageArray
from padel_cv.court import COURT_LENGTH_M, COURT_WIDTH_M
from padel_cv.court_annotator import OUTLINE, homography_from_points
from padel_cv.paths import REPO

SERVICE_LINE_M = 3.05
"""Distance from each back wall to its service line, as in the marking tool."""
CLICKED = (60, 240, 60)
"""The tool's green (BGR) for the clicked points, the outline and the net."""
PROJECTED = (255, 200, 0)
"""The tool's colour (BGR) for the lines projected from the homography."""
LINE_WIDTH = 4
MARKER_SIZE = 40
MARKER_RADIUS = 20
LABEL_SCALE = 1.4

Point = tuple[int, int]


@dataclass(frozen=True)
class MarkedCourt:
    """What a court file keeps: where the frame came from and the six clicks."""

    video: Path
    frame_index: int
    points_px: npt.NDArray[np.float64]


def load_marked_court(path: Path) -> MarkedCourt:
    data = json.loads(path.read_text())
    return MarkedCourt(
        video=REPO / data["source_video"],
        frame_index=int(data["frame_index"]),
        points_px=np.array(data["points_px"], dtype=np.float64),
    )


def court_lines_m() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The painted lines the homography is checked against, in metres.

    The two service lines, wall to wall, and the centre line from one service
    line to the other, which crosses the net and forms the T on each side.
    """
    far_service = COURT_LENGTH_M - SERVICE_LINE_M
    middle = COURT_WIDTH_M / 2
    return [
        ((0.0, SERVICE_LINE_M), (COURT_WIDTH_M, SERVICE_LINE_M)),
        ((0.0, far_service), (COURT_WIDTH_M, far_service)),
        ((middle, SERVICE_LINE_M), (middle, far_service)),
    ]


def to_image(metres_to_px: npt.NDArray[np.float64], point_m: tuple[float, float]) -> Point:
    """Where a court point (metres) lands in the image."""
    x, y, w = metres_to_px @ np.array([point_m[0], point_m[1], 1.0])
    return round(x / w), round(y / w)


def _label(canvas: ImageArray, text: str, origin: Point) -> None:
    """A number in the tool's green with a dark outline, readable over any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, origin, font, LABEL_SCALE, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, font, LABEL_SCALE, CLICKED, 3, cv2.LINE_AA)


def marking_view(image: ImageArray, points_px: npt.NDArray[np.float64]) -> ImageArray:
    """The frame with the six clicks, the outline and the projected court lines."""
    canvas = image.copy()
    metres_to_px = np.asarray(np.linalg.inv(homography_from_points(points_px)), np.float64)
    for start, end in court_lines_m():
        start_px, end_px = to_image(metres_to_px, start), to_image(metres_to_px, end)
        cv2.line(canvas, start_px, end_px, PROJECTED, LINE_WIDTH, cv2.LINE_AA)

    points: list[Point] = [(round(x), round(y)) for x, y in points_px]
    for a, b in OUTLINE:
        cv2.line(canvas, points[a], points[b], CLICKED, LINE_WIDTH, cv2.LINE_AA)
    for number, (x, y) in enumerate(points, start=1):
        cv2.drawMarker(
            canvas, (x, y), CLICKED, cv2.MARKER_CROSS, MARKER_SIZE, LINE_WIDTH, cv2.LINE_AA
        )
        cv2.circle(canvas, (x, y), MARKER_RADIUS, CLICKED, LINE_WIDTH, cv2.LINE_AA)
        _label(canvas, str(number), (x + 26, y - 20))
    return canvas
