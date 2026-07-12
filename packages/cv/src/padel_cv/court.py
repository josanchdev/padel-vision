"""Court geometry: homography projection and court-region logic.

A homography H maps image pixels to real-world court coordinates in meters,
where the court floor is the plane x in [0, 10], y in [0, 20] (net at y=10).
Players are located by the midpoint of their ankle keypoints (following
PadelTracker100's convention), since that is their contact point with the
court plane — only floor points are valid under a homography.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from padel_cv.pipeline import PoseDetection

HomographyArray = npt.NDArray[np.float64]
"""3x3 projective matrix mapping image pixels to court meters."""

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = 10.0
SERVICE_LINE_FROM_NET_M = 6.95

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
MIN_ANKLE_CONFIDENCE = 0.3


def project_point(homography: HomographyArray, x: float, y: float) -> tuple[float, float]:
    """Project one image point (pixels) to court coordinates (meters)."""
    vector = homography @ np.array([x, y, 1.0])
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


def ankle_midpoint(pose: PoseDetection) -> tuple[float, float]:
    """Ground-contact point of a player in image pixels.

    Midpoint of both ankles when they are confidently detected; otherwise the
    bottom-center of the bounding box as a fallback.
    """
    ankles = pose.keypoints[[LEFT_ANKLE, RIGHT_ANKLE]]
    visible = ankles[ankles[:, 2] >= MIN_ANKLE_CONFIDENCE]
    if len(visible) > 0:
        return float(visible[:, 0].mean()), float(visible[:, 1].mean())
    x1, _, x2, y2 = pose.bbox_xyxy
    return (x1 + x2) / 2, y2


def is_on_court(x_m: float, y_m: float, margin_m: float = 0.5) -> bool:
    """Whether a court-coordinate position lies inside the court (with margin)."""
    return (
        -margin_m <= x_m <= COURT_WIDTH_M + margin_m
        and -margin_m <= y_m <= COURT_LENGTH_M + margin_m
    )
