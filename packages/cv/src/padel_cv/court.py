"""Court geometry: homography projection and court-region logic.

A homography H maps image pixels to real-world court coordinates in meters,
where the court floor is the plane x in [0, 10], y in [0, 20] (net at y=10).
Players are located by the midpoint of their ankle keypoints (following
PadelTracker100's convention), since that is their contact point with the
court plane — only floor points are valid under a homography.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from padel_cv.pipeline import Frame, PoseDetection

HomographyArray = npt.NDArray[np.float64]
"""3x3 projective matrix mapping image pixels to court meters."""

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = 10.0
SERVICE_LINE_FROM_NET_M = 6.95

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
MIN_ANKLE_CONFIDENCE = 0.3

SERVICE_NEAR_Y = NET_Y_M - SERVICE_LINE_FROM_NET_M  # 3.05
SERVICE_FAR_Y = NET_Y_M + SERVICE_LINE_FROM_NET_M  # 16.95

# The 13-point court keypoint schema (ADR-0001): floor-line intersections in
# court coordinates (meters). Only floor points are valid homography anchors.
# Order matters: it defines the keypoint indices of the learned detector.
COURT_KEYPOINT_NAMES: list[str] = [
    "corner_near_left",  # 0  A1
    "corner_near_right",  # 1  A2
    "service_near_left",  # 2  S1
    "service_near_center",  # 3  T1
    "service_near_right",  # 4  S2
    "net_left",  # 5  N1
    "net_center",  # 6  C
    "net_right",  # 7  N2
    "service_far_left",  # 8  S3
    "service_far_center",  # 9  T2
    "service_far_right",  # 10 S4
    "corner_far_left",  # 11 B1
    "corner_far_right",  # 12 B2
]

COURT_KEYPOINTS_M: npt.NDArray[np.float64] = np.array(
    [
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [0.0, SERVICE_NEAR_Y],
        [COURT_WIDTH_M / 2, SERVICE_NEAR_Y],
        [COURT_WIDTH_M, SERVICE_NEAR_Y],
        [0.0, NET_Y_M],
        [COURT_WIDTH_M / 2, NET_Y_M],
        [COURT_WIDTH_M, NET_Y_M],
        [0.0, SERVICE_FAR_Y],
        [COURT_WIDTH_M / 2, SERVICE_FAR_Y],
        [COURT_WIDTH_M, SERVICE_FAR_Y],
        [0.0, COURT_LENGTH_M],
        [COURT_WIDTH_M, COURT_LENGTH_M],
    ]
)

# Mirror pairs for horizontal-flip augmentation (world x -> 10 - x).
COURT_FLIP_IDX: list[int] = [1, 0, 4, 3, 2, 7, 6, 5, 10, 9, 8, 12, 11]


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


def localize_players(frame: Frame) -> None:
    """Project every pose to court coordinates using the frame homography."""
    assert frame.homography is not None
    for pose in frame.poses:
        x_px, y_px = ankle_midpoint(pose)
        x_m, y_m = project_point(frame.homography, x_px, y_px)
        pose.court_position_m = (x_m, y_m)
        pose.on_court = is_on_court(x_m, y_m)


def homography_from_keypoints(
    keypoints_px: npt.NDArray[np.float32],
    min_confidence: float = 0.5,
    min_points: int = 4,
    max_error_m: float = 0.35,
) -> HomographyArray | None:
    """Robust px->m homography from detected court keypoints (13, 3: x, y, conf).

    Uses RANSAC so a few badly-placed points do not corrupt the estimate, then
    gates on mean reprojection error of the inliers: with too few confident
    points or a poor fit it returns None — the pipeline degrades with a warning
    instead of producing invented positions (ADR-0005).
    """
    confident = keypoints_px[:, 2] >= min_confidence
    if confident.sum() < min_points:
        return None
    source = keypoints_px[confident, :2].astype(np.float64)
    target = COURT_KEYPOINTS_M[confident]
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, max_error_m)
    if homography is None or inlier_mask.sum() < min_points:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    ones = np.ones((inliers.sum(), 1))
    projected = (homography @ np.concatenate([source[inliers], ones], axis=1).T).T
    projected = projected[:, :2] / projected[:, 2:3]
    mean_error = float(np.linalg.norm(projected - target[inliers], axis=1).mean())
    if mean_error > max_error_m:
        return None
    return np.asarray(homography, dtype=np.float64)
