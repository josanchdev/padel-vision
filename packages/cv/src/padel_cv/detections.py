"""What the detectors find in a frame: people with their skeletons, and the ball."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

KeypointArray = npt.NDArray[np.float32]
"""Array of shape (17, 3): COCO keypoints as (x, y, confidence) rows."""


@dataclass
class PoseDetection:
    """One detected person: bounding box, skeleton keypoints and confidence."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: KeypointArray
    track_id: int | None = None
    court_position_m: tuple[float, float] | None = None
    on_court: bool | None = None
    player_id: int | None = None
    """Stable player slot 1-4 (1-2 far half, 3-4 near the camera); None if unassigned."""


@dataclass
class BallDetection:
    """The ball's location in one frame, as found by the ball detector.

    image_xy is the pixel position of the heatmap peak; confidence is the peak
    height in [0, 1]. Our own TrackNet model fills this — no external annotation
    at inference.
    """

    image_xy: tuple[float, float]
    confidence: float
