"""Core per-frame data types.

A ``Frame`` carries one video frame and what has been found in it. A step that
works frame by frame implements ``PipelineStage``: it receives a ``Frame``,
enriches it with its own results, and returns it; if it needs temporal context
it keeps that state internally (see ``stages.pose.PlayerPoseStage``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]
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


@dataclass
class Frame:
    """A single video frame plus everything the pipeline has learned about it."""

    index: int
    timestamp_s: float
    image: ImageArray
    poses: list[PoseDetection] = field(default_factory=list)


@runtime_checkable
class PipelineStage(Protocol):
    """Common interface for every step of the analysis pipeline."""

    def process(self, frame: Frame) -> Frame:
        """Enrich ``frame`` with this stage's results and return it."""
        ...
