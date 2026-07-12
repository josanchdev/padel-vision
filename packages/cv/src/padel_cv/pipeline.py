"""Core pipeline abstractions.

Every processing step (court detection, player pose, shot classification, ...)
implements ``PipelineStage``: it receives a ``Frame``, enriches it with its own
results, and returns it. A ``Pipeline`` is just an ordered list of stages applied
to each frame of a video. Stages that need temporal context (e.g. a shot
classifier looking at a window of poses) keep that state internally.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

import cv2
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
    """Stable player slot 1-4 (1-2 near half, 3-4 far half); None if unassigned."""


@dataclass
class Frame:
    """A single video frame plus everything the pipeline has learned about it."""

    index: int
    timestamp_s: float
    image: ImageArray
    poses: list[PoseDetection] = field(default_factory=list)
    homography: npt.NDArray[np.float64] | None = None


@runtime_checkable
class PipelineStage(Protocol):
    """Common interface for every step of the analysis pipeline."""

    def process(self, frame: Frame) -> Frame:
        """Enrich ``frame`` with this stage's results and return it."""
        ...


class Pipeline:
    """Applies an ordered list of stages to every frame of a video."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        self.stages = stages

    def process_frame(self, frame: Frame) -> Frame:
        for stage in self.stages:
            frame = stage.process(frame)
        return frame

    def run(self, video_path: str) -> Iterator[Frame]:
        """Yield processed frames from a video file, one at a time."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        fps: float = capture.get(cv2.CAP_PROP_FPS) or 30.0
        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frame = Frame(index=index, timestamp_s=index / fps, image=cast(ImageArray, image))
                yield self.process_frame(frame)
                index += 1
        finally:
            capture.release()
