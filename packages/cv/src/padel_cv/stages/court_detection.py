"""Learned court detection stage (ADR-0001): keypoints -> automatic homography.

This is the production replacement for GroundTruthCourtStage: a YOLO pose
model trained on the 13-point court schema detects the visible floor-line
intersections in each frame, and a RANSAC homography maps pixels to court
meters. Works on any video with no per-video calibration (ADR-0005); when too
few points are confidently detected (e.g. court-level views), the frame simply
gets no homography and downstream stages skip their work.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from padel_cv.court import homography_from_keypoints, localize_players
from padel_cv.pipeline import Frame


class CourtDetectionStage:
    """Detects court keypoints and computes the px->m homography per frame."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        # v6 was trained at 1280 and measures best at 1280 inference (0.196 m
        # vs 0.246 m at 1920 on the WPT benchmark); see docs/experiments.md.
        image_size: int = 1280,
        keypoint_confidence: float = 0.5,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._confidence = confidence
        self._image_size = image_size
        self._keypoint_confidence = keypoint_confidence
        self._device = device
        self.frames_without_court = 0

    def process(self, frame: Frame) -> Frame:
        results: list[Any] = self._model.predict(
            frame.image,
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            verbose=False,
        )
        result = results[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            self.frames_without_court += 1
            return frame
        best = int(result.boxes.conf.argmax())
        keypoints = result.keypoints.data[best].cpu().numpy().astype(np.float32)
        homography = homography_from_keypoints(keypoints, min_confidence=self._keypoint_confidence)
        if homography is None:
            self.frames_without_court += 1
            return frame
        frame.homography = homography
        localize_players(frame)
        return frame
