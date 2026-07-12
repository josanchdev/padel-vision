"""Player detection + pose estimation stage (YOLO pose, ADR-0003)."""

from __future__ import annotations

from typing import Any

import numpy as np

from padel_cv.pipeline import Frame, PoseDetection

DEFAULT_MODEL = "yolo26n-pose.pt"


class PlayerPoseStage:
    """Detects people and their COCO-17 skeletons in each frame.

    Uses pretrained COCO weights: no padel-specific training yet. Filtering
    detections down to the four actual players (vs spectators/referee) is a
    later concern that will use the court region once court detection exists.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence: float = 0.4,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._confidence = confidence
        self._device = device

    def process(self, frame: Frame) -> Frame:
        results: list[Any] = self._model.predict(
            frame.image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return frame
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy().astype(np.float32)
        for box, conf, kpts in zip(boxes, confidences, keypoints, strict=True):
            frame.poses.append(
                PoseDetection(
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                    keypoints=kpts,
                )
            )
        return frame
