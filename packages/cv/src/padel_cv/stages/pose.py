"""Player detection + pose estimation stage (YOLO pose, ADR-0003)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from padel_cv.pipeline import Frame, PoseDetection

DEFAULT_MODEL = "yolo26n-pose.pt"

# ByteTrack tuned for tiny far-side players (see trackers/padel_bytetrack.yaml).
DEFAULT_TRACKER = str(Path(__file__).resolve().parent.parent / "trackers" / "padel_bytetrack.yaml")


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
        image_size: int = 1920,
        device: str | None = None,
        tracker: str | None = DEFAULT_TRACKER,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._confidence = confidence
        # Far-side players are ~60 px tall in 1080p broadcast footage; at the
        # default 640 inference size they vanish. Full-resolution inference is
        # required to detect all four players.
        self._image_size = image_size
        self._device = device
        # Tracker config ("bytetrack.yaml" / "botsort.yaml") or None to disable
        # tracking and run per-frame detection only.
        self._tracker = tracker

    def process(self, frame: Frame) -> Frame:
        if self._tracker is not None:
            # persist=True keeps tracker state across calls, so track IDs stay
            # stable over the video instead of resetting on every frame.
            results: list[Any] = self._model.track(
                frame.image,
                conf=self._confidence,
                imgsz=self._image_size,
                device=self._device,
                tracker=self._tracker,
                persist=True,
                verbose=False,
            )
        else:
            results = self._model.predict(
                frame.image,
                conf=self._confidence,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )
        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return frame
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy().astype(np.float32)
        track_ids: list[int | None]
        if result.boxes.id is not None:
            track_ids = [int(i) for i in result.boxes.id.cpu().numpy()]
        else:
            track_ids = [None] * len(boxes)
        for box, conf, kpts, track_id in zip(boxes, confidences, keypoints, track_ids, strict=True):
            frame.poses.append(
                PoseDetection(
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                    keypoints=kpts,
                    track_id=track_id,
                )
            )
        return frame
