"""Court stage for fixed cameras: exact homography from a one-time annotation.

For a camera on a tripod the court sits at constant pixels for the whole
recording (measured drift < 1 px over 73 min on PADELVIC), so the homography
computed once from a manual 13-point annotation is exact for every frame — no
detector, no per-frame inference cost, no model error. This is the intended
deployment for the URJC court: annotate their camera once, use forever.

The learned detector (CourtDetectionStage) remains the path for arbitrary or
moving viewpoints (broadcast footage, unknown videos). See ADR-0006.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from padel_cv.court import HomographyArray, homography_from_keypoints, localize_players
from padel_cv.pipeline import Frame


def load_camera_annotation(coco_json: str | Path, camera: str) -> np.ndarray:
    """Canonical (13, 3) court points for a camera from a CVAT COCO export.

    Points marked outside get confidence 0 so the homography ignores them.
    """
    with open(coco_json) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    for annotation in coco["annotations"]:
        file_name = images_by_id[annotation["image_id"]]["file_name"]
        if file_name.rsplit("_", 1)[0] == camera:
            raw = np.array(annotation["keypoints"], dtype=np.float32).reshape(13, 3)
            points = raw.copy()
            points[:, 2] = np.where(raw[:, 2] > 0, 1.0, 0.0)
            return points
    raise ValueError(f"no annotation found for camera {camera!r} in {coco_json}")


class StaticCourtStage:
    """Applies a constant, annotation-derived homography to every frame."""

    def __init__(self, coco_json: str | Path, camera: str) -> None:
        points = load_camera_annotation(coco_json, camera)
        homography: HomographyArray | None = homography_from_keypoints(points)
        if homography is None:
            raise ValueError(
                f"camera {camera!r}: not enough annotated points for a reliable homography"
            )
        self._homography = homography

    def process(self, frame: Frame) -> Frame:
        frame.homography = self._homography
        localize_players(frame)
        return frame
