import json
from pathlib import Path

import numpy as np
import pytest

from padel_cv.datasets import project_court_points_to_pixels
from padel_cv.pipeline import Frame, PoseDetection
from padel_cv.stages import StaticCourtStage

# Camera-like homography: near half at the bottom of the image.
H_TRUE = np.array([[0.01, 0.0, -1.0], [0.0, -0.02, 20.0], [0.0, 0.0, 1.0]])


def write_coco(tmp_path: Path, camera: str, visibilities: list[int] | None = None) -> Path:
    pixels = project_court_points_to_pixels(H_TRUE)
    vis = visibilities or [2] * 13
    keypoints = []
    for (x, y), v in zip(pixels, vis, strict=True):
        keypoints += [float(x), float(y), v]
    coco = {
        "images": [{"id": 1, "file_name": f"{camera}_000123.png", "width": 1920, "height": 1080}],
        "annotations": [{"id": 1, "image_id": 1, "keypoints": keypoints}],
        "categories": [{"name": "court"}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    return path


def make_pose(x_px: float, y_px: float) -> PoseDetection:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[15] = [x_px - 5, y_px, 0.9]
    keypoints[16] = [x_px + 5, y_px, 0.9]
    return PoseDetection(bbox_xyxy=(0, 0, 10, 10), confidence=0.9, keypoints=keypoints)


def test_static_stage_localizes_players_with_exact_homography(tmp_path: Path) -> None:
    stage = StaticCourtStage(write_coco(tmp_path, "urjc_cam"), "urjc_cam")
    frame = Frame(index=7, timestamp_s=0.2, image=np.zeros((2, 2, 3), np.uint8))
    # A player standing at court point (5 m, 5 m): pixels via H_TRUE inverse.
    x_px = (5.0 + 1.0) / 0.01
    y_px = (5.0 - 20.0) / -0.02
    frame.poses = [make_pose(x_px, y_px)]
    frame = stage.process(frame)
    assert frame.homography is not None
    position = frame.poses[0].court_position_m
    assert position is not None
    np.testing.assert_allclose(position, (5.0, 5.0), atol=0.05)
    assert frame.poses[0].on_court is True


def test_static_stage_rejects_insufficient_annotation(tmp_path: Path) -> None:
    coco = write_coco(tmp_path, "cam", visibilities=[2] * 4 + [0] * 9)
    with pytest.raises(ValueError, match="not enough annotated points"):
        StaticCourtStage(coco, "cam")


def test_static_stage_unknown_camera(tmp_path: Path) -> None:
    coco = write_coco(tmp_path, "cam_a")
    with pytest.raises(ValueError, match="no annotation found"):
        StaticCourtStage(coco, "cam_b")
