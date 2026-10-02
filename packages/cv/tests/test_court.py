import numpy as np

from padel_cv.court import ankle_midpoint, is_on_court, localize_pose, project_point
from padel_cv.pipeline import PoseDetection


def make_pose(ankle_confidence: float = 0.9) -> PoseDetection:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[15] = [100.0, 200.0, ankle_confidence]  # left ankle
    keypoints[16] = [120.0, 210.0, ankle_confidence]  # right ankle
    return PoseDetection(bbox_xyxy=(90.0, 50.0, 130.0, 215.0), confidence=0.9, keypoints=keypoints)


def test_project_point_identity() -> None:
    assert project_point(np.eye(3), 3.0, 7.0) == (3.0, 7.0)


def test_project_point_scale() -> None:
    scale = np.diag([0.5, 0.5, 1.0])
    assert project_point(scale, 10.0, 20.0) == (5.0, 10.0)


def test_ankle_midpoint_uses_ankles_when_visible() -> None:
    assert ankle_midpoint(make_pose()) == (110.0, 205.0)


def test_ankle_midpoint_falls_back_to_bbox_bottom() -> None:
    pose = make_pose(ankle_confidence=0.0)
    assert ankle_midpoint(pose) == (110.0, 215.0)


def test_is_on_court_bounds() -> None:
    assert is_on_court(5.0, 10.0)
    assert is_on_court(-0.3, 20.2)  # within margin
    assert not is_on_court(15.0, 10.0)
    assert not is_on_court(5.0, 25.0)


def test_localize_pose_projects_and_flags_on_court() -> None:
    homography = np.diag([0.01, 0.01, 1.0])  # pixels -> metres by dividing by 100
    on_court = make_pose()  # ankles midpoint (110, 205) px -> (1.1, 2.05) m
    off_court = make_pose()
    off_court.keypoints[15] = [1800.0, 300.0, 0.9]  # -> (18.3, 3.0) m: outside
    off_court.keypoints[16] = [1860.0, 300.0, 0.9]

    localize_pose(homography, on_court)
    localize_pose(homography, off_court)

    assert on_court.court_position_m == (1.1, 2.05)
    assert on_court.on_court is True
    assert off_court.on_court is False
