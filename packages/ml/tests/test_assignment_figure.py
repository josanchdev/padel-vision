import numpy as np
from padel_ml.thesis.assignment_figure import measure, measured_point, table

from padel_cv.detections import PoseDetection


def _pose(
    player: int, box: tuple[float, float, float, float], wrists: list[tuple[float, float, float]]
) -> PoseDetection:
    keypoints = np.zeros((17, 3), np.float32)
    for index, wrist in zip((9, 10), wrists, strict=False):
        keypoints[index] = wrist
    return PoseDetection(bbox_xyxy=box, confidence=0.9, keypoints=keypoints, player_id=player)


def test_the_point_is_the_closer_confident_wrist() -> None:
    pose = _pose(1, (0, 0, 10, 100), [(5.0, 20.0, 0.9), (50.0, 50.0, 0.9)])
    assert measured_point(pose, (48.0, 48.0)) == (50.0, 50.0)


def test_without_a_confident_wrist_the_point_is_the_box_centre() -> None:
    pose = _pose(1, (0, 0, 10, 100), [(5.0, 20.0, 0.1), (50.0, 50.0, 0.2)])
    assert measured_point(pose, (48.0, 48.0)) == (5.0, 50.0)


def test_heights_divide_the_pixels_by_the_box_height() -> None:
    near = _pose(4, (0, 0, 10, 300), [(0.0, 0.0, 0.9)])
    far = _pose(2, (0, 0, 10, 100), [(0.0, 0.0, 0.9)])
    measures = measure([near, far], (0.0, 120.0))
    assert [m.player for m in measures] == [2, 4]
    assert [round(m.heights, 2) for m in measures] == [1.2, 0.4]


def test_the_table_marks_the_nearest_in_pixels_and_in_heights() -> None:
    near = _pose(4, (0, 0, 10, 300), [(0.0, 0.0, 0.9)])
    far = _pose(2, (0, 0, 10, 100), [(0.0, 0.0, 0.9)])
    rows = table(measure([near, far], (0.0, 90.0))).splitlines()
    far_row = next(row for row in rows if row.startswith("J2"))
    near_row = next(row for row in rows if row.startswith("J4"))
    assert r"\textbf{90}" in far_row and r"\textbf{0,30}" in near_row
