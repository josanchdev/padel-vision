import numpy as np

from padel_cv.court import COURT_KEYPOINTS_M, homography_from_keypoints, project_point
from padel_cv.datasets import project_court_points_to_pixels, yolo_pose_label

# A plausible px->m homography: meters = (pixels - offset) / scale.
H_TRUE = np.array([[0.01, 0.0, -1.0], [0.0, 0.02, -2.0], [0.0, 0.0, 1.0]])


def perfect_keypoints(confidence: float = 0.9) -> np.ndarray:
    pixels = project_court_points_to_pixels(H_TRUE)
    return np.concatenate([pixels, np.full((len(pixels), 1), confidence)], axis=1).astype(
        np.float32
    )


def test_project_court_points_roundtrip() -> None:
    pixels = project_court_points_to_pixels(H_TRUE)
    for (x_px, y_px), (x_m, y_m) in zip(pixels, COURT_KEYPOINTS_M, strict=True):
        np.testing.assert_allclose(project_point(H_TRUE, x_px, y_px), (x_m, y_m), atol=1e-9)


def test_homography_recovered_from_perfect_points() -> None:
    homography = homography_from_keypoints(perfect_keypoints())
    assert homography is not None
    for (x_px, y_px), (x_m, y_m) in zip(
        project_court_points_to_pixels(H_TRUE), COURT_KEYPOINTS_M, strict=True
    ):
        np.testing.assert_allclose(project_point(homography, x_px, y_px), (x_m, y_m), atol=1e-3)


def test_homography_survives_one_outlier_via_ransac() -> None:
    keypoints = perfect_keypoints()
    keypoints[6, :2] += 300.0  # net_center wildly wrong
    homography = homography_from_keypoints(keypoints)
    assert homography is not None
    x_px, y_px = project_court_points_to_pixels(H_TRUE)[0]
    np.testing.assert_allclose(project_point(homography, x_px, y_px), (0.0, 0.0), atol=0.05)


def test_homography_refused_with_too_few_confident_points() -> None:
    keypoints = perfect_keypoints(confidence=0.1)
    keypoints[:3, 2] = 0.9  # only 3 confident points
    assert homography_from_keypoints(keypoints) is None


def test_yolo_pose_label_marks_out_of_image_points_invisible() -> None:
    pixels = project_court_points_to_pixels(H_TRUE)
    label = yolo_pose_label(pixels, width=800, height=2500)
    assert label is not None
    fields = label.split()
    assert fields[0] == "0"
    kpt_fields = fields[5:]
    visibilities = [int(kpt_fields[i * 3 + 2]) for i in range(13)]
    # x range of projected points is [100, 1100]: right-side points fall outside
    # width=800 and must be flagged invisible.
    assert 0 in visibilities and 2 in visibilities
    assert len(visibilities) == 13


def test_yolo_pose_label_none_when_court_not_visible() -> None:
    pixels = project_court_points_to_pixels(H_TRUE)
    assert yolo_pose_label(pixels, width=50, height=50) is None
