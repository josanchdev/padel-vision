import numpy as np

from padel_cv.court import COURT_KEYPOINTS_M, homography_from_keypoints, project_point
from padel_cv.datasets import project_court_points_to_pixels, yolo_pose_label

# A plausible camera-like px->m homography: image y grows downward, so the
# near half (small court y) sits at the BOTTOM of the image.
H_TRUE = np.array([[0.01, 0.0, -1.0], [0.0, -0.02, 20.0], [0.0, 0.0, 1.0]])
# Same court seen with the opposite end as y=0: near half at the TOP.
H_FLIPPED = np.array([[0.01, 0.0, -1.0], [0.0, 0.02, -2.0], [0.0, 0.0, 1.0]])


def perfect_keypoints(confidence: float = 0.9) -> np.ndarray:
    pixels = project_court_points_to_pixels(H_TRUE)
    return np.concatenate([pixels, np.full((len(pixels), 1), confidence)], axis=1).astype(
        np.float32
    )


def test_project_court_points_roundtrip() -> None:
    pixels = project_court_points_to_pixels(H_TRUE)
    for (x_px, y_px), (x_m, y_m) in zip(pixels, COURT_KEYPOINTS_M, strict=True):
        np.testing.assert_allclose(project_point(H_TRUE, x_px, y_px), (x_m, y_m), atol=1e-9)


def test_project_court_points_normalizes_orientation() -> None:
    """With a GT that puts "near" at the top, labels must be flipped 180°."""
    pixels = project_court_points_to_pixels(H_FLIPPED)
    # Index 0 (corner_near_left) must be at the bottom of the image and map to
    # the OTHER end of the court under the flipped homography.
    assert pixels[0, 1] > pixels[12, 1]  # near point below far point
    x_m, y_m = project_point(H_FLIPPED, pixels[0, 0], pixels[0, 1])
    np.testing.assert_allclose((x_m, y_m), (10.0, 20.0), atol=1e-9)  # corner_far_right in world


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
