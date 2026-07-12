import numpy as np

from padel_cv.pipeline import Frame, PoseDetection
from padel_cv.visualize import draw_poses


def test_draw_poses_returns_copy_with_annotations() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.full((17, 3), 50.0, dtype=np.float32)
    keypoints[:, 2] = 0.9
    frame = Frame(index=0, timestamp_s=0.0, image=image)
    frame.poses.append(
        PoseDetection(bbox_xyxy=(10.0, 10.0, 90.0, 90.0), confidence=0.8, keypoints=keypoints)
    )

    canvas = draw_poses(frame)

    assert canvas is not frame.image
    assert not np.array_equal(canvas, frame.image)  # something was drawn
    assert np.array_equal(frame.image, np.zeros((100, 100, 3), dtype=np.uint8))  # original intact


def test_draw_poses_skips_low_confidence_keypoints() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.full((17, 3), 50.0, dtype=np.float32)
    keypoints[:, 2] = 0.0  # all keypoints below threshold
    frame = Frame(index=0, timestamp_s=0.0, image=image)
    frame.poses.append(
        PoseDetection(bbox_xyxy=(10.0, 10.0, 90.0, 90.0), confidence=0.8, keypoints=keypoints)
    )

    canvas = draw_poses(frame)

    # Box and label are drawn, but no joints: center of image stays black.
    assert canvas[50, 50].tolist() == [0, 0, 0]
