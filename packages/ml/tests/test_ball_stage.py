import numpy as np
import torch
from padel_ml.tracknet import TrackNetV2

from padel_cv.pipeline import Frame


def _save_ckpt(path) -> None:
    model = TrackNetV2()
    torch.save({"state_dict": model.state_dict(), "model_name": "tracknetv2"}, path)


def _frame(i: int, homography=None) -> Frame:
    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    return Frame(index=i, timestamp_s=i / 30, image=img, homography=homography)


def test_no_detection_before_buffer_fills(tmp_path) -> None:
    from padel_ml.ball_stage import BallDetectionStage

    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    stage = BallDetectionStage(ckpt, min_confidence=0.0)  # accept any peak

    # First two frames: buffer not full (needs 3), so no ball.
    assert stage.process(_frame(0)).ball is None
    assert stage.process(_frame(1)).ball is None
    # Third frame completes the window -> a detection is emitted.
    assert stage.process(_frame(2)).ball is not None


def test_detection_respects_confidence_threshold(tmp_path) -> None:
    from padel_ml.ball_stage import BallDetectionStage

    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    # Untrained sigmoid output is ~0.5; a threshold above 1 can never fire.
    stage = BallDetectionStage(ckpt, min_confidence=1.01)
    for i in range(3):
        out = stage.process(_frame(i))
    assert out.ball is None


def test_detection_projects_to_court_with_homography(tmp_path) -> None:
    from padel_ml.ball_stage import BallDetectionStage

    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    stage = BallDetectionStage(ckpt, min_confidence=0.0)
    identity = np.eye(3, dtype=np.float64)  # trivial homography for the test
    out = None
    for i in range(3):
        out = stage.process(_frame(i, homography=identity))
    assert out.ball is not None
    assert out.ball.court_xy_m is not None
    # Under an identity homography, court coords equal image pixels.
    assert out.ball.court_xy_m == out.ball.image_xy


def test_image_peak_is_within_frame(tmp_path) -> None:
    from padel_ml.ball_stage import BallDetectionStage

    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    stage = BallDetectionStage(ckpt, min_confidence=0.0)
    out = None
    for i in range(3):
        out = stage.process(_frame(i))
    assert out.ball is not None
    x, y = out.ball.image_xy
    assert 0 <= x <= 1920 and 0 <= y <= 1080
