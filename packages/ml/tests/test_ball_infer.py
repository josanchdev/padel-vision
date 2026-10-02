import numpy as np
import pytest
import torch
from padel_ml.ball_infer import BallDetector, build_ball_model
from padel_ml.tracknet import TrackNetV2
from padel_ml.tracknet_v3 import TrackNetV3


def _save_ckpt(path) -> None:
    model = TrackNetV2()
    torch.save({"state_dict": model.state_dict(), "model_name": "tracknetv2"}, path)


def test_detector_needs_full_buffer_before_detecting(tmp_path) -> None:
    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    det = BallDetector(ckpt, min_confidence=0.0)  # accept any peak
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    assert det.detect(img, 0) is None  # buffer not full
    assert det.detect(img, 1) is None
    hit = det.detect(img, 2)  # 3rd frame completes the window
    assert hit is not None
    assert 0 <= hit.x_px <= 640 and 0 <= hit.y_px <= 360


def test_reset_empties_the_buffer_between_videos(tmp_path) -> None:
    """After a reset the next video must fill its own buffer, not borrow frames."""
    ckpt = tmp_path / "ball.pt"
    _save_ckpt(ckpt)
    det = BallDetector(ckpt, min_confidence=0.0)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    for i in range(3):
        det.detect(img, i)
    det.reset()
    assert det.detect(img, 0) is None
    assert det.detect(img, 1) is None
    assert det.detect(img, 2) is not None


def test_ball_models_are_built_by_the_name_in_the_checkpoint() -> None:
    assert isinstance(build_ball_model("tracknetv2"), TrackNetV2)
    assert isinstance(build_ball_model("tracknetv3"), TrackNetV3)
    with pytest.raises(ValueError):
        build_ball_model("yolo")
