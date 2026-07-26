import torch
from padel_ml.ball_dataset import GRID_H, GRID_W
from padel_ml.tracknet import TrackNetV2
from padel_ml.tracknet_v3 import TrackNetV3

from padel_cv.ball_cache import FRAME_H, FRAME_W


def test_v2_outputs_quarter_res_probability_map() -> None:
    model = TrackNetV2()
    x = torch.rand(2, 9, FRAME_H, FRAME_W)
    y = model(x)
    assert y.shape == (2, 1, GRID_H, GRID_W)
    assert (y >= 0).all() and (y <= 1).all()  # sigmoid output


def test_v3_matches_v2_output_shape() -> None:
    model = TrackNetV3()
    x = torch.rand(2, 9, FRAME_H, FRAME_W)
    y = model(x)
    assert y.shape == (2, 1, GRID_H, GRID_W)
    assert (y >= 0).all() and (y <= 1).all()


def test_models_are_differentiable() -> None:
    for model in (TrackNetV2(), TrackNetV3()):
        x = torch.rand(1, 9, FRAME_H, FRAME_W, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None
