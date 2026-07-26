import numpy as np
import torch
from padel_ml.heatmap import batch_to_heatmaps, clip_to_heatmap
from padel_ml.poseconv3d import PoseConv3D


def test_clip_to_heatmap_shape_and_peak() -> None:
    clip = np.zeros((3, 8, 17), dtype=np.float32)
    clip[0, :, 5] = 0.0  # joint 5 x = 0 (origin)
    clip[1, :, 5] = 0.0  # y = 0
    clip[2, :, 5] = 1.0  # full confidence
    volume = clip_to_heatmap(clip, size=32, sigma=1.5)
    assert volume.shape == (17, 8, 32, 32)
    # A blob at the origin peaks at the grid center for joint 5.
    peak = volume[5, 0].argmax()
    row, col = divmod(int(peak), 32)
    assert 14 <= row <= 17 and 14 <= col <= 17


def test_heatmap_zero_confidence_is_empty() -> None:
    clip = np.zeros((3, 4, 17), dtype=np.float32)  # all confidence 0
    volume = clip_to_heatmap(clip, size=16)
    assert float(volume.sum()) == 0.0


def test_batch_to_heatmaps_shape() -> None:
    clips = torch.randn(2, 3, 8, 17)
    clips[:, 2] = 1.0  # confidence channel
    volumes = batch_to_heatmaps(clips, size=16)
    assert volumes.shape == (2, 17, 8, 16, 16)


def test_poseconv3d_forward_and_backward() -> None:
    model = PoseConv3D(num_classes=6)
    x = torch.randn(2, 17, 8, 16, 16)
    out = model(x)
    assert out.shape == (2, 6)
    torch.nn.functional.cross_entropy(out, torch.tensor([0, 3])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
