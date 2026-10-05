"""The ball detector the system uses: the U-Net of `tracknet.py` plus a refiner.

Named "V3" in this project only. It is NOT the published TrackNetV3 (Chen & Wang,
2023), which adds an estimated background, mixup and a second model that repairs
the coordinate trajectory; none of that is here.

The U-Net predicts each frame's ball heatmap independently. Its weakness is
occlusion: when a player or the glass hides the ball, it has nothing to output.
The refiner is a small residual module fed the concatenation of the U-Net's
heatmap and the (downsampled) input frames: it learns a correction added back to
the heatmap (ADR-0009). Kept compact on purpose to stay comparable to the U-Net
and cheap to train on this box.

Whether it recovers occluded balls could not be measured: PadelTracker100's
``occluded`` attribute is false on every annotation, so the two models were
compared on overall metrics only (a tie, see docs/experiments.md); the system
uses this one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import interpolate

from padel_ml.tracknet import TrackNetV2, double_conv


class TrackNetV3(nn.Module):
    """V2 backbone + a residual refiner that corrects occluded predictions."""

    def __init__(self, in_channels: int = 9) -> None:
        super().__init__()
        self.backbone = TrackNetV2(in_channels)
        # Refiner sees V2's heatmap (1ch) + the frame stack downsampled to the
        # heatmap grid (in_channels), and predicts a correction to add back.
        self.refiner = nn.Sequential(
            double_conv(1 + in_channels, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = self.backbone(x)  # (N, 1, H/4, W/4), already sigmoid
        grid_hw = coarse.shape[-2:]
        context = interpolate(x, size=grid_hw, mode="bilinear", align_corners=False)
        correction = self.refiner(torch.cat([coarse, context], dim=1))
        # Refine in logit space so the residual can both add and remove mass,
        # then squash back to a probability.
        return torch.sigmoid(torch.logit(coarse.clamp(1e-4, 1 - 1e-4)) + correction)
