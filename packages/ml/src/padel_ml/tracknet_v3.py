"""TrackNetV3: TrackNetV2 plus a trajectory-refinement head.

V2 predicts each frame's ball heatmap independently. Its weakness is occlusion:
when a player or the glass hides the ball, V2 has nothing to output. V3's idea is
to *refine* V2's coarse heatmap using the surrounding motion context, recovering
the ball where a single-frame view fails (ADR-0009). The hypothesis we test is
that V3 wins specifically on the frames PadelTracker100 flags as ``occluded``.

Here the refiner is a small residual module fed the concatenation of V2's
heatmap and the (downsampled) input frames: it learns a correction added back to
V2's output. Kept compact on purpose to stay comparable to V2 and cheap to train
on this box. The comparison V2-vs-V3, same protocol, is the scientific core.
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
