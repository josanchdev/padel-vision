"""PoseConv3D: a 3D CNN over skeleton heatmap volumes.

Input: (N, V=17, T, H, W) heatmap volumes (joints as channels). A small 3D
ResNet-style stack pools over space and time to a class logit. Compact on
purpose for the modest dataset. This is the modern counterpart to the ST-GCN
baseline; the comparison of the two is the scientific core (ADR-0008).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int] = (1, 1, 1)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        if in_ch == out_ch and stride == (1, 1, 1):
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.relu(self.net(x) + self.skip(x)))


class PoseConv3D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 17) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            Conv3dBlock(32, 32),
            Conv3dBlock(32, 64, stride=(2, 2, 2)),
            Conv3dBlock(64, 128, stride=(2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return cast(torch.Tensor, self.head(x))
