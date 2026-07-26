"""TrackNetV2: a U-Net that predicts a ball heatmap from 3 stacked frames.

Input (N, 9, H, W) = 3 consecutive RGB frames; output (N, 1, H/4, W/4) heatmap
whose peak is the ball. The movement between the three frames is what lets the
net find an ~8px ball the background hides (ADR-0009).

Structure: a VGG-style encoder (double-conv + maxpool, three downsampling
stages) and a decoder that upsamples back to quarter resolution, with skip
connections carrying fine detail across. The head is a 1x1 conv + sigmoid, so
the output is a per-pixel ball probability. This is the citable baseline against
which TrackNetV3 is measured.
"""

from __future__ import annotations

import torch
from torch import nn


def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 conv-BN-ReLU layers, the VGG/U-Net building block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TrackNetV2(nn.Module):
    """9-channel frame stack -> single-channel ball heatmap at quarter res."""

    def __init__(self, in_channels: int = 9) -> None:
        super().__init__()
        self.enc1 = double_conv(in_channels, 32)  # full res
        self.enc2 = double_conv(32, 64)  # 1/2
        self.enc3 = double_conv(64, 128)  # 1/4
        self.bottleneck = double_conv(128, 256)  # 1/8
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # ->1/4
        self.dec3 = double_conv(256, 128)  # 128 up + 128 skip
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # (N, 32, H, W)
        e2 = self.enc2(self.pool(e1))  # (N, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (N, 128, H/4, W/4)
        b = self.bottleneck(self.pool(e3))  # (N, 256, H/8, W/8)

        d3 = self.up3(b)  # (N, 128, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # merge skip at 1/4 res
        logits = self.head(d3)  # (N, 1, H/4, W/4)
        return torch.sigmoid(logits)
